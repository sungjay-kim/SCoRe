#!/usr/bin/env python3
"""Check whether PC1 latent trajectory across time follows logistic growth.

For each fruit in the train/val split, run the frozen encoder on every date's
partial PCD, project the latent onto PC1 (from aniso_npz), fit a 3-param
logistic curve to the (date, PC1) sequence, and plot all fruits in a grid.

Outputs:
  - <out_dir>/pc1_logistic_fit_grid.png   (one subplot per fruit)
  - <out_dir>/pc1_logistic_summary.json   (per-fruit fit RMSE, params)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit

sys.path.insert(0, "/home/mainuser/mount/sungjay/tcore/corepp")
sys.path.insert(0, "/home/mainuser/mount/sungjay/tcore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from corepp.temporal.core import build_encoder, prepare_dataset, run_encoder_forward


def logistic(t, K, r, t0):
    return K / (1.0 + np.exp(-r * (t - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Temporal config (uses pretrained_encoder)")
    ap.add_argument("--checkpoint", required=True, help="Encoder+fusion ckpt (for encoder weights)")
    ap.add_argument("--aniso_npz", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TCORE_PCD_FAST", "1")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    with open(args.cfg) as f:
        param = json.load(f)
    # Resolve experiment dir for encoder/decoder
    exp_dir = param.get("experiment")
    if exp_dir is None:
        raise RuntimeError("config has no 'experiment' field")
    param["pretrain_path"] = os.path.join(exp_dir, "Reconstructions",
                                          str(param.get("checkpoint_decoder", "301")),
                                          "Codes", "complete")
    # latent_size from deepsdf experiment specs (same way test.py resolves it)
    import deepsdf.deep_sdf.workspace as ws
    specs = ws.load_experiment_specifications(exp_dir)
    latent_size = int(specs["CodeLength"])

    # Load encoder
    encoder = build_encoder(param, latent_size, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    print(f"[pc1-check] encoder loaded from {args.checkpoint}")

    # Load PC1 direction
    ad = np.load(args.aniso_npz)
    pc1 = torch.from_numpy(ad["v_basis_K2"][0]).float().to(device)         # (D,)
    pc1 = pc1 / (pc1.norm() + 1e-12)
    print(f"[pc1-check] PC1 loaded: shape={tuple(pc1.shape)}  pc_var_explained[0:3]={ad['pc_var_explained'][:3]}")

    # Load dataset
    dataset = prepare_dataset(param, pretrain=param["pretrain_path"], split=args.split)
    print(f"[pc1-check] dataset n_files={len(dataset.files)}  split={args.split}")

    # Iterate fruit_id → list of (date_int, pc1_scalar)
    from collections import defaultdict
    per_fruit = defaultdict(list)

    # Need fruit_id and date per index
    # parse from filename
    orbbec_suf = param.get("orbbec_pcd_suffix", "_preprocessed")
    print("[pc1-check] running encoder on all frames...")
    with torch.no_grad():
        for i, fpath in enumerate(dataset.files):
            if i % 100 == 0:
                print(f"  {i}/{len(dataset.files)}")
            parts = fpath.split("/")
            fid = parts[-4]
            base = os.path.splitext(os.path.basename(fpath))[0]
            frame = base.replace(orbbec_suf, "")
            try:
                date_str = frame.split("_")[0]
                # YYYYMMDD → days_since_epoch-like int
                import datetime as dt
                d = dt.date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
                # days since 2026-01-01 (arbitrary epoch)
                days = (d - dt.date(2026, 1, 1)).days
            except Exception:
                continue
            item = dataset[i]
            latent = run_encoder_forward(encoder, item, param, device)   # (1, D) or (D,)
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)
            pc1_val = (latent * pc1.unsqueeze(0)).sum(dim=-1).item()
            per_fruit[fid].append((days, pc1_val))

    print(f"[pc1-check] collected {len(per_fruit)} fruits, mean {np.mean([len(v) for v in per_fruit.values()]):.1f} frames/fruit")

    # Per-fruit logistic fit
    summary = {}
    for fid, seq in per_fruit.items():
        seq.sort(key=lambda x: x[0])
        t = np.array([s[0] for s in seq], dtype=float)
        v = np.array([s[1] for s in seq], dtype=float)
        if len(t) < 4:
            summary[fid] = {"n": len(t), "fit": None, "reason": "too few frames"}
            continue
        try:
            # initial guess: K = max(v) * 1.1, r = 0.1, t0 = mean(t)
            p0 = [float(v.max() * 1.1 + 1e-3), 0.1, float(t.mean())]
            popt, _ = curve_fit(logistic, t, v, p0=p0, maxfev=20000)
            pred = logistic(t, *popt)
            rmse = float(np.sqrt(np.mean((pred - v) ** 2)))
            r2 = 1.0 - float(np.sum((v - pred) ** 2) / max(float(np.sum((v - v.mean()) ** 2)), 1e-12))
            summary[fid] = {"n": len(t), "K": float(popt[0]), "r": float(popt[1]),
                            "t0": float(popt[2]), "rmse": rmse, "r2": r2,
                            "v_range": [float(v.min()), float(v.max())],
                            "t_range": [float(t.min()), float(t.max())]}
        except Exception as e:
            summary[fid] = {"n": len(t), "fit": None, "reason": str(e)[:80]}

    # Grid plot
    fids = sorted(per_fruit.keys())
    n = len(fids)
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2 * rows), squeeze=False)
    for ax in axes.flatten():
        ax.axis("off")
    for idx, fid in enumerate(fids):
        ax = axes[idx // cols][idx % cols]
        ax.axis("on")
        seq = sorted(per_fruit[fid], key=lambda x: x[0])
        t = np.array([s[0] for s in seq], dtype=float)
        v = np.array([s[1] for s in seq], dtype=float)
        ax.plot(t, v, "ko", markersize=2.5, label="PC1")
        ax.set_title(fid, fontsize=7)
        ax.tick_params(labelsize=6)
        fit = summary[fid]
        if fit.get("fit") is None:
            continue
        if "K" in fit:
            t_smooth = np.linspace(t.min() - 1, t.max() + 1, 100)
            ax.plot(t_smooth, logistic(t_smooth, fit["K"], fit["r"], fit["t0"]),
                    "r-", linewidth=1.0, alpha=0.7, label=f"logistic r²={fit['r2']:.2f}")
            ax.text(0.05, 0.95, f"r²={fit['r2']:.2f}", transform=ax.transAxes,
                    fontsize=6, va="top", ha="left",
                    bbox=dict(facecolor="white", alpha=0.7, pad=1))
    fig.suptitle(f"PC1 latent trajectory vs logistic fit — split={args.split}, n={n} fruits",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plot_path = out / "pc1_logistic_fit_grid.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[pc1-check] grid plot → {plot_path}")

    # Summary stats
    r2_vals = [s["r2"] for s in summary.values() if "r2" in s]
    rmse_vals = [s["rmse"] for s in summary.values() if "rmse" in s]
    print(f"[pc1-check] {len(r2_vals)}/{len(summary)} fruits fitted successfully")
    if r2_vals:
        print(f"[pc1-check] r²  mean={np.mean(r2_vals):.3f}  median={np.median(r2_vals):.3f}  "
              f"p10={np.percentile(r2_vals, 10):.3f}  p90={np.percentile(r2_vals, 90):.3f}")
        print(f"[pc1-check] rmse mean={np.mean(rmse_vals):.4f}  median={np.median(rmse_vals):.4f}")

    json.dump(summary, open(out / "pc1_logistic_summary.json", "w"), indent=2)
    print(f"[pc1-check] summary → {out/'pc1_logistic_summary.json'}")


if __name__ == "__main__":
    main()
