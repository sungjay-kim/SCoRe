"""Run inference saving full per-frame (z_t, mu_t) latent trajectories.

Lightweight: skips decoder/mesh extraction. Only encoder + fusion forward.
Outputs npz: fruit_id (str), frame_id (str), frame_idx, z_t (D,), mu_t (D,), gate_mean.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / "corepp") not in sys.path:
    sys.path.append(str(ROOT / "corepp"))

from corepp.networks.temporal_models import LatentKalmanFusion
from corepp.temporal.core import build_encoder, prepare_dataset, run_encoder_forward
from corepp.temporal.utils import TemporalMetadataIndex
from corepp.temporal.train import compute_smoothness, compute_pca_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_points_cache", default=None)
    ap.add_argument("--min_n_points", type=int, default=300)
    ap.add_argument("--split", default=None, help="Override eval_split (train / val / test)")
    args = ap.parse_args()

    with open(args.cfg) as f:
        param = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_size = int(param.get("latent_size", 64))

    # Build fusion (Kalman) — only Kalman supported for now
    init_R = float(param.get("kalman_init_R", 0.1))
    growth_mode = param.get("growth_mode", "none")
    fusion = LatentKalmanFusion(
        latent_dim=latent_size,
        quality_mode=param.get("quality_mode", "normalized"),
        init_R=init_R,
        growth_mode=growth_mode,
        robust_gating=bool(param.get("robust_gating", False)),
        mahal_threshold=float(param.get("mahal_threshold", 100.0)),
        warmup_frames=int(param.get("warmup_frames", 3)),
        gate_sharpness=float(param.get("gate_sharpness", 2.0)),
        gate_R_floor=float(param.get("gate_R_floor", 0.01)),
        anisotropic_gating=bool(param.get("anisotropic_gating", False)),
        aniso_npz_path=param.get("aniso_npz_path", None),
        aniso_K=int(param.get("aniso_K", 2)),
        aniso_threshold_kalman=float(param.get("aniso_threshold_kalman", 100.0)),
        aniso_threshold_anchor=float(param.get("aniso_threshold_anchor", 100.0)),
        aniso_use_anchor=bool(param.get("aniso_use_anchor", True)),
    ).to(device)

    encoder = build_encoder(param, latent_size, device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    fusion.load_state_dict(ckpt["fusion_state_dict"])
    encoder.eval()
    fusion.eval()

    split = args.split if args.split else param.get("eval_split", "test")
    print(f"[extract] using split={split}")
    dataset = prepare_dataset(param, pretrain=param["pretrain_path"], split=split)

    # min_n_points filter (same as test.py): need to read each file's PCD count from cache
    if args.n_points_cache and os.path.isfile(args.n_points_cache):
        import pandas as pd
        import re
        cache_df = pd.read_csv(args.n_points_cache)
        cache_map = {(r.fruit_id, str(r.date)): int(r.n_points) for r in cache_df.itertuples()}
        orbbec_suf = param.get("orbbec_pcd_suffix", "_preprocessed")
        def file_key(fpath):
            fruit_id = fpath.split('/')[-4]
            frame_basename = os.path.splitext(os.path.basename(fpath))[0]
            frame_basename = frame_basename.replace(orbbec_suf, '')
            m = re.match(r"^(\d{8})", frame_basename)
            date_str = m.group(1) if m else frame_basename
            return (fruit_id, date_str)
        kept = []
        for fpath in dataset.files:
            n = cache_map.get(file_key(fpath), args.min_n_points)
            if n >= args.min_n_points:
                kept.append(fpath)
        before = len(dataset.files)
        dataset.files = kept
        print(f"[extract] min_n_points={args.min_n_points}: kept {len(kept)}/{before}")

    delta_unit = float(param.get("delta_time_unit", 86400.0))

    # Iterate dataset, collect items grouped by fruit_id (sorted by timestamp)
    print("[extract] grouping items by fruit (this scans dataset once)...")
    items_by_fruit = {}
    for ds_idx in range(len(dataset.files)):
        item = dataset[ds_idx]
        fid = item["fruit_id"]
        if isinstance(fid, (list, tuple)): fid = fid[0]
        items_by_fruit.setdefault(fid, []).append((ds_idx, item))
    for fid in items_by_fruit:
        items_by_fruit[fid].sort(key=lambda x: x[1]["timestamp"] if "timestamp" in x[1] else 0.0)
    print(f"[extract] {len(items_by_fruit)} fruits, {sum(len(v) for v in items_by_fruit.values())} frames")

    # Parse date from frame_id for delta_t (test.py uses item["timestamp"] but dataset doesn't expose it; fall back to date diff)
    import re as _re
    out_rows = []
    with torch.no_grad():
        for fruit_idx, (fruit_id, ds_items) in enumerate(items_by_fruit.items()):
            hidden = None
            prev_date = None
            cumulative_time = 0.0
            running_max_smooth = 0.0
            running_max_pca = 0.0
            for idx, (ds_idx, item) in enumerate(ds_items):
                latent = run_encoder_forward(encoder, item, param, device)  # (1, D)
                z_t = latent.detach().cpu().numpy().reshape(-1)

                # Date from frame_id (YYYYMMDD)
                fr = item.get("frame_id", "")
                if isinstance(fr, (list, tuple)): fr = fr[0]
                m = _re.match(r"^(\d{8})", str(fr))
                if m:
                    from datetime import datetime as _dt
                    cur_date = _dt.strptime(m.group(1), "%Y%m%d")
                    if prev_date is not None:
                        delta = max(0.0, (cur_date - prev_date).days)
                    else:
                        delta = 0.0
                    prev_date = cur_date
                else:
                    delta = 0.0
                cumulative_time += delta
                delta_tensor = torch.tensor([[delta]], device=device, dtype=latent.dtype)
                t_abs_tensor = torch.tensor([[cumulative_time]], device=device, dtype=latent.dtype)

                # Quality: mimic test.py exactly — running_max normalization
                use_quality = bool(param.get("use_quality", True))
                quality = None
                if use_quality and "partial_pcd" in item:
                    qfn = param.get("quality_fn", "smoothness")
                    qmode = param.get("quality_mode", "normalized")
                    if qmode == "raw_exp":
                        if qfn == "pca_trace":
                            q_t = float(compute_pca_trace(item["partial_pcd"]))
                        else:
                            q_t = float(compute_smoothness(item["partial_pcd"]))
                    else:
                        if qfn == "pca_trace":
                            pca_t = float(compute_pca_trace(item["partial_pcd"]))
                            running_max_pca = max(running_max_pca, pca_t)
                            q_t = pca_t / running_max_pca if running_max_pca > 1e-8 else 1.0
                        else:
                            smooth_t = float(compute_smoothness(item["partial_pcd"]))
                            running_max_smooth = max(running_max_smooth, smooth_t)
                            q_t = smooth_t / running_max_smooth if running_max_smooth > 1e-8 else 1.0
                    quality = torch.tensor([[float(q_t)]], device=device, dtype=latent.dtype)

                hidden, gate_vals = fusion(latent, hidden, delta_tensor, quality=quality, t_abs=t_abs_tensor)
                mu_new = fusion.get_output(hidden).detach().cpu().numpy().reshape(-1)
                gate_mean = float(gate_vals.mean().item()) if gate_vals is not None else 1.0

                fr = item.get("frame_id", "")
                if isinstance(fr, (list, tuple)): fr = fr[0]
                out_rows.append({
                    "fruit_id": fruit_id,
                    "frame_id": str(fr),
                    "frame_idx": idx,
                    "seq_len": len(ds_items),
                    "delta_days": delta,
                    "cum_days": cumulative_time,
                    "z_t": z_t,
                    "mu_t": mu_new,
                    "gate_mean": gate_mean,
                })
            if (fruit_idx + 1) % 20 == 0:
                print(f"[extract] {fruit_idx+1}/{len(items_by_fruit)} fruits done", flush=True)

    # Save as npz
    print(f"[extract] saving {len(out_rows)} rows -> {args.output}")
    arr = {
        "fruit_id": np.array([r["fruit_id"] for r in out_rows]),
        "frame_id": np.array([r["frame_id"] for r in out_rows]),
        "frame_idx": np.array([r["frame_idx"] for r in out_rows], dtype=int),
        "seq_len": np.array([r["seq_len"] for r in out_rows], dtype=int),
        "delta_days": np.array([r["delta_days"] for r in out_rows], dtype=float),
        "cum_days": np.array([r["cum_days"] for r in out_rows], dtype=float),
        "z_t": np.stack([r["z_t"] for r in out_rows]),
        "mu_t": np.stack([r["mu_t"] for r in out_rows]),
        "gate_mean": np.array([r["gate_mean"] for r in out_rows], dtype=float),
    }
    np.savez_compressed(args.output, **arr)
    print(f"[extract] done. shapes: z_t={arr['z_t'].shape} mu_t={arr['mu_t'].shape}")


if __name__ == "__main__":
    main()
