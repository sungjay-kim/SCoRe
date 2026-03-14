#!/usr/bin/env python3
"""Dump per-frame static (z_t) and fused (mu_t, pc1_only_ema_max) latents for t-SNE.

Output: NPZ with columns:
  fruit_ids: (N,) str
  frame_ids: (N,) str
  gt_volumes: (N,) float  (last-date laser; intermediate from caliper ellipsoid)
  z_static:  (N, D)
  mu_fused:  (N, D)
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "corepp"))

import numpy as np
import pandas as pd
import torch

from corepp.temporal.core import build_encoder, prepare_dataset, run_encoder_forward
from corepp.temporal.utils import TemporalMetadataIndex
from corepp.temporal.test import SimpleFusion


def caliper_ellipsoid_volume_ml(h_mm: float, w_mm: float) -> float:
    """Prolate ellipsoid with axes h, w, w (mm) → volume in mL."""
    return (4.0 / 3.0) * np.pi * (h_mm / 2.0) * (w_mm / 2.0) ** 2 / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.58)
    ap.add_argument("--split", default="test")
    ap.add_argument("--min_n_points", type=int, default=300)
    ap.add_argument("--n_points_cache", default=str(ROOT / "corepp/data/kmelon_centered/n_points_cache.csv"))
    args = ap.parse_args()

    param = json.load(open(args.cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Encoder
    encoder = build_encoder(param, int(param["latent_size"]), device)
    ckpt = torch.load(param["pretrained_encoder"], map_location=device)
    state_dict = ckpt.get("encoder_state_dict", ckpt.get("state_dict", ckpt))
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()

    # PC1 with sign auto-flip
    aniso = np.load(param["aniso_npz_path"])
    pc1 = aniso["v_basis_K2"][0]  # (D,)
    mu_d_size = aniso["mu_d_size_K2"][0]
    if mu_d_size < 0:
        pc1 = -pc1
        print(f"[dump] PC1 sign flipped (mu_d_size_K2[0]={mu_d_size:.4f} < 0)")
    pc1_t = torch.from_numpy(pc1).float()
    fusion = SimpleFusion(int(param["latent_size"]), "pc1_only_ema_max",
                          alpha=args.alpha, pc1_dir=pc1_t).to(device).eval()

    # Dataset
    pretrain = param["pretrain_path"]
    dataset = prepare_dataset(param, pretrain, split=args.split)

    # Filter low-info frames (mirrors test.py default)
    if args.n_points_cache and os.path.exists(args.n_points_cache):
        cache = pd.read_csv(args.n_points_cache)
        n_lookup = {(str(r["fruit_id"]), str(r["date"])): int(r["n_points"]) for _, r in cache.iterrows()}
        orbbec_suf = param.get("orbbec_pcd_suffix", "_preprocessed")
        kept = []
        for p in dataset.files:
            fid = p.split('/')[-4]
            base = os.path.splitext(os.path.basename(p))[0]
            frame = base.replace(orbbec_suf, '') if base.endswith(orbbec_suf) else base
            date = frame.split("_")[0]
            n = n_lookup.get((fid, date), -1)
            if n >= args.min_n_points:
                kept.append(p)
        before = len(dataset.files)
        dataset.files = kept
        print(f"[dump] min_n_points filter: kept {len(kept)}/{before}")

    metadata = TemporalMetadataIndex(
        dataset.files,
        timestamp_csv=param.get("timestamp_csv"),
        min_sequence_length=int(param.get("min_sequence_length", 2)),
        orbbec_suffix=param.get("orbbec_pcd_suffix", "_preprocessed"),
    )

    # Ground truth
    gt_last = pd.read_csv(param["gt_volume_csv"]).set_index("label")
    caliper = pd.read_csv(str(ROOT / "corepp/data/kmelon_centered/field_size_data_v2.csv"))
    caliper["date"] = caliper["date"].astype(str)
    caliper["fruit_id"] = caliper["fruit_id"].astype(str)

    def gt_volume(fruit_id, date, is_last):
        if is_last and fruit_id in gt_last.index:
            try:
                return float(gt_last.loc[fruit_id, "volume_ml"])
            except Exception:
                pass
        sub = caliper[(caliper["fruit_id"] == fruit_id) & (caliper["date"] == date)]
        if len(sub):
            return caliper_ellipsoid_volume_ml(float(sub["height"].iloc[0]), float(sub["width"].iloc[0]))
        return np.nan

    rows = []
    delta_unit = float(param.get("delta_time_unit", 86400.0))
    seqs = list(metadata.iter_sequences(shuffle=False))
    print(f"[dump] {len(seqs)} sequences")

    for seq_idx, sequence in enumerate(seqs):
        hidden = None
        prev_ts = None
        for idx, record in enumerate(sequence):
            item = dataset[record.dataset_index]
            with torch.no_grad():
                z = run_encoder_forward(encoder, item, param, device)  # (1, D)
                curr_ts = record.timestamp
                delta = metadata.compute_delta(prev_ts, curr_ts, unit=delta_unit)
                delta_t = torch.tensor([[delta]], device=device, dtype=z.dtype)
                hidden, _ = fusion(z, hidden, delta_t)
                mu = fusion.get_output(hidden)
            fruit_id = item["fruit_id"]
            frame_id = item["frame_id"]
            if isinstance(fruit_id, (list, tuple)): fruit_id = fruit_id[0]
            if isinstance(frame_id, (list, tuple)): frame_id = frame_id[0]
            is_last = (idx == len(sequence) - 1)
            v = gt_volume(str(fruit_id), str(frame_id), is_last)
            rows.append({
                "fruit_id": str(fruit_id),
                "frame_id": str(frame_id),
                "is_last": bool(is_last),
                "gt_volume": float(v) if v is not None else float("nan"),
                "z_static": z.detach().cpu().numpy().squeeze(0),
                "mu_fused": mu.detach().cpu().numpy().squeeze(0),
            })
            prev_ts = curr_ts
        if (seq_idx + 1) % 20 == 0:
            print(f"  [seq {seq_idx+1}/{len(seqs)}] {len(rows)} frames")

    fruit_ids = np.array([r["fruit_id"] for r in rows])
    frame_ids = np.array([r["frame_id"] for r in rows])
    is_last_arr = np.array([r["is_last"] for r in rows])
    gt_v = np.array([r["gt_volume"] for r in rows], dtype=np.float64)
    z_arr = np.stack([r["z_static"] for r in rows], axis=0)
    mu_arr = np.stack([r["mu_fused"] for r in rows], axis=0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        fruit_ids=fruit_ids, frame_ids=frame_ids, is_last=is_last_arr,
        gt_volumes=gt_v, z_static=z_arr, mu_fused=mu_arr,
    )
    print(f"[dump] saved {len(rows)} frames -> {args.out}")
    print(f"[dump] fruits unique: {len(np.unique(fruit_ids))}")


if __name__ == "__main__":
    main()
