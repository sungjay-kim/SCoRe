#!/usr/bin/env python3
"""
Analyze correlation between ||z_t|| (L2 norm of DeepSDF latent codes) and
ground-truth fruit volume across all healthy samples (S1-S70).

For each sample, we:
  1. Load all available orbbec _preprocessed point clouds (multiple timepoints)
  2. Pad/subsample to 1024 points (matching training)
  3. Run through the frozen PointCloudEncoderMaxMean to get 32d latent z_t
  4. Compute ||z_t||_2
  5. Match to GT volume from ground_truth.csv

Reports: per-sample table, Pearson/Spearman correlation using
  - LAST timepoint norm (most mature fruit)
  - MEAN norm across all timepoints
  - FIRST timepoint norm
  - ALL (sample, timepoint) pairs pooled
"""

import sys
import os
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "corepp"))

import csv
import glob
import re
import numpy as np
import torch
from scipy import stats

# We need open3d for PLY loading
import open3d as o3d

from corepp.networks.models import PointCloudEncoderMaxMean

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
CHECKPOINT_PATH = (
    ROOT / "corepp" / "experiments" / "kmelon_latent32"
    / "20260227_104901_finetune_pce_maxmean_orbbec_patch_occ"
    / "logs" / "checkpoints" / "pce_maxmean_finetune_orbbec_patch_occ_best_model.pt"
)
GT_CSV = ROOT / "corepp" / "data" / "3DKmelonDemo" / "ground_truth.csv"
DATA_DIR = ROOT / "corepp" / "data" / "kmelon_centered"
LATENT_DIM = 32
PAD_SIZE = 1024  # match training config input_size

# Healthy samples: S1-S70 (excluding S3, S11, S17, S20, S26, S47 which
# don't exist in the data dir or have no volume)
# We'll just iterate S1-S70 and skip any that don't have data or GT volume.


def load_gt_volumes(csv_path: Path) -> dict:
    """Load label -> volume_ml from ground_truth.csv. Skip empty volumes."""
    volumes = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip()
            vol_str = row["volume_ml"].strip()
            if vol_str:
                volumes[label] = float(vol_str)
    return volumes


def load_and_pad_pcd(ply_path: str, pad_size: int = 1024) -> np.ndarray:
    """Load a PLY point cloud and pad/subsample to exactly pad_size points."""
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    n = pts.shape[0]

    if n == 0:
        return np.zeros((pad_size, 3), dtype=np.float32)
    elif n < pad_size:
        # Upsample with replacement
        indices = np.random.choice(n, size=pad_size, replace=True)
        pts = pts[indices]
    elif n > pad_size:
        # Downsample without replacement
        indices = np.random.choice(n, size=pad_size, replace=False)
        pts = pts[indices]

    return pts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load GT volumes
    gt_volumes = load_gt_volumes(GT_CSV)
    print(f"Loaded {len(gt_volumes)} GT volumes from {GT_CSV}")

    # 2. Build and load encoder
    encoder = PointCloudEncoderMaxMean(in_channels=3, out_channels=LATENT_DIM)
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu")
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder = encoder.to(device)
    encoder.eval()
    print(f"Loaded encoder from {CHECKPOINT_PATH} (epoch {ckpt['epoch']})")

    # 3. Find all healthy samples S1-S70
    # We consider Sx where x is 1..70
    target_labels = [f"S{i}" for i in range(1, 71)]

    results = []  # list of dicts: {label, volume, timepoints: [{frame_id, norm}]}

    np.random.seed(42)  # For reproducible padding

    with torch.no_grad():
        for label in target_labels:
            # Check if sample directory exists
            sample_dir = DATA_DIR / label / "orbbec" / "pointclouds"
            if not sample_dir.exists():
                continue

            # Check if GT volume exists
            if label not in gt_volumes:
                continue
            volume = gt_volumes[label]

            # Find all preprocessed PLY files
            # Multiple variants exist per date: _masked_points_preprocessed,
            # _masked_color_points_preprocessed, _masked_depth_turbo_*_preprocessed
            # We pick ONE per date, preferring _masked_points_preprocessed,
            # then _masked_color_points_preprocessed as fallback.
            all_preprocessed = sorted(glob.glob(str(sample_dir / "*_preprocessed.ply")))

            # Group by date (YYYYMMDD prefix)
            date_pattern = re.compile(r"^(\d{8})_")
            date_to_files = {}
            for f in all_preprocessed:
                fname = os.path.basename(f)
                m = date_pattern.match(fname)
                if m:
                    date = m.group(1)
                    if date not in date_to_files:
                        date_to_files[date] = []
                    date_to_files[date].append(f)

            # For each date, pick the best file
            ply_files = []
            for date in sorted(date_to_files.keys()):
                candidates = date_to_files[date]
                # Priority: _masked_points_preprocessed > _masked_color_points_preprocessed > any other
                chosen = None
                for c in candidates:
                    bn = os.path.basename(c)
                    if bn == f"{date}_masked_points_preprocessed.ply":
                        chosen = c
                        break
                if chosen is None:
                    for c in candidates:
                        bn = os.path.basename(c)
                        if bn == f"{date}_masked_color_points_preprocessed.ply":
                            chosen = c
                            break
                if chosen is None:
                    chosen = candidates[0]  # fallback to first
                ply_files.append(chosen)

            if not ply_files:
                print(f"  WARNING: {label} has no preprocessed PLY files, skipping")
                continue

            timepoints = []
            for ply_path in ply_files:
                # Extract frame_id from filename
                fname = os.path.basename(ply_path)
                frame_id = date_pattern.match(fname).group(1)

                # Load and prepare point cloud
                pts = load_and_pad_pcd(ply_path, PAD_SIZE)

                # Convert to tensor: (1, 3, N) for Conv1d
                pts_tensor = torch.from_numpy(pts).unsqueeze(0).permute(0, 2, 1).contiguous().to(device)

                # Forward through encoder
                z = encoder(pts_tensor)  # (1, 32)
                z_norm = torch.norm(z, p=2).item()

                timepoints.append({"frame_id": frame_id, "norm": z_norm})

            # Sort timepoints by frame_id (date)
            timepoints.sort(key=lambda x: x["frame_id"])

            results.append({
                "label": label,
                "volume": volume,
                "timepoints": timepoints,
            })

    print(f"\n{'='*90}")
    print(f"Processed {len(results)} healthy samples (S1-S70) with both data and GT volume")
    print(f"{'='*90}\n")

    # 4. Print per-sample table
    print(f"{'Sample':<8} {'Volume(ml)':<12} {'#TP':<5} {'First||z||':<12} {'Last||z||':<12} {'Mean||z||':<12} {'Std||z||':<12}")
    print("-" * 90)

    labels_arr = []
    volumes_arr = []
    first_norms = []
    last_norms = []
    mean_norms = []

    # For pooled correlation
    all_norms_pooled = []
    all_volumes_pooled = []

    for r in results:
        norms = [tp["norm"] for tp in r["timepoints"]]
        mean_n = np.mean(norms)
        std_n = np.std(norms)
        first_n = norms[0]
        last_n = norms[-1]

        print(f"{r['label']:<8} {r['volume']:<12.1f} {len(norms):<5} {first_n:<12.4f} {last_n:<12.4f} {mean_n:<12.4f} {std_n:<12.4f}")

        labels_arr.append(r["label"])
        volumes_arr.append(r["volume"])
        first_norms.append(first_n)
        last_norms.append(last_n)
        mean_norms.append(mean_n)

        for n in norms:
            all_norms_pooled.append(n)
            all_volumes_pooled.append(r["volume"])

    volumes_arr = np.array(volumes_arr)
    first_norms = np.array(first_norms)
    last_norms = np.array(last_norms)
    mean_norms = np.array(mean_norms)
    all_norms_pooled = np.array(all_norms_pooled)
    all_volumes_pooled = np.array(all_volumes_pooled)

    # 5. Compute correlations
    print(f"\n{'='*90}")
    print("CORRELATION ANALYSIS: ||z_t||_2 vs GT Volume (ml)")
    print(f"{'='*90}")

    for name, norms, vols in [
        ("LAST timepoint", last_norms, volumes_arr),
        ("FIRST timepoint", first_norms, volumes_arr),
        ("MEAN across timepoints", mean_norms, volumes_arr),
        ("ALL timepoints (pooled)", all_norms_pooled, all_volumes_pooled),
    ]:
        pearson_r, pearson_p = stats.pearsonr(norms, vols)
        spearman_r, spearman_p = stats.spearmanr(norms, vols)
        print(f"\n  {name} (n={len(norms)}):")
        print(f"    Pearson  r = {pearson_r:+.4f}  (p = {pearson_p:.2e})")
        print(f"    Spearman r = {spearman_r:+.4f}  (p = {spearman_p:.2e})")

    # 6. Additional stats
    print(f"\n{'='*90}")
    print("SUMMARY STATISTICS")
    print(f"{'='*90}")
    print(f"  Number of samples:       {len(results)}")
    print(f"  Volume range:            [{volumes_arr.min():.1f}, {volumes_arr.max():.1f}] ml")
    print(f"  Volume mean +/- std:     {volumes_arr.mean():.1f} +/- {volumes_arr.std():.1f} ml")
    print(f"  ||z|| range (last TP):   [{last_norms.min():.4f}, {last_norms.max():.4f}]")
    print(f"  ||z|| mean +/- std:      {last_norms.mean():.4f} +/- {last_norms.std():.4f}")

    # 7. Print timepoint norms for a few samples to show temporal trend
    print(f"\n{'='*90}")
    print("TEMPORAL NORM EVOLUTION (first 5 samples)")
    print(f"{'='*90}")
    for r in results[:5]:
        norms_str = " -> ".join(f"{tp['frame_id']}:{tp['norm']:.4f}" for tp in r["timepoints"])
        print(f"  {r['label']} (vol={r['volume']:.1f}ml): {norms_str}")


if __name__ == "__main__":
    main()
