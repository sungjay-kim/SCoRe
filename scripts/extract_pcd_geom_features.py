"""Extract simple PCD geometric features per frame: PCA eigenvalues, bbox dims.

For testing if PCD physical geometry can serve as a history-free size prior to
catch encoder hallucination ("small fruit predicted large").
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

from corepp.temporal.core import prepare_dataset


def pcd_geom_features(pcd):
    """Compute simple geometric features from a (N, 3) point cloud (numpy or tensor).

    Returns dict with:
      - n_pts: number of points
      - bbox_dims: (3,) — bbox extent along each axis
      - pca_eigvals: (3,) — eigenvalues of covariance, descending
      - pca_axes_lengths: (3,) — 2*sqrt(eigval), interpreted as full axis length
      - ellipsoid_vol_ml: 4/3 π × a × b × c with a,b,c = pca_axes_lengths/2 (i.e., semi-axes)
                         times 1e6 to convert m³ → ml
      - bbox_vol_ml: prod(bbox_dims) * 1e6
    """
    if hasattr(pcd, "detach"):
        pcd = pcd.detach().cpu().numpy()
    pcd = np.asarray(pcd, dtype=np.float64).reshape(-1, 3)

    n = len(pcd)
    if n < 4:
        return dict(n_pts=n, bbox_dims=np.zeros(3), pca_eigvals=np.zeros(3),
                    pca_axes_lengths=np.zeros(3), ellipsoid_vol_ml=0.0, bbox_vol_ml=0.0)

    bbox = pcd.max(axis=0) - pcd.min(axis=0)
    pcd_c = pcd - pcd.mean(axis=0)
    cov = (pcd_c.T @ pcd_c) / max(1, len(pcd_c) - 1)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.clip(eigvals, 0, None)
    axes_full = 2.0 * np.sqrt(eigvals)  # full-axis length = 2σ

    # Volume estimates
    a, b, c = axes_full / 2.0  # semi-axes
    ellipsoid_vol_m3 = (4.0 / 3.0) * np.pi * a * b * c
    bbox_vol_m3 = float(np.prod(bbox))
    return dict(
        n_pts=n,
        bbox_dims=bbox,
        pca_eigvals=eigvals,
        pca_axes_lengths=axes_full,
        ellipsoid_vol_ml=ellipsoid_vol_m3 * 1e6,
        bbox_vol_ml=bbox_vol_m3 * 1e6,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_points_cache", default=None)
    ap.add_argument("--min_n_points", type=int, default=300)
    args = ap.parse_args()

    with open(args.cfg) as f:
        param = json.load(f)
    dataset = prepare_dataset(param, pretrain=param["pretrain_path"], split=param.get("eval_split", "test"))

    if args.n_points_cache and os.path.isfile(args.n_points_cache):
        import pandas as pd
        import re
        cache_df = pd.read_csv(args.n_points_cache)
        cache_map = {(r.fruit_id, str(r.date)): int(r.n_points) for r in cache_df.itertuples()}
        orbbec_suf = param.get("orbbec_pcd_suffix", "_preprocessed")
        def file_key(fpath):
            fruit_id = fpath.split('/')[-4]
            base = os.path.splitext(os.path.basename(fpath))[0].replace(orbbec_suf, '')
            m = re.match(r"^(\d{8})", base)
            return (fruit_id, m.group(1) if m else base)
        kept = [f for f in dataset.files if cache_map.get(file_key(f), args.min_n_points) >= args.min_n_points]
        before = len(dataset.files); dataset.files = kept
        print(f"[geom] min_n_points={args.min_n_points}: kept {len(kept)}/{before}")

    rows = []
    for ds_idx in range(len(dataset.files)):
        item = dataset[ds_idx]
        fid = item.get("fruit_id", "")
        if isinstance(fid, (list, tuple)): fid = fid[0]
        fr = item.get("frame_id", "")
        if isinstance(fr, (list, tuple)): fr = fr[0]
        if "partial_pcd" not in item:
            continue
        feats = pcd_geom_features(item["partial_pcd"])
        rows.append({
            "fruit_id": str(fid), "frame_id": str(fr),
            "n_pts": feats["n_pts"],
            "axis_a": feats["pca_axes_lengths"][0],
            "axis_b": feats["pca_axes_lengths"][1],
            "axis_c": feats["pca_axes_lengths"][2],
            "ellipsoid_vol_ml": feats["ellipsoid_vol_ml"],
            "bbox_vol_ml": feats["bbox_vol_ml"],
        })
        if (ds_idx + 1) % 200 == 0:
            print(f"[geom] {ds_idx+1}/{len(dataset.files)}")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"[geom] wrote {len(df)} rows -> {args.output}")
    print(df.describe())


if __name__ == "__main__":
    main()
