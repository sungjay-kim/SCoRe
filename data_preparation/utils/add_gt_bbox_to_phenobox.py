#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute GT bounding-box dimensions from kmelon_centered point clouds and append to phenobox_data.csv.

Reads:
  - GT point clouds: <gt_root>/<S_name>/laser/fruit.ply
    default gt_root: /home/mainuser/mount/sungjay/tcore/corepp/data/kmelon_centered
  - phenobox csv (must include S_name column): /home/mainuser/mount/sungjay/kmelon_data/csvs/phenobox_data.csv

Writes:
  - Updates phenobox_data.csv in-place (creates timestamped backup) by adding columns:
      GT_width  = max(dx, dy)
      GT_height = dz
    where dx,dy,dz are axis-aligned bbox extents in the point cloud coordinate system.

Notes:
  - This is read-only w.r.t. training artifacts; it only reads GT point clouds.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import open3d as o3d


DEFAULT_GT_ROOT = Path("/home/mainuser/mount/sungjay/tcore/corepp/data/kmelon_centered")
DEFAULT_PHENO = Path("/home/mainuser/mount/sungjay/kmelon_data/csvs/phenobox_data.csv")


def compute_bbox_dims(ply_path: Path) -> Tuple[float, float]:
    """
    Returns (GT_width, GT_height).
      GT_width  = max(x_extent, y_extent)
      GT_height = z_extent
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return float("nan"), float("nan")
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    ext = mx - mn
    dx, dy, dz = float(ext[0]), float(ext[1]), float(ext[2])
    return max(dx, dy), dz


def build_gt_map(gt_root: Path) -> Dict[str, Tuple[float, float]]:
    """
    Map instance name (e.g. 'S34' or 'S84_unhealthy') -> (GT_width, GT_height)
    """
    mp: Dict[str, Tuple[float, float]] = {}
    for sdir in sorted(gt_root.iterdir(), key=lambda p: p.name):
        if not sdir.is_dir():
            continue
        ply = sdir / "laser" / "fruit.ply"
        if not ply.exists():
            continue
        w, h = compute_bbox_dims(ply)
        mp[sdir.name] = (w, h)
    return mp


def main() -> None:
    ap = argparse.ArgumentParser(description="Add GT bbox dimensions to phenobox_data.csv")
    ap.add_argument("--gt_root", type=str, default=str(DEFAULT_GT_ROOT))
    ap.add_argument("--phenobox_csv", type=str, default=str(DEFAULT_PHENO))
    ap.add_argument("--inplace", action="store_true", help="Update phenobox CSV in-place (default).")
    ap.add_argument("--no_backup", action="store_true", help="Do not write backup before overwriting.")
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    phenobox = Path(args.phenobox_csv)
    if not gt_root.is_dir():
        raise SystemExit(f"gt_root not found: {gt_root}")
    if not phenobox.exists():
        raise SystemExit(f"phenobox_csv not found: {phenobox}")

    gt_map = build_gt_map(gt_root)
    if not gt_map:
        raise SystemExit(f"No fruit.ply files found under {gt_root}")

    # Read phenobox CSV (already normalized to utf-8 in previous step)
    with phenobox.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise SystemExit(f"No header found in {phenobox}")
        fieldnames = list(r.fieldnames)
        rows = list(r)

    if "S_name" not in fieldnames:
        raise SystemExit(f"phenobox_data.csv must contain S_name column. Found: {fieldnames}")

    # Ensure new columns exist
    for c in ["GT_width", "GT_height"]:
        if c not in fieldnames:
            fieldnames.append(c)

    updated = 0
    missing = 0
    for row in rows:
        s_name = (row.get("S_name") or "").strip()
        inst = s_name[:-4] if s_name.lower().endswith(".ply") else s_name
        if not inst:
            row["GT_width"] = ""
            row["GT_height"] = ""
            missing += 1
            continue
        v = gt_map.get(inst)
        if v is None:
            row["GT_width"] = ""
            row["GT_height"] = ""
            missing += 1
            continue
        w, h = v
        row["GT_width"] = f"{w:.6f}"
        row["GT_height"] = f"{h:.6f}"
        updated += 1

    out_path = phenobox if args.inplace or True else phenobox.with_suffix(phenobox.suffix + ".with_gt.csv")
    if out_path == phenobox and not args.no_backup:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = phenobox.with_suffix(phenobox.suffix + f".bak_gt_{ts}")
        backup.write_bytes(phenobox.read_bytes())
        print(f"Wrote backup: {backup}")

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"GT instances available: {len(gt_map)} (from {gt_root})")
    print(f"Wrote updated phenobox: {out_path}")
    print(f"Rows updated: {updated}, missing GT mapping: {missing}")


if __name__ == "__main__":
    main()


