#!/usr/bin/env python3

import argparse
import os
import sys
from typing import Tuple, List

import numpy as np
import open3d as o3d


def compute_aabb_corners(point_cloud: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Compute the 8 corners of the axis-aligned bounding box of a point cloud.

    Returns an array of shape (8, 3) with the corner coordinates.
    """
    aabb = point_cloud.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()

    xmin, ymin, zmin = min_bound
    xmax, ymax, zmax = max_bound

    corners = np.array([
        [xmin, ymin, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],
        [xmax, ymin, zmin],
        [xmax, ymin, zmax],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax],
    ], dtype=np.float32)

    return corners


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def find_sample_ids(root: str) -> List[str]:
    """
    Return a list of sample IDs (immediate subdirectories) under the dataset root.
    """
    try:
        return [d for d in next(os.walk(root))[1]]
    except StopIteration:
        return []


def process_sample(root: str, sample_id: str, laser_relpath: str, tf_subdir: str, fruit_filename: str) -> Tuple[bool, str]:
    sample_dir = os.path.join(root, sample_id)
    fruit_path = os.path.join(sample_dir, laser_relpath, fruit_filename)
    tf_dir = os.path.join(sample_dir, tf_subdir)
    out_path = os.path.join(tf_dir, "bounding_box.npz")

    if not os.path.exists(fruit_path):
        return False, f"[skip] fruit file not found: {fruit_path}"

    try:
        pcd = o3d.io.read_point_cloud(fruit_path)
        if len(pcd.points) == 0:
            return False, f"[skip] empty point cloud: {fruit_path}"

        corners = compute_aabb_corners(pcd)
        ensure_dir(tf_dir)
        # Save as npz with default key 'arr_0' to match downstream loader usage
        np.savez(out_path, corners)
        return True, f"[ok] {sample_id}: wrote {out_path}"
    except Exception as e:
        return False, f"[err] {sample_id}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Generate per-sample tf/bounding_box.npz from fruit.ply")
    parser.add_argument("--root", required=True, help="Dataset root containing sample subfolders (e.g., data/kmelon_centered)")
    parser.add_argument("--laser_relpath", default="laser", help="Relative path from sample to laser dir (default: laser)")
    parser.add_argument("--tf_subdir", default="tf", help="Subdirectory under sample where to write bounding_box.npz (default: tf)")
    parser.add_argument("--fruit_filename", default="fruit.ply", help="Filename of the fruit point cloud (default: fruit.ply)")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"Root directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    sample_ids = find_sample_ids(root)
    if not sample_ids:
        print(f"No sample subdirectories found under: {root}", file=sys.stderr)
        sys.exit(1)

    num_ok = 0
    num_total = 0
    for sid in sorted(sample_ids):
        num_total += 1
        ok, msg = process_sample(root, sid, args.laser_relpath, args.tf_subdir, args.fruit_filename)
        print(msg)
        if ok:
            num_ok += 1

    print(f"Done. Wrote {num_ok}/{num_total} bounding_box.npz files.")


if __name__ == "__main__":
    main()


