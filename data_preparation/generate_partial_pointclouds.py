#!/usr/bin/env python3
"""
Generate synthetic partial point clouds from complete laser scans.

Uses Open3D's hidden_point_removal to simulate partial observations
from different virtual camera viewpoints. Produces 10 partial views
per sample, matching the format of existing phenobox data.

Usage:
    python corepp/data_preparation/generate_partial_pointclouds.py \
        --src corepp/data/kmelon_centered \
        --samples P101 P102 P103 ... \
        --skip_existing
"""
import os
import argparse
import numpy as np
import open3d as o3d


def generate_viewpoints(center: np.ndarray, n_views: int = 10,
                        radius: float = 0.3, seed: int = 42) -> np.ndarray:
    """Generate camera positions on a sphere around the object center."""
    rng = np.random.default_rng(seed)
    # Use Fibonacci sphere for uniform distribution
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n_views):
        theta = np.arccos(1 - 2 * (i + 0.5) / n_views)
        phi = 2 * np.pi * i / golden_ratio
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        points.append([x, y, z])
    points = np.array(points) * radius + center
    # Add small random perturbation
    points += rng.normal(0, radius * 0.05, points.shape)
    return points


def _add_leaf_stem_occluders(pcd_pts: np.ndarray, vp: np.ndarray,
                              center: np.ndarray, fruit_radius: float,
                              rng: np.random.Generator,
                              n_leaves: int = (1, 3),
                              n_stems: int = (0, 2)) -> np.ndarray:
    """Remove points that are 'occluded' by virtual leaves/stems between camera and fruit.

    Occluder geometry:
      - Leaf: thin disk (radius ~30-60% fruit_radius) placed near fruit surface, normal random
              tilted toward camera. Points behind the disk (from camera ray) are removed.
      - Stem: thin cylinder (radius 3-6mm, length 2x fruit) crossing the scene at a random angle.

    Returns boolean mask: True = visible (keep), False = occluded.
    """
    keep = np.ones(len(pcd_pts), dtype=bool)
    cam_dir = (pcd_pts - vp)  # ray vectors from camera to each point
    cam_dir_norm = cam_dir / (np.linalg.norm(cam_dir, axis=1, keepdims=True) + 1e-9)

    # Sample number of leaves and stems for this view
    n_l = rng.integers(n_leaves[0], n_leaves[1] + 1)
    n_s = rng.integers(n_stems[0], n_stems[1] + 1)

    # ── Leaves: disks tangent-ish to fruit, between camera and fruit ──
    for _ in range(n_l):
        # Pick a random anchor point on fruit surface
        anchor_idx = rng.integers(0, len(pcd_pts))
        anchor = pcd_pts[anchor_idx]
        # Move anchor slightly toward camera (so disk is in front of fruit surface)
        toward_cam = (vp - anchor)
        toward_cam /= (np.linalg.norm(toward_cam) + 1e-9)
        leaf_center = anchor + toward_cam * fruit_radius * 0.3
        # Leaf normal: roughly facing camera with random tilt
        normal = toward_cam + rng.normal(0, 0.4, 3)
        normal /= (np.linalg.norm(normal) + 1e-9)
        leaf_radius = fruit_radius * rng.uniform(0.35, 0.7)
        # Compute intersection of camera→point ray with leaf plane
        # ray: vp + t*cam_dir_norm; plane: (X - leaf_center) · normal = 0
        denom = cam_dir_norm @ normal
        denom_safe = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        t = ((leaf_center - vp) @ normal) / denom_safe
        # Intersection point
        hit = vp[None, :] + t[:, None] * cam_dir_norm
        # Distance from leaf center
        d = np.linalg.norm(hit - leaf_center, axis=1)
        # Point distance from camera
        pt_dist = np.linalg.norm(pcd_pts - vp, axis=1)
        leaf_dist = np.linalg.norm(leaf_center - vp)
        # Occluded if: ray hits within leaf disk AND point is behind leaf (further from camera)
        occluded = (d < leaf_radius) & (t > 0) & (pt_dist > leaf_dist)
        keep &= ~occluded

    # ── Stems: thin cylinders ──
    for _ in range(n_s):
        # Random axis through scene, passing near fruit
        axis_dir = rng.normal(0, 1, 3)
        axis_dir /= (np.linalg.norm(axis_dir) + 1e-9)
        # Stem center: near fruit, slightly toward camera
        stem_center = center + rng.normal(0, fruit_radius * 0.3, 3)
        toward_cam_c = (vp - stem_center)
        toward_cam_c /= (np.linalg.norm(toward_cam_c) + 1e-9)
        stem_center = stem_center + toward_cam_c * fruit_radius * 0.4
        stem_radius = rng.uniform(0.003, 0.006)  # 3-6 mm in meters
        # For each point, compute distance from camera→point ray to the stem axis (line-line distance)
        # Ray: vp + t*cam_dir_norm.  Line: stem_center + s*axis_dir.
        # Closest distance between two lines:
        w0 = (vp - stem_center)[None, :]  # (1,3)
        a = (cam_dir_norm * cam_dir_norm).sum(axis=1)            # =1
        b = cam_dir_norm @ axis_dir                               # (N,)
        c_ = float(axis_dir @ axis_dir)                           # =1
        d_ = (cam_dir_norm * w0).sum(axis=1)                      # (N,)
        e_ = float(w0 @ axis_dir)                                 # scalar
        denom_ll = a*c_ - b*b
        denom_ll = np.where(np.abs(denom_ll) < 1e-9, 1e-9, denom_ll)
        sc = (b*e_ - c_*d_) / denom_ll  # ray param
        tc = (a*e_ - b*d_) / denom_ll   # line param
        ray_hit = vp[None, :] + sc[:, None] * cam_dir_norm
        line_hit = stem_center[None, :] + tc[:, None] * axis_dir
        dist_to_stem = np.linalg.norm(ray_hit - line_hit, axis=1)
        # Occluded if ray passes within stem_radius and point is behind closest approach
        pt_dist = np.linalg.norm(pcd_pts - vp, axis=1)
        approach_dist = sc  # distance along ray to closest approach
        occluded = (dist_to_stem < stem_radius) & (sc > 0) & (pt_dist > approach_dist)
        keep &= ~occluded

    return keep


def generate_partial_views(fruit_ply_path: str, n_views: int = 10,
                           seed: int = 42,
                           occlusion_leaves: bool = True) -> list:
    """Generate partial point clouds using HPR + simulated leaf/stem occluders."""
    pcd = o3d.io.read_point_cloud(fruit_ply_path)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return []

    center = points.mean(axis=0)
    extent = points.max(axis=0) - points.min(axis=0)
    fruit_radius = max(extent) * 0.5  # roughly half max extent
    radius = max(extent) * 4.0

    viewpoints = generate_viewpoints(center, n_views, radius, seed)
    rng = np.random.default_rng(seed)

    partial_views = []
    for vp in viewpoints:
        # Step 1: HPR
        _, pt_map = pcd.hidden_point_removal(vp, radius * 100)
        visible_pcd = pcd.select_by_index(pt_map)
        visible_pts = np.asarray(visible_pcd.points)

        if len(visible_pts) < 100:
            idx = np.random.choice(len(points), min(1500, len(points)), replace=False)
            visible_pcd = pcd.select_by_index(idx)
            visible_pts = np.asarray(visible_pcd.points)

        # Step 2: leaf/stem occluders (remove points blocked by virtual obstacles)
        if occlusion_leaves and len(visible_pts) > 200:
            keep = _add_leaf_stem_occluders(visible_pts, vp, center, fruit_radius, rng)
            kept_idx = np.where(keep)[0]
            if len(kept_idx) > 100:
                visible_pcd = visible_pcd.select_by_index(kept_idx)
                visible_pts = np.asarray(visible_pcd.points)

        # Step 3: downsample to ≤3000 (preserve spatial distribution)
        target_count = min(len(visible_pts), 3000)
        if target_count > 100:
            visible_pcd = visible_pcd.farthest_point_down_sample(target_count)

        partial_views.append(visible_pcd)

    return partial_views


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic partial point clouds from laser scans")
    parser.add_argument("--src", required=True,
                        help="Data root (e.g., corepp/data/kmelon_centered)")
    parser.add_argument("--samples", nargs="+", default=None,
                        help="Specific sample IDs to process (e.g., P101 P102). "
                             "If not given, processes all samples missing phenobox data.")
    parser.add_argument("--n_views", type=int, default=10,
                        help="Number of partial views per sample (default: 10)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip samples that already have phenobox data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no_occlusion", action="store_true",
                        help="Disable leaf/stem occluder simulation (HPR-only partial views)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting partial index for output filenames (default 0). Use >0 to append views.")
    args = parser.parse_args()

    # Determine which samples to process
    if args.samples:
        sample_ids = args.samples
    else:
        # Auto-detect: all directories with laser/fruit.ply but no phenobox/pointclouds/
        sample_ids = []
        for name in sorted(os.listdir(args.src)):
            sample_dir = os.path.join(args.src, name)
            if not os.path.isdir(sample_dir):
                continue
            fruit_path = os.path.join(sample_dir, "laser", "fruit.ply")
            phenobox_dir = os.path.join(sample_dir, "phenobox", "pointclouds")
            if os.path.isfile(fruit_path) and not os.path.isdir(phenobox_dir):
                sample_ids.append(name)

    print(f"Processing {len(sample_ids)} samples...")

    for sid in sample_ids:
        fruit_path = os.path.join(args.src, sid, "laser", "fruit.ply")
        out_dir = os.path.join(args.src, sid, "phenobox", "pointclouds")

        if not os.path.isfile(fruit_path):
            print(f"  {sid}: no laser/fruit.ply, skipping")
            continue

        if args.skip_existing and os.path.isdir(out_dir):
            existing = [f for f in os.listdir(out_dir) if f.startswith("partial_")]
            if len(existing) >= args.n_views:
                continue

        print(f"  {sid}: generating {args.n_views} partial views...", end=" ", flush=True)

        views = generate_partial_views(fruit_path, args.n_views,
                                       seed=args.seed + hash(sid) % 10000,
                                       occlusion_leaves=not args.no_occlusion)

        if not views:
            print("FAILED (empty point cloud)")
            continue

        os.makedirs(out_dir, exist_ok=True)
        for i, pcd in enumerate(views):
            out_path = os.path.join(out_dir, f"partial_{(i + args.start_idx):02d}.ply")
            o3d.io.write_point_cloud(out_path, pcd)

        counts = [len(np.asarray(v.points)) for v in views]
        print(f"done ({min(counts)}-{max(counts)} pts/view)")

    print("All done.")


if __name__ == "__main__":
    main()
