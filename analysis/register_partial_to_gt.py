"""
Partial PCD to ground truth registration.

Strategy:
  1. Scale normalization (bbox diagonal ratio)
  2. Detect black markers on GT only
  3. Compute partial PCD viewing direction (mean point direction from origin)
  4. Rotate partial so its viewing direction faces the GT marker region
  5. Refine with ICP (point-to-point → point-to-plane)
"""

import open3d as o3d
import numpy as np
import json
import argparse
import os
from scipy.spatial import KDTree


def load_pcds(sample_dir, date="20260323"):
    """Load partial and GT point clouds for a sample."""
    partial_path = os.path.join(sample_dir, "orbbec", "pointclouds", f"{date}_preprocessed.ply")
    gt_path = os.path.join(sample_dir, "laser", "fruit.ply")
    return o3d.io.read_point_cloud(partial_path), o3d.io.read_point_cloud(gt_path)


def compute_scale_factor(partial, gt):
    """Compute scale ratio from bounding box diagonals."""
    p_bbox = partial.get_axis_aligned_bounding_box()
    g_bbox = gt.get_axis_aligned_bounding_box()
    p_diag = np.linalg.norm(p_bbox.get_max_bound() - p_bbox.get_min_bound())
    g_diag = np.linalg.norm(g_bbox.get_max_bound() - g_bbox.get_min_bound())
    return g_diag / p_diag


def scale_point_cloud(pcd, scale):
    """Scale a point cloud uniformly around origin."""
    scaled = o3d.geometry.PointCloud(pcd)
    pts = np.asarray(scaled.points)
    scaled.points = o3d.utility.Vector3dVector(pts * scale)
    return scaled


def detect_gt_markers(pcd, brightness_thr=60, cluster_radius_ratio=0.03, min_cluster_size=3):
    """Detect dark marker clusters on GT PCD by color thresholding.

    Returns list of (centroid, size) sorted by size descending.
    Largest cluster is excluded if 3x bigger than second (likely stem scar).
    """
    if not pcd.has_colors():
        return []

    colors = np.asarray(pcd.colors)
    pts = np.asarray(pcd.points)
    brightness = colors.mean(axis=1)
    dark_mask = brightness < (brightness_thr / 255.0)
    dark_indices = np.where(dark_mask)[0]

    if len(dark_indices) < min_cluster_size:
        return []

    dark_pts = pts[dark_indices]
    bbox_diag = np.linalg.norm(pts.max(0) - pts.min(0))
    radius = bbox_diag * cluster_radius_ratio

    # Union-find clustering
    tree = KDTree(dark_pts)
    pairs = tree.query_pairs(r=radius)
    parent = list(range(len(dark_pts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    for i, j in pairs:
        union(i, j)

    clusters = {}
    for i in range(len(dark_pts)):
        clusters.setdefault(find(i), []).append(i)

    results = []
    for indices in clusters.values():
        if len(indices) >= min_cluster_size:
            centroid = dark_pts[indices].mean(axis=0)
            results.append((centroid, len(indices)))

    results.sort(key=lambda x: -x[1])

    # Skip largest if likely stem scar
    if len(results) > 1 and results[0][1] > results[1][1] * 3:
        results = results[1:]

    return results


def rotation_align_vectors(v_from, v_to):
    """Rotation matrix aligning v_from to v_to (Rodrigues)."""
    a = v_from / np.linalg.norm(v_from)
    b = v_to / np.linalg.norm(v_to)
    cross = np.cross(a, b)
    dot = np.dot(a, b)

    if np.linalg.norm(cross) < 1e-8:
        if dot > 0:
            return np.eye(3)
        perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)

    K = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0],
    ])
    return np.eye(3) + K + K @ K / (1 + dot)


def compute_viewing_direction(pcd):
    """Compute viewing direction of partial PCD = mean point direction from origin."""
    pts = np.asarray(pcd.points)
    mean_dir = pts.mean(axis=0)
    norm = np.linalg.norm(mean_dir)
    if norm < 1e-10:
        return np.array([0, 0, 1])
    return mean_dir / norm


def compute_marker_init_transform(partial_scaled, gt_markers):
    """Compute initial rotation: align partial viewing direction → GT marker direction.

    Tries each GT marker as the target direction. For each, rotates partial's
    viewing direction to face the marker, then runs a quick ICP check.
    Returns the best initial transform.
    """
    if not gt_markers:
        return np.eye(4), None

    view_dir = compute_viewing_direction(partial_scaled)

    # Try each GT marker as the target face
    candidates = []
    for i, (marker_pos, marker_size) in enumerate(gt_markers):
        target_dir = marker_pos / np.linalg.norm(marker_pos)
        R = rotation_align_vectors(view_dir, target_dir)
        T = np.eye(4)
        T[:3, :3] = R
        candidates.append((T, i, marker_size))

    return candidates


def run_icp(source, target, max_correspondence_distance, init_transform=None):
    """Run point-to-point then point-to-plane ICP."""
    if init_transform is None:
        init_transform = np.eye(4)

    result_p2p = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )

    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=max_correspondence_distance * 2, max_nn=30))
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=max_correspondence_distance * 2, max_nn=30))

    result_p2l = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance * 0.5,
        result_p2p.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )

    return result_p2l


def chamfer_distance_partial(source_pts, target_pts):
    """Compute partial (source→target) Chamfer distance."""
    tree = KDTree(target_pts)
    dists, _ = tree.query(source_pts)
    return dists.mean(), dists


def register_sample(sample_dir, date="20260323", g_marker_thr=60):
    """Full registration pipeline for one sample."""
    sample_name = os.path.basename(sample_dir)
    print(f"\n{'='*60}")
    print(f"Processing: {sample_name}")
    print(f"{'='*60}")

    # Load
    partial, gt = load_pcds(sample_dir, date)
    print(f"Partial: {len(partial.points)} pts, GT: {len(gt.points)} pts")

    # Scale normalization
    scale = compute_scale_factor(partial, gt)
    print(f"Scale factor: {scale:.6f}")
    partial_scaled = scale_point_cloud(partial, scale)

    g_pts = np.asarray(gt.points)
    bbox_diag = np.linalg.norm(g_pts.max(0) - g_pts.min(0))
    max_corr_dist = bbox_diag * 0.3

    # Partial viewing direction
    view_dir = compute_viewing_direction(partial_scaled)
    print(f"Partial viewing dir: [{view_dir[0]:+.4f}, {view_dir[1]:+.4f}, {view_dir[2]:+.4f}]")

    # GT marker detection
    gt_markers = detect_gt_markers(gt, brightness_thr=g_marker_thr)
    print(f"GT markers: {len(gt_markers)}")
    for i, (c, sz) in enumerate(gt_markers[:10]):
        d = c / np.linalg.norm(c)
        print(f"  [{i}] size={sz:>4}, pos=[{c[0]:+.5f}, {c[1]:+.5f}, {c[2]:+.5f}], "
              f"dir=[{d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}]")

    # Try each GT marker as init orientation, pick best after ICP
    candidates = compute_marker_init_transform(partial_scaled, gt_markers)

    best_cd = float('inf')
    best_T = np.eye(4)
    best_marker_idx = -1

    if candidates:
        print(f"\nTrying {len(candidates)} marker orientations...")
        for init_T, marker_idx, marker_size in candidates:
            src = o3d.geometry.PointCloud(partial_scaled)
            src.transform(init_T)
            icp_res = run_icp(src, gt, max_corr_dist)
            final_T = icp_res.transformation @ init_T

            test_pcd = o3d.geometry.PointCloud(partial_scaled)
            test_pcd.transform(final_T)
            cd, _ = chamfer_distance_partial(np.asarray(test_pcd.points), g_pts)

            if cd < best_cd:
                best_cd = cd
                best_T = final_T
                best_marker_idx = marker_idx

        print(f"Best: marker[{best_marker_idx}] → CD={best_cd:.6f}")

    # Also try plain ICP (no marker init) for comparison
    icp_plain = run_icp(o3d.geometry.PointCloud(partial_scaled), gt, max_corr_dist)
    plain_pcd = o3d.geometry.PointCloud(partial_scaled)
    plain_pcd.transform(icp_plain.transformation)
    cd_plain, _ = chamfer_distance_partial(np.asarray(plain_pcd.points), g_pts)
    print(f"Plain ICP (no marker): CD={cd_plain:.6f}")

    # Pick winner
    if cd_plain < best_cd:
        best_T = icp_plain.transformation
        best_cd = cd_plain
        method = "icp_only"
        print(f"Winner: plain ICP")
    else:
        method = f"marker[{best_marker_idx}]+icp"
        print(f"Winner: marker[{best_marker_idx}]+ICP (improved {(cd_plain - best_cd)/cd_plain*100:.1f}%)")

    # Final evaluation
    partial_final = o3d.geometry.PointCloud(partial_scaled)
    partial_final.transform(best_T)
    p_final_pts = np.asarray(partial_final.points)

    tree = KDTree(g_pts)
    cov_dists, cov_idx = tree.query(p_final_pts)
    covered = np.unique(cov_idx[cov_dists < bbox_diag * 0.05])
    coverage_ratio = len(covered) / len(g_pts)
    print(f"Final CD: {best_cd:.6f}, Coverage: {len(covered)}/{len(g_pts)} ({coverage_ratio*100:.1f}%)")

    return {
        "sample": sample_name,
        "n_partial": len(partial.points),
        "n_gt": len(gt.points),
        "scale_factor": float(scale),
        "n_gt_markers": len(gt_markers),
        "method": method,
        "chamfer_final": float(best_cd),
        "chamfer_plain_icp": float(cd_plain),
        "transform": best_T.tolist(),
        "gt_coverage_ratio": float(coverage_ratio),
        "gt_covered_points": int(len(covered)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Register partial PCD to GT: GT marker init + ICP refine")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--samples", nargs="+",
                        default=["Ptest_1", "Ptest_2", "Ptest_3", "Ptest_4"])
    parser.add_argument("--date", default="20260323")
    parser.add_argument("--g-marker-thr", type=int, default=60,
                        help="Brightness threshold for GT marker detection")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    all_results = []
    for sample in args.samples:
        sample_dir = os.path.join(args.data_dir, sample)
        if not os.path.isdir(sample_dir):
            print(f"WARNING: {sample_dir} not found, skipping")
            continue
        all_results.append(register_sample(sample_dir, args.date, args.g_marker_thr))

    # Summary
    print(f"\n{'='*75}")
    print("SUMMARY")
    print(f"{'='*75}")
    print(f"{'Sample':<10} {'Method':<20} {'GT markers':>10} "
          f"{'CD_final':>10} {'CD_plain':>10} {'Coverage':>8}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['sample']:<10} {r['method']:<20} {r['n_gt_markers']:>10} "
              f"{r['chamfer_final']:>10.6f} {r['chamfer_plain_icp']:>10.6f} "
              f"{r['gt_coverage_ratio']*100:>7.1f}%")

    out_path = args.out_json or os.path.join(args.data_dir, "registration_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
