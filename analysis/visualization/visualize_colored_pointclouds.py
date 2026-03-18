#!/usr/bin/env python3
"""
Utility script to visualize and analyze colored point clouds with distance information.

Usage:
    # Visualize a single colored point cloud
    python visualize_colored_pointclouds.py --file path/to/colored.ply
    
    # Analyze all colored point clouds in a directory
    python visualize_colored_pointclouds.py --dir path/to/colored_pointclouds/
    
    # Create distance histograms
    python visualize_colored_pointclouds.py --dir path/to/colored_pointclouds/ --histogram
"""

import argparse
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_and_extract_distances(pcd_path, gt_pcd_path=None):
    """
    Load a colored point cloud and optionally recompute distances if GT is available.
    
    Returns:
        pcd: Open3D point cloud
        distances: Distance array (None if can't compute)
    """
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    
    if gt_pcd_path is not None and Path(gt_pcd_path).exists():
        # Recompute distances from GT
        gt_pcd = o3d.io.read_point_cloud(str(gt_pcd_path))
        distances = np.asarray(pcd.compute_point_cloud_distance(gt_pcd))
        return pcd, distances
    
    # Try to extract distances from color information
    # Note: This is approximate and won't work if colormap isn't monotonic
    return pcd, None


def visualize_single(pcd_path, show_stats=True):
    """Visualize a single colored point cloud."""
    print(f"Loading: {pcd_path}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    
    num_points = len(pcd.points)
    print(f"Number of points: {num_points}")
    
    # Try to load corresponding statistics file
    stats_path = Path(str(pcd_path).replace('_colored.ply', '_distance_stats.txt'))
    if stats_path.exists() and show_stats:
        print(f"\n{stats_path.name}:")
        print("-" * 50)
        with open(stats_path, 'r') as f:
            print(f.read())
    
    # Visualize
    print("\nOpening 3D viewer...")
    print("Controls:")
    print("  - Left mouse: Rotate")
    print("  - Right mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - Q or ESC: Close")
    
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"Colored Point Cloud - {pcd_path.name}",
        width=1280,
        height=720,
        point_show_normal=False
    )


def create_distance_histogram(distances, title, save_path=None):
    """Create a histogram of distance errors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale histogram
    ax1.hist(distances * 1000, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Distance (mm)')
    ax1.set_ylabel('Number of Points')
    ax1.set_title(f'{title}\n(Linear Scale)')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(distances)*1000:.2f} mm\n'
    stats_text += f'Median: {np.median(distances)*1000:.2f} mm\n'
    stats_text += f'Std: {np.std(distances)*1000:.2f} mm\n'
    stats_text += f'95%: {np.percentile(distances, 95)*1000:.2f} mm'
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    # Log scale histogram (useful for seeing tail behavior)
    ax2.hist(distances * 1000, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Distance (mm)')
    ax2.set_ylabel('Number of Points')
    ax2.set_title(f'{title}\n(Log Scale)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_directory(directory, create_histograms=False, gt_dir=None):
    """Analyze all colored point clouds in a directory."""
    directory = Path(directory)
    colored_files = sorted(directory.glob("*_colored.ply"))
    
    if len(colored_files) == 0:
        print(f"No colored point cloud files found in {directory}")
        return
    
    print(f"Found {len(colored_files)} colored point clouds")
    
    # Collect statistics
    all_distances = []
    sample_stats = []
    
    for pcd_path in colored_files:
        # Load stats file
        stats_path = Path(str(pcd_path).replace('_colored.ply', '_distance_stats.txt'))
        
        if stats_path.exists():
            # Parse statistics from file
            with open(stats_path, 'r') as f:
                lines = f.readlines()
                stats = {}
                for line in lines:
                    if 'Mean:' in line:
                        stats['mean'] = float(line.split(':')[1].strip())
                    elif 'Max:' in line:
                        stats['max'] = float(line.split(':')[1].strip())
                    elif 'Min:' in line:
                        stats['min'] = float(line.split(':')[1].strip())
                    elif 'Median:' in line:
                        stats['median'] = float(line.split(':')[1].strip())
                
                if stats:
                    stats['file'] = pcd_path.name
                    sample_stats.append(stats)
        
        # Try to get actual distances if GT is available
        if gt_dir is not None:
            fruit_id = pcd_path.stem.replace('_colored', '')
            gt_path = Path(gt_dir) / fruit_id / 'laser' / 'fruit.ply'
            pcd, distances = load_and_extract_distances(pcd_path, gt_path)
            if distances is not None:
                all_distances.append(distances)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS ACROSS ALL SAMPLES")
    print("=" * 80)
    
    if sample_stats:
        means = [s['mean'] for s in sample_stats]
        medians = [s['median'] for s in sample_stats]
        maxs = [s['max'] for s in sample_stats]
        
        print(f"\nMean Distance across samples:")
        print(f"  Average: {np.mean(means)*1000:.3f} mm")
        print(f"  Std:     {np.std(means)*1000:.3f} mm")
        print(f"  Min:     {np.min(means)*1000:.3f} mm")
        print(f"  Max:     {np.max(means)*1000:.3f} mm")
        
        print(f"\nMedian Distance across samples:")
        print(f"  Average: {np.mean(medians)*1000:.3f} mm")
        print(f"  Std:     {np.std(medians)*1000:.3f} mm")
        
        print(f"\nMax Distance across samples:")
        print(f"  Average: {np.max(maxs)*1000:.3f} mm")
        print(f"  Worst:   {np.max(maxs)*1000:.3f} mm")
    
    # Create histograms if requested
    if create_histograms and all_distances:
        print(f"\nCreating histograms for {len(all_distances)} samples...")
        
        # Create individual histograms
        for i, (distances, pcd_path) in enumerate(zip(all_distances, colored_files)):
            hist_path = pcd_path.parent / f"{pcd_path.stem}_histogram.png"
            create_distance_histogram(
                distances,
                f"Distance Distribution - {pcd_path.name}",
                save_path=hist_path
            )
        
        # Create combined histogram
        all_dist_combined = np.concatenate(all_distances)
        combined_path = directory / "combined_distance_histogram.png"
        create_distance_histogram(
            all_dist_combined,
            f"Combined Distance Distribution ({len(colored_files)} samples)",
            save_path=combined_path
        )
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and analyze colored point clouds with distance information",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Path to a single colored point cloud file (.ply)'
    )
    
    parser.add_argument(
        '--dir', '-d',
        type=str,
        help='Path to directory containing colored point clouds'
    )
    
    parser.add_argument(
        '--histogram',
        action='store_true',
        help='Create distance histograms (requires GT point clouds)'
    )
    
    parser.add_argument(
        '--gt-dir',
        type=str,
        help='Path to directory containing ground truth point clouds'
    )
    
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Do not show statistics files'
    )
    
    args = parser.parse_args()
    
    if args.file:
        # Visualize single file
        visualize_single(Path(args.file), show_stats=not args.no_stats)
    
    elif args.dir:
        # Analyze directory
        analyze_directory(
            args.dir,
            create_histograms=args.histogram,
            gt_dir=args.gt_dir
        )
    
    else:
        parser.print_help()
        print("\nError: Must specify either --file or --dir")


if __name__ == "__main__":
    main()

