#!/usr/bin/env python3
"""
Demo script to test the new visualization features.
Creates synthetic point clouds and visualizes them with different colormaps.
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
# Add parent directory to path for imports
# Points to corepp root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the colorization function from test.py
from test import colorize_point_cloud, save_colorbar_legend


def create_synthetic_pointcloud(num_points=5000, add_noise=True):
    """
    Create a synthetic sphere point cloud for demonstration.
    """
    # Create a unit sphere
    phi = np.random.uniform(0, 2*np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(costheta)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    points = np.stack([x, y, z], axis=1) * 0.1  # Scale to 10cm radius
    
    # Add noise to simulate prediction errors
    if add_noise:
        # Add varying levels of noise (simulating better/worse predictions)
        noise_levels = np.random.exponential(0.005, num_points)
        noise = np.random.randn(num_points, 3) * noise_levels[:, np.newaxis]
        points += noise
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd


def demo_colormaps():
    """
    Demonstrate different colormap options.
    """
    print("=" * 80)
    print("DEMONSTRATION: Colormap Options for Distance Visualization")
    print("=" * 80)
    
    # Create synthetic data
    print("\nCreating synthetic point clouds...")
    gt_pcd = create_synthetic_pointcloud(num_points=3000, add_noise=False)
    pred_pcd = create_synthetic_pointcloud(num_points=5000, add_noise=True)
    
    # Test different colormaps
    colormaps = ['jet', 'viridis', 'plasma', 'turbo', 'RdYlGn_r']
    max_distance = 0.02  # 2cm
    
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nTesting {len(colormaps)} different colormaps...")
    print(f"Output directory: {output_dir.absolute()}")
    
    for cmap in colormaps:
        print(f"\n  Processing colormap: {cmap}")
        
        # Colorize with distance information
        colored_pcd, distances = colorize_point_cloud(
            pred_pcd, gt_pcd,
            cmap_name=cmap,
            max_distance=max_distance,
            return_distances=True
        )
        
        if colored_pcd is not None:
            # Save colored point cloud
            pcd_file = output_dir / f"demo_{cmap}.ply"
            o3d.io.write_point_cloud(str(pcd_file), colored_pcd)
            print(f"    ✓ Saved: {pcd_file.name}")
            
            # Save colorbar legend
            legend_file = output_dir / f"legend_{cmap}.png"
            save_colorbar_legend(
                legend_file,
                cmap_name=cmap,
                min_val=0.0,
                max_val=max_distance,
                title=f"Distance (m) - {cmap}"
            )
            
            # Print statistics
            if distances is not None:
                print(f"    Distance stats: min={distances.min()*1000:.2f}mm, "
                      f"mean={distances.mean()*1000:.2f}mm, "
                      f"max={distances.max()*1000:.2f}mm")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated files in: {output_dir.absolute()}")
    print("\nFiles created:")
    print("  - demo_*.ply           : Colored point clouds (view in CloudCompare/MeshLab)")
    print("  - legend_*.png         : Colorbar legends showing color-to-distance mapping")
    print("\nRecommendations:")
    print("  - 'jet' or 'turbo'     : Most intuitive (blue=good, red=bad)")
    print("  - 'viridis' or 'plasma': Best for publications (perceptually uniform)")
    print("  - 'RdYlGn_r'          : Traffic light style (green=good, red=bad)")
    print("\n" + "=" * 80)


def demo_statistics():
    """
    Demonstrate statistics extraction.
    """
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Distance Statistics Extraction")
    print("=" * 80)
    
    # Create synthetic data
    gt_pcd = create_synthetic_pointcloud(num_points=3000, add_noise=False)
    pred_pcd = create_synthetic_pointcloud(num_points=5000, add_noise=True)
    
    # Colorize and get distances
    colored_pcd, distances = colorize_point_cloud(
        pred_pcd, gt_pcd,
        cmap_name='jet',
        return_distances=True
    )
    
    if distances is not None:
        print("\nDistance Statistics (meters):")
        print("=" * 40)
        print(f"Min:     {distances.min():.6f}")
        print(f"Max:     {distances.max():.6f}")
        print(f"Mean:    {distances.mean():.6f}")
        print(f"Median:  {np.median(distances):.6f}")
        print(f"Std:     {distances.std():.6f}")
        print(f"\nPercentiles:")
        print(f"  10%:   {np.percentile(distances, 10):.6f}")
        print(f"  25%:   {np.percentile(distances, 25):.6f}")
        print(f"  50%:   {np.percentile(distances, 50):.6f}")
        print(f"  75%:   {np.percentile(distances, 75):.6f}")
        print(f"  90%:   {np.percentile(distances, 90):.6f}")
        print(f"  95%:   {np.percentile(distances, 95):.6f}")
        print(f"  99%:   {np.percentile(distances, 99):.6f}")
        print("=" * 40)
        
        # Create histogram
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(distances * 1000, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Distance (mm)')
        ax.set_ylabel('Number of Points')
        ax.set_title('Distance Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {np.mean(distances)*1000:.2f} mm\n'
        stats_text += f'Median: {np.median(distances)*1000:.2f} mm\n'
        stats_text += f'Std: {np.std(distances)*1000:.2f} mm\n'
        stats_text += f'95%: {np.percentile(distances, 95)*1000:.2f} mm'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
        
        hist_file = output_dir / "distance_histogram.png"
        plt.savefig(hist_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nHistogram saved to: {hist_file}")
        print("=" * 80)


def demo_comparison():
    """
    Demonstrate comparing good vs. poor predictions.
    """
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Comparing Good vs. Poor Predictions")
    print("=" * 80)
    
    gt_pcd = create_synthetic_pointcloud(num_points=3000, add_noise=False)
    
    # Good prediction (low noise)
    pred_good = create_synthetic_pointcloud(num_points=5000, add_noise=False)
    pred_good_points = np.asarray(pred_good.points)
    pred_good_points += np.random.randn(*pred_good_points.shape) * 0.002  # 2mm noise
    pred_good.points = o3d.utility.Vector3dVector(pred_good_points)
    
    # Poor prediction (high noise)
    pred_poor = create_synthetic_pointcloud(num_points=5000, add_noise=False)
    pred_poor_points = np.asarray(pred_poor.points)
    pred_poor_points += np.random.randn(*pred_poor_points.shape) * 0.01  # 10mm noise
    pred_poor.points = o3d.utility.Vector3dVector(pred_poor_points)
    
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    max_distance = 0.02
    
    # Process good prediction
    print("\nProcessing GOOD prediction (low noise)...")
    colored_good, dist_good = colorize_point_cloud(
        pred_good, gt_pcd,
        cmap_name='jet',
        max_distance=max_distance,
        return_distances=True
    )
    o3d.io.write_point_cloud(str(output_dir / "demo_good_prediction.ply"), colored_good)
    print(f"  Mean error: {dist_good.mean()*1000:.2f} mm")
    print(f"  Max error:  {dist_good.max()*1000:.2f} mm")
    
    # Process poor prediction
    print("\nProcessing POOR prediction (high noise)...")
    colored_poor, dist_poor = colorize_point_cloud(
        pred_poor, gt_pcd,
        cmap_name='jet',
        max_distance=max_distance,
        return_distances=True
    )
    o3d.io.write_point_cloud(str(output_dir / "demo_poor_prediction.ply"), colored_poor)
    print(f"  Mean error: {dist_poor.mean()*1000:.2f} mm")
    print(f"  Max error:  {dist_poor.max()*1000:.2f} mm")
    
    print("\n✓ Files saved:")
    print(f"  - demo_good_prediction.ply  (mostly blue = low error)")
    print(f"  - demo_poor_prediction.ply  (more red = high error)")
    print("=" * 80)


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print(" " * 20 + "VISUALIZATION SYSTEM DEMO")
    print("="*80)
    print("\nThis demo script tests the new visualization features:")
    print("  1. Different colormap options")
    print("  2. Distance statistics extraction")
    print("  3. Comparison of good vs. poor predictions")
    print("\n" + "="*80)
    
    try:
        # Run demos
        demo_colormaps()
        demo_statistics()
        demo_comparison()
        
        print("\n" + "="*80)
        print(" " * 25 + "ALL DEMOS COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("  1. View the generated PLY files in CloudCompare or MeshLab")
        print("  2. Check the legend PNG files to understand the color scales")
        print("  3. Examine the histogram to see distance distributions")
        print("  4. Run actual inference with your trained models")
        print("\nFor actual inference, add these to your config JSON:")
        print('  "save_colored_mesh": true,')
        print('  "visualization_colormap": "jet",')
        print('  "visualization_max_distance": 0.02')
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

