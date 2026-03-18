#!/usr/bin/env python3
"""
Visualize temporal shape evolution.
Creates a GIF showing how the reconstructed shape changes over time for a specific fruit.

Usage:
    python visualize_temporal_evolution.py \
        --inf-dir ../experiments/.../temporal_inf/2025... \
        --fruit-id KMELON_0001 \
        --output evolution.gif
"""

import argparse
import os
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from PIL import Image

def render_pointcloud_view(pcd, width, height, camera_params, bg_color=(1, 1, 1), point_size=5):
    """Render a point cloud from a specific camera viewpoint using matplotlib."""
    points = np.asarray(pcd.points)
    
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones_like(points) * 0.5
    
    dpi = 100
    figsize = (width / dpi, height / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=bg_color)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=colors, s=point_size, marker='.', alpha=1.0, linewidths=0)
    
    # View setup
    center = camera_params['lookat']
    distance = camera_params.get('distance', 0.5)
    limit = distance * 0.4
    
    ax.set_xlim([center[0] - limit, center[0] + limit])
    ax.set_ylim([center[1] - limit, center[1] + limit])
    ax.set_zlim([center[2] - limit, center[2] + limit])
    
    angle = camera_params.get('angle', 0)
    ax.view_init(elev=15, azim=angle)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_array = img_array[:, :, :3]
    plt.close(fig)
    return img_array

def add_text(img, text, font_size=15):
    """Add text overlay."""
    h, w = img.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis('off')
    
    # Bottom center text
    ax.text(w/2, h-20, text, fontsize=font_size, fontweight='bold',
            ha='center', va='bottom', color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
            
    fig.canvas.draw()
    res = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    res = res.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return res[:, :, :3]

def main():
    parser = argparse.ArgumentParser(description="Visualize temporal evolution")
    parser.add_argument("--inf-dir", required=True, help="Path to temporal inference directory (containing 'meshes')")
    parser.add_argument("--fruit-id", required=True, help="Fruit ID to visualize (e.g. KMELON_0001)")
    parser.add_argument("--output", default="evolution.gif", help="Output GIF path")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--hold-frames", type=int, default=5, help="Number of frames to hold each timestep")
    args = parser.parse_args()

    inf_dir = Path(args.inf_dir)
    mesh_dir = inf_dir / "meshes"
    
    if not mesh_dir.exists():
        print(f"Error: {mesh_dir} does not exist")
        return

    # Find all meshes for this fruit
    # Expected format: {fruit_id}_{timestamp_or_frameid}.ply
    pattern = f"{args.fruit_id}_*.ply"
    mesh_files = sorted(list(mesh_dir.glob(pattern)))
    
    if not mesh_files:
        print(f"No meshes found for {args.fruit_id} in {mesh_dir}")
        return

    print(f"Found {len(mesh_files)} frames for {args.fruit_id}")

    # Load all meshes to find common center/scale
    meshes = []
    all_points = []
    
    for f in mesh_files:
        mesh = o3d.io.read_triangle_mesh(str(f))
        if mesh.has_vertices():
            # Sample points for point cloud rendering (looks better usually)
            pcd = mesh.sample_points_poisson_disk(number_of_points=5000)
            # Or just take vertices if dense enough
            # pcd = o3d.geometry.PointCloud(mesh.vertices)
            
            # Color based on index for fun? Or uniform?
            # Let's clean uniform color
            pcd.paint_uniform_color([0.2, 0.6, 1.0]) # Nice blue
            
            meshes.append((f, pcd))
            all_points.append(np.asarray(pcd.points))
        else:
            print(f"Warning: Empty mesh {f.name}")

    if not all_points:
        return

    # Calculate global center and bounds
    all_points_concat = np.concatenate(all_points, axis=0)
    center = all_points_concat.mean(axis=0)
    bounds = all_points_concat.max(axis=0) - all_points_concat.min(axis=0)
    radius = np.linalg.norm(bounds) * 1.5

    print(f"Global center: {center}")
    print(f"Visualization radius: {radius}")

    frames = []
    
    # Render loop
    angle = 0
    angle_step = 360 / (len(meshes) * args.hold_frames) # One full rotation over growth?
    # Or just slow rotation
    angle_step = 5 # degrees per step
    
    current_angle = 0
    
    for i, (fpath, pcd) in enumerate(meshes):
        # Extract timestamp/frame from filename
        # file: KMELON_0001_20240501.ply -> 20240501
        label = fpath.stem.replace(f"{args.fruit_id}_", "")
        
        print(f"Rendering frame {i+1}/{len(meshes)}: {label}")
        
        for _ in range(args.hold_frames):
            camera_params = {
                'lookat': center,
                'distance': radius,
                'angle': current_angle
            }
            
            img = render_pointcloud_view(pcd, 640, 640, camera_params)
            img = add_text(img, f"Time: {label}", font_size=12)
            frames.append(img)
            
            current_angle = (current_angle + angle_step) % 360

    print(f"Saving GIF to {args.output}...")
    imageio.mimsave(args.output, frames, fps=args.fps)
    print("Done!")

if __name__ == "__main__":
    main()
