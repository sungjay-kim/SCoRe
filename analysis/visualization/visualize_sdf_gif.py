#!/usr/bin/env python3
"""
Create rotating GIF of SDF iso-levels (inner, surface, outer).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
import sys
import imageio

import deepsdf.deep_sdf.workspace as ws
from sdfrenderer.grid import Grid3D
import networks.utils as net_utils


def load_decoder_and_latent(experiment_dir, checkpoint, sample_name):
    """Load trained DeepSDF decoder and latent code."""
    specs = ws.load_experiment_specifications(experiment_dir)
    latent_size = specs["CodeLength"]
    
    # Load decoder
    arch = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()
    
    model_path = Path(experiment_dir) / 'ModelParameters' / f'{checkpoint}.pth'
    model_state = net_utils.load_without_parallel(torch.load(model_path))
    decoder.load_state_dict(model_state)
    decoder.eval()
    
    print(f"✓ Loaded decoder (latent size: {latent_size})")
    
    # Load latent code
    codes_dir = Path(experiment_dir) / 'Reconstructions' / checkpoint / 'Codes' / 'complete'
    code_path = codes_dir / f'{sample_name}.pth'
    if not code_path.exists():
        print(f"Error: Code file not found: {code_path}")
        sys.exit(1)
    
    latent_vector = torch.load(code_path).cuda().squeeze()
    print(f"✓ Loaded latent code for {sample_name}")
    
    return decoder, latent_vector, latent_size


def generate_sdf_grid(decoder, latent_vector, grid_density=64, bbox_size=0.15):
    """Generate a dense SDF grid."""
    device = 'cuda'
    
    bbox = {
        'xmin': -bbox_size, 'xmax': bbox_size,
        'ymin': -bbox_size, 'ymax': bbox_size,
        'zmin': -bbox_size, 'zmax': bbox_size
    }
    
    grid = Grid3D(grid_density, device, torch.float32, bbox=bbox)
    
    # Prepare decoder input
    decoder_input = torch.cat([
        latent_vector.expand(grid.points.size(0), -1),
        grid.points
    ], dim=1)
    
    # Predict SDF values
    with torch.no_grad():
        sdf_values = decoder(decoder_input)
    
    sdf_grid = sdf_values.cpu().numpy().reshape(grid_density, grid_density, grid_density)
    
    print(f"✓ Generated SDF grid: {sdf_grid.shape}")
    print(f"  SDF range: [{sdf_grid.min():.4f}, {sdf_grid.max():.4f}]")
    
    return sdf_grid, bbox


def render_sdf_frame(sdf_grid, bbox, azimuth, width=800, height=800):
    """
    Render a single frame of the SDF visualization at a specific angle.
    """
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    grid_density = sdf_grid.shape[0]
    
    # Adaptive iso-levels based on actual SDF range
    sdf_min = sdf_grid.min()
    sdf_max = sdf_grid.max()
    
    # Inner: halfway between minimum and surface (0)
    inner_level = sdf_min * 0.5
    # Outer: use a reasonable offset from surface
    outer_level = min(0.02, sdf_max * 0.2)  # Don't go too far out
    
    # Define iso-levels: inner=red, surface=blue, outer=green (smaller point sizes)
    levels = [
        {'value': inner_level, 'color': 'red', 'alpha': 0.5, 'size': 8, 'name': 'inner'},
        {'value': 0.0, 'color': 'blue', 'alpha': 0.8, 'size': 12, 'name': 'surface'},
        {'value': outer_level, 'color': 'green', 'alpha': 0.5, 'size': 8, 'name': 'outer'}
    ]
    
    # Create coordinate grids
    x = np.linspace(bbox['xmin'], bbox['xmax'], grid_density)
    y = np.linspace(bbox['ymin'], bbox['ymax'], grid_density)
    z = np.linspace(bbox['zmin'], bbox['zmax'], grid_density)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Plot each iso-level as points
    for level_info in levels:
        level = level_info['value']
        threshold = 0.005  # Tolerance for iso-surface
        
        # Find points near this SDF level
        mask = np.abs(sdf_grid - level) < threshold
        
        if mask.any():
            # Extract coordinates of points at this level
            pts_x = X[mask]
            pts_y = Y[mask]
            pts_z = Z[mask]
            
            # Plot as scatter points
            ax.scatter(pts_x, pts_y, pts_z, 
                      c=level_info['color'], 
                      alpha=level_info['alpha'],
                      s=level_info['size'],
                      marker='o',
                      edgecolors='none')
    
    # Remove all decorations - just show the object
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Remove axis lines and panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Completely hide axes
    ax.set_axis_off()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set viewing angle (rotate around Z-axis)
    ax.view_init(elev=20, azim=azimuth)
    
    # Set axis limits to keep zoom consistent
    ax.set_xlim([bbox['xmin'], bbox['xmax']])
    ax.set_ylim([bbox['ymin'], bbox['ymax']])
    ax.set_zlim([bbox['zmin'], bbox['zmax']])
    
    plt.tight_layout(pad=0)
    
    # Render to image array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_array = img_array[:, :, :3]  # Drop alpha channel
    
    plt.close(fig)
    
    return img_array


def create_rotating_gif(sdf_grid, bbox, sample_name, output_path, 
                       num_frames=60, fps=30, resolution=(800, 800)):
    """
    Create a rotating GIF of the SDF visualization.
    """
    # Show adaptive levels
    sdf_min = sdf_grid.min()
    sdf_max = sdf_grid.max()
    inner_level = sdf_min * 0.5
    outer_level = min(0.02, sdf_max * 0.2)
    
    print(f"\nAdaptive SDF levels:")
    print(f"  Inner (red):   {inner_level:.4f}m")
    print(f"  Surface (blue): 0.0000m")
    print(f"  Outer (green):  {outer_level:.4f}m")
    
    print(f"\nRendering {num_frames} frames...")
    frames = []
    
    for i in range(num_frames):
        angle = (360 / num_frames) * i
        
        if i % 10 == 0:
            print(f"  Frame {i+1}/{num_frames} (angle: {angle:.1f}°)")
        
        frame = render_sdf_frame(sdf_grid, bbox, angle, 
                                width=resolution[0], height=resolution[1])
        frames.append(frame)
    
    print(f"\nSaving GIF to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    
    # Print file size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"✓ GIF created successfully!")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Duration: {num_frames/fps:.1f} seconds")
    print(f"  Resolution: {resolution[0]}×{resolution[1]}")


def main():
    parser = argparse.ArgumentParser(description='Create rotating GIF of SDF iso-levels')
    
    parser.add_argument('--experiment', '-e', type=str, 
                       default='./deepsdf/experiments/kmelon_latent64/',
                       help='DeepSDF experiment directory')
    parser.add_argument('--checkpoint', '-c', type=str, default='130',
                       help='Checkpoint number')
    parser.add_argument('--sample', '-s', type=str, default='S1',
                       help='Sample name (e.g., S1, S2, S3)')
    parser.add_argument('--grid-density', '-g', type=int, default=64,
                       help='Grid resolution (default: 64)')
    parser.add_argument('--output', '-o', type=str, default='sdf_rotating.gif',
                       help='Output GIF path')
    parser.add_argument('--bbox-size', type=float, default=0.12,
                       help='Half-size of bounding box in meters (default: 0.12, smaller=more zoom)')
    parser.add_argument('--frames', type=int, default=60,
                       help='Number of frames for 360° rotation (default: 60)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--width', type=int, default=800,
                       help='Width in pixels (default: 800)')
    parser.add_argument('--height', type=int, default=800,
                       help='Height in pixels (default: 800)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SDF Rotating GIF Generator")
    print("=" * 60)
    print("Colors: Red=Inner | Blue=Surface | Green=Outer")
    print("=" * 60)
    
    # Load decoder and latent code
    decoder, latent_vector, latent_size = load_decoder_and_latent(
        args.experiment, args.checkpoint, args.sample
    )
    
    # Generate SDF grid
    print(f"\nGenerating SDF grid ({args.grid_density}³ points)...")
    sdf_grid, bbox = generate_sdf_grid(
        decoder, latent_vector, 
        grid_density=args.grid_density,
        bbox_size=args.bbox_size
    )
    
    # Create rotating GIF
    create_rotating_gif(
        sdf_grid, bbox, args.sample, args.output,
        num_frames=args.frames,
        fps=args.fps,
        resolution=(args.width, args.height)
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()

