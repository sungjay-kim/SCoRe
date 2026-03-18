#!/usr/bin/env python3
"""
Simple SDF visualization showing multiple iso-levels (outer, surface, inner).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
import sys
from pathlib import Path

# Add root directory to python path for imports
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(root_dir / "corepp") not in sys.path:
    sys.path.append(str(root_dir / "corepp"))

import deepsdf.deep_sdf.workspace as ws
from corepp.sdfrenderer.grid import Grid3D
import corepp.networks.utils as net_utils


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


def visualize_sdf_levels(sdf_grid, bbox, sample_name, output_path='sdf_levels.png'):
    """
    Visualize multiple SDF iso-levels as point clouds.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    grid_density = sdf_grid.shape[0]
    
    # Define iso-levels: inner, surface, outer
    levels = [
        {'value': -0.02, 'color': 'blue', 'alpha': 0.4, 'size': 20},
        {'value': 0.0, 'color': 'red', 'alpha': 0.8, 'size': 30},
        {'value': 0.02, 'color': 'green', 'alpha': 0.4, 'size': 20}
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
            print(f"  Level {level:.3f}m: {mask.sum()} points")
    
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
    
    # Remove axis lines
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Better viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Simple SDF multi-level visualization')
    
    parser.add_argument('--experiment', '-e', type=str, 
                       default='./deepsdf/experiments/kmelon_latent32/',
                       help='DeepSDF experiment directory')
    parser.add_argument('--checkpoint', '-c', type=str, default='100',
                       help='Checkpoint number')
    parser.add_argument('--sample', '-s', type=str, default='S1',
                       help='Sample name (e.g., S1, S2, S3)')
    parser.add_argument('--grid-density', '-g', type=int, default=64,
                       help='Grid resolution (default: 64)')
    parser.add_argument('--output', '-o', type=str, default='sdf_levels.png',
                       help='Output image path')
    parser.add_argument('--bbox-size', type=float, default=0.15,
                       help='Half-size of bounding box in meters (default: 0.15)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SDF Multi-Level Visualization")
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
    
    # Visualize
    print("\nCreating visualization...")
    fig = visualize_sdf_levels(sdf_grid, bbox, args.sample, args.output)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()

