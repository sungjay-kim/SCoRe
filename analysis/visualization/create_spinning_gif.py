#!/usr/bin/env python3
"""
Create a spinning GIF of a point cloud using matplotlib (headless-compatible).

Usage:
    python create_spinning_gif.py --file path/to/cloud.ply --output spinning.gif
    python create_spinning_gif.py --file path/to/cloud.ply --output spinning.gif --frames 60 --point-size 5
"""

import argparse
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio
from pathlib import Path
from io import BytesIO
from PIL import Image


def render_spinning_gif(pcd_path, output_path, num_frames=72, point_size=8,
                        width=640, height=480, fps=20, bg_color="white",
                        elevation=20):
    """Render a spinning point cloud as a GIF using matplotlib."""
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    print(f"Loaded {len(pcd.points)} points from {pcd_path}")

    points = np.asarray(pcd.points)

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.full((len(points), 3), [0.2, 0.5, 0.8])

    # Center
    center = points.mean(axis=0)
    points = points - center

    # Compute axis limits (equal aspect)
    max_range = np.abs(points).max() * 1.2

    dpi = 100
    fig_w, fig_h = width / dpi, height / dpi

    frames = []
    for i in range(num_frames):
        azim = 360.0 * i / num_frames

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=bg_color)
        ax = fig.add_subplot(111, projection="3d", facecolor=bg_color)

        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors, s=point_size, edgecolors="none", depthshade=True
        )

        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        ax.view_init(elev=elevation, azim=azim)
        ax.set_axis_off()

        # Render to numpy array
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    pad_inches=0, facecolor=bg_color)
        plt.close(fig)
        buf.seek(0)
        img = np.array(Image.open(buf))
        frames.append(img)

        if (i + 1) % 10 == 0:
            print(f"  Rendered frame {i+1}/{num_frames}")

    output_path = Path(output_path)
    imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
    print(f"Saved GIF to {output_path} ({len(frames)} frames, {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Create spinning point cloud GIF")
    parser.add_argument("--file", "-f", type=str, required=True,
                        help="Path to point cloud file (.ply)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output GIF path (default: same dir as input)")
    parser.add_argument("--frames", type=int, default=72,
                        help="Number of frames (default: 72)")
    parser.add_argument("--point-size", type=float, default=8,
                        help="Point size for scatter plot (default: 8)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20,
                        help="Frames per second (default: 20)")
    parser.add_argument("--bg", type=str, default="white",
                        choices=["white", "black"],
                        help="Background color (default: white)")
    parser.add_argument("--elevation", type=float, default=20,
                        help="Camera elevation angle in degrees (default: 20)")
    args = parser.parse_args()

    pcd_path = Path(args.file)
    if not pcd_path.exists():
        print(f"Error: {pcd_path} not found")
        return

    if args.output is None:
        output_path = pcd_path.parent / f"{pcd_path.stem}_spinning.gif"
    else:
        output_path = Path(args.output)

    render_spinning_gif(
        pcd_path, output_path,
        num_frames=args.frames,
        point_size=args.point_size,
        width=args.width,
        height=args.height,
        fps=args.fps,
        bg_color=args.bg,
        elevation=args.elevation,
    )


if __name__ == "__main__":
    main()
