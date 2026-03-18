#!/usr/bin/env python3
"""
Visualize SDF samples from DeepSDF npz files (pos/neg arrays with xyz + sdf).

Usage:
  python3 size_evaluation/visualize_sdf_samples.py \
      --input_dir data/sdf_samples \
      --out_dir data/sdf_samples/vis \
      --num_points 20000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def load_points(npz_path: Path):
    data = np.load(npz_path)
    pts = []
    for key in ["pos", "neg"]:
        if key in data:
            pts.append(data[key])
    if not pts:
        raise ValueError(f"No pos/neg arrays in {npz_path}")
    arr = np.concatenate(pts, axis=0)
    if arr.shape[1] < 4:
        raise ValueError(f"Expected xyz+sdf in last column, got shape {arr.shape} in {npz_path}")
    xyz = arr[:, :3].astype(np.float32)
    sdf = arr[:, 3].astype(np.float32)
    return xyz, sdf


def scatter_plot(xyz: np.ndarray, sdf: np.ndarray, out_path: Path, title: str, num_points: int):
    n = xyz.shape[0]
    if n > num_points > 0:
        rng = np.random.default_rng(seed=1234)
        idx = rng.choice(n, size=num_points, replace=False)
        xyz = xyz[idx]
        sdf = sdf[idx]
    near_mask = (sdf >= -1e-3) & (sdf <= 1e-3)
    # near-surface = green, neg = red, pos = blue
    colors = np.where(near_mask, "green", np.where(sdf < 0, "red", "blue"))

    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=2, alpha=0.6, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # equal-ish aspect
    for_axis = np.concatenate([xyz.min(axis=0), xyz.max(axis=0)]).reshape(2, 3)
    mins = for_axis[0]
    maxs = for_axis[1]
    ranges = maxs - mins
    max_range = ranges.max()
    centers = 0.5 * (mins + maxs)
    ax.set_xlim(centers[0] - max_range / 2, centers[0] + max_range / 2)
    ax.set_ylim(centers[1] - max_range / 2, centers[1] + max_range / 2)
    ax.set_zlim(centers[2] - max_range / 2, centers[2] + max_range / 2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def scatter_plot_html(xyz: np.ndarray, sdf: np.ndarray, out_path: Path, title: str, num_points: int):
    n = xyz.shape[0]
    if n > num_points > 0:
        rng = np.random.default_rng(seed=1234)
        idx = rng.choice(n, size=num_points, replace=False)
        xyz = xyz[idx]
        sdf = sdf[idx]
    near_mask = (sdf >= -1e-3) & (sdf <= 1e-3)
    colors = np.where(near_mask, "green", np.where(sdf < 0, "red", "blue"))

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=dict(size=2, color=colors, opacity=0.65),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)


def main():
    ap = argparse.ArgumentParser(description="Visualize SDF samples from npz (pos/neg xyz+sdf).")
    ap.add_argument("--input_dir", default="data/sdf_samples", help="Directory containing *_samples.npz files.")
    ap.add_argument("--out_dir", default="data/sdf_samples/vis", help="Directory to save PNGs.")
    ap.add_argument("--num_points", type=int, default=20000, help="Max points to plot per file.")
    ap.add_argument("--html", action="store_true", help="Also write interactive HTML (Plotly) scatter.")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*_samples.npz"))
    if not files:
        raise SystemExit(f"No *_samples.npz found in {in_dir}")

    for f in files:
        try:
            xyz, sdf = load_points(f)
        except Exception as e:
            print(f"[skip] {f.name}: {e}")
            continue
        title = f.name
        out_path_png = out_dir / f"{f.stem}.png"
        scatter_plot(xyz, sdf, out_path_png, title, args.num_points)
        print(f"Wrote: {out_path_png}")
        if args.html:
            out_path_html = out_dir / f"{f.stem}.html"
            scatter_plot_html(xyz, sdf, out_path_html, title, args.num_points)
            print(f"Wrote: {out_path_html}")


if __name__ == "__main__":
    main()


