#!/usr/bin/env python3
"""
Compare hand-measured sizes vs PCA-measured sizes from GT meshes.

This script helps identify systematic bias/offset between manual measurements
and PCA-based measurements from 3D meshes.

Usage:
    python3 compare_hand_vs_mesh_size.py \
        --gt_csv ./data/3DKmelonDemo/size_data.csv \
        --gt_mesh_dir ./data/3DKmelonDemo \
        --output ./comparison_hand_vs_mesh.csv

PCD mode (PCA dimensions from point clouds; compares hand height/width vs PCA length/width):
    python3 compare_hand_vs_mesh_size.py \
        --hand_csv ./data/3DKmelonDemo/phenobox_size_data.csv \
        --pcd_root ./data/kmelon_centered \
        --output ./comparison_hand_vs_pcd.csv
"""

import argparse
import pandas as pd
import numpy as np
import open3d as o3d
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import re


def compute_pca_dimensions(points):
    """
    Compute length and width from 3D points using PCA.
    
    Args:
        points: numpy array of shape (N, 3)
    
    Returns:
        length_mm, width_mm, height_mm
    """
    if points is None or len(points) < 3:
        return None, None, None
    
    # Center the points
    centroid = points.mean(axis=0)
    centered = points - centroid
    
    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(centered)
    
    # Transform to principal component space
    transformed = pca.transform(centered)
    
    # Get extents along each principal axis
    extents = np.max(transformed, axis=0) - np.min(transformed, axis=0)
    
    # Sort to get length (longest), width (2nd longest), height (3rd)
    sorted_indices = np.argsort(extents)[::-1]
    sorted_extents = extents[sorted_indices]
    
    # Convert from meters to millimeters
    length_mm = sorted_extents[0] * 1000
    width_mm = sorted_extents[1] * 1000
    height_mm = sorted_extents[2] * 1000
    
    return length_mm, width_mm, height_mm


def load_hand_measurements(csv_path):
    """Load hand measurements from CSV."""
    df = pd.read_csv(csv_path)
    
    # Detect length/width columns
    length_col = None
    width_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'length' in col_lower or 'height' in col_lower:
            if length_col is None:
                length_col = col
        if 'width' in col_lower:
            width_col = col
    
    if length_col is None or width_col is None:
        raise ValueError(f"Could not find length/width columns. Available: {list(df.columns)}")
    
    print(f"[Hand] Using columns: length='{length_col}', width='{width_col}'")
    
    # Create dictionary: (sample, date) -> (length, width)
    hand_dict = {}
    for _, row in df.iterrows():
        sample = str(int(row['sample'])).strip()
        date = str(int(row['date'])).strip()
        length = float(row[length_col])
        width = float(row[width_col])
        
        # Skip zero measurements (missing data)
        if length == 0 or width == 0:
            continue
        
        hand_dict[(sample, date)] = (length, width)
    
    print(f"[Hand] Loaded {len(hand_dict)} measurements")
    return hand_dict


def _normalize_date_digits(s: str) -> str:
    """
    Normalize dates like:
      - 20250525 -> 20250525
      - 2025.8.4 -> 20250804
      - 2025-08-04 -> 20250804
    """
    raw = str(s or "").strip()
    # split by non-digits first (handles dotted/slashed)
    parts = [p for p in re.split(r"[^\d]+", raw) if p]
    if len(parts) == 3 and len(parts[0]) == 4:
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{y:04d}{m:02d}{d:02d}"
        except Exception:
            pass
    digits = re.sub(r"\D+", "", raw)
    return digits


def load_phenobox_size_measurements(csv_path: str) -> pd.DataFrame:
    """
    Load phenobox_size_data.csv (hand measurements). Expects at least:
      - height, width, date
    Preferably also:
      - S_number (numeric string) and/or S_name, sample_id
    """
    # Read as strings to avoid pandas converting IDs like "72" -> 72.0 when there are missing values.
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    # Normalize columns
    cols_lower = {c.lower(): c for c in df.columns}
    # Ensure required columns exist
    for req in ["height", "width", "date"]:
        if req not in cols_lower:
            raise ValueError(f"Missing required column '{req}' in {csv_path}. Columns={list(df.columns)}")

    # Normalize date digits for joining/debugging
    date_col = cols_lower["date"]
    df["date_digits"] = df[date_col].astype(str).map(_normalize_date_digits)

    # Coerce to numeric and drop missing/zero measurements
    h_col = cols_lower["height"]
    w_col = cols_lower["width"]
    df["hand_height_mm"] = pd.to_numeric(df[h_col], errors="coerce")
    df["hand_width_mm"] = pd.to_numeric(df[w_col], errors="coerce")
    df = df.dropna(subset=["hand_height_mm", "hand_width_mm"])
    df = df[(df["hand_height_mm"] != 0) & (df["hand_width_mm"] != 0)]
    return df


def compute_pcd_pca_dimensions_mm(ply_path: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute PCA-based dimensions from a point cloud PLY.
    Returns (length_mm, width_mm, height_mm) where:
      - length_mm: largest extent
      - width_mm:  second largest extent
      - height_mm: third extent
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    if pts is None or len(pts) < 3:
        return None, None, None
    return compute_pca_dimensions(pts)


def compare_hand_vs_pcd_bbox(hand_csv: str, pcd_root: str, output_csv: str) -> None:
    """
    Compare hand height/width (mm) vs PCA dimensions from point clouds under:
      <pcd_root>/S{S_number}/laser/fruit.ply

    Hand height is compared to PCA length (largest), hand width to PCA width (second largest).
    """
    df_hand = load_phenobox_size_measurements(hand_csv)
    if "S_number" not in df_hand.columns:
        raise ValueError(
            "phenobox_size_data.csv must contain 'S_number' column to map to point clouds. "
            "Please add it from link_map_mesh.csv first."
        )

    pcd_root = Path(pcd_root)
    cache: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    results = []

    for _, row in tqdm(df_hand.iterrows(), total=len(df_hand), desc="Processing rows"):
        s_num = str(row.get("S_number", "")).strip()
        if not s_num:
            continue
        # guard against any accidental float-like strings
        if s_num.endswith(".0"):
            s_num = s_num[:-2]
        sid = f"S{s_num}"
        sid_unhealthy = f"{sid}_unhealthy"

        # Determine which folder exists; exclude unhealthy samples robustly.
        dir_healthy = pcd_root / sid
        dir_unhealthy = pcd_root / sid_unhealthy

        # If hand table explicitly marks unhealthy, skip.
        sname_in = str(row.get("S_name", "") or "")
        if "_unhealthy" in sname_in:
            continue

        # If only unhealthy folder exists, skip.
        if (not dir_healthy.exists()) and dir_unhealthy.exists():
            continue

        # Prefer healthy folder when present (even if an unhealthy folder also exists)
        pcd_dir = dir_healthy if dir_healthy.exists() else dir_unhealthy
        ply_path = pcd_dir / "laser" / "fruit.ply"
        if not ply_path.exists():
            continue

        # Normalize S_name in outputs: unhealthy samples get the _unhealthy suffix
        sname_out = f"{sid_unhealthy}.ply" if pcd_dir.name.endswith("_unhealthy") else f"{sid}.ply"

        if s_num not in cache:
            cache[s_num] = compute_pcd_pca_dimensions_mm(ply_path)
        length_mm, width_mm, height_mm = cache[s_num]
        if length_mm is None or width_mm is None:
            continue

        hand_h = float(row["hand_height_mm"])
        hand_w = float(row["hand_width_mm"])

        # Compare hand height vs PCA length; hand width vs PCA width
        height_diff = length_mm - hand_h
        width_diff = width_mm - hand_w
        height_diff_pct = (height_diff / hand_h * 100.0) if hand_h else None
        width_diff_pct = (width_diff / hand_w * 100.0) if hand_w else None

        results.append(
            {
                "S_number": s_num,
                "S_name": sname_out,
                "sample_id": row.get("sample_id", ""),
                "date": row.get("date", ""),
                "hand_height_mm": round(hand_h, 2),
                "hand_width_mm": round(hand_w, 2),
                "pcd_pca_length_mm": round(float(length_mm), 2),
                "pcd_pca_width_mm": round(float(width_mm), 2),
                "pcd_pca_thickness_mm": round(float(height_mm), 2) if height_mm is not None else None,
                "height_diff_mm": round(float(height_diff), 2),
                "width_diff_mm": round(float(width_diff), 2),
                "height_diff_pct": round(float(height_diff_pct), 2) if height_diff_pct is not None else None,
                "width_diff_pct": round(float(width_diff_pct), 2) if width_diff_pct is not None else None,
                "pcd_path": str(ply_path),
            }
        )

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("[Warning] No matching rows found (check S_number mapping and pcd_root path).")
        return

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"Compared rows: {len(df)}")

    def _mape(y_true: pd.Series, y_pred: pd.Series) -> float:
        y_true = y_true.astype(float)
        y_pred = y_pred.astype(float)
        mask = y_true != 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        return float((abs((y_pred - y_true) / y_true)).mean() * 100.0)

    def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
        y_true = y_true.astype(float)
        y_pred = y_pred.astype(float)
        return float((((y_true - y_pred) ** 2).mean()) ** 0.5)

    def _r2(y_true: pd.Series, y_pred: pd.Series) -> float:
        y_true = y_true.astype(float)
        y_pred = y_pred.astype(float)
        ss_res = float(((y_true - y_pred) ** 2).sum())
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
        return float(1.0 - (ss_res / ss_tot if ss_tot != 0 else float("nan")))

    # Metrics
    h_true = df["hand_height_mm"].astype(float)
    h_pred = df["pcd_pca_length_mm"].astype(float)
    w_true = df["hand_width_mm"].astype(float)
    w_pred = df["pcd_pca_width_mm"].astype(float)

    h_rmse = _rmse(h_true, h_pred)
    h_mape = _mape(h_true, h_pred)
    h_r2 = _r2(h_true, h_pred)

    w_rmse = _rmse(w_true, w_pred)
    w_mape = _mape(w_true, w_pred)
    w_r2 = _r2(w_true, w_pred)

    print("\nHeight Bias (PCD PCA length - Hand height):")
    print(f"  RMSE:              {h_rmse:7.2f} mm")
    print(f"  MAPE:              {h_mape:7.2f} %")
    print(f"  R^2:               {h_r2:7.3f}")

    print("\nWidth Bias (PCD PCA width - Hand width):")
    print(f"  RMSE:              {w_rmse:7.2f} mm")
    print(f"  MAPE:              {w_mape:7.2f} %")
    print(f"  R^2:               {w_r2:7.3f}")

    # Write summary text (PCD mode): do NOT overwrite the legacy mesh summary filename,
    # because it can be confusing and can continuously reappear during runs.
    summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
    summary_text = (
        "COMPARISON SUMMARY: Hand Measurements vs Point Cloud (PCA)\n"
        "(Healthy-only rows from phenobox_size_data.csv (excluding *_unhealthy) with valid height/width and matching point clouds)\n"
        "================================================================================\n\n"
        f"Total samples compared: {len(df)}\n\n"
        "Height (PCD PCA length vs Hand height):\n"
        f"  RMSE:                {h_rmse:7.2f} mm\n"
        f"  MAPE:                {h_mape:7.2f} %\n"
        f"  R^2:                 {h_r2:7.3f}\n\n"
        "Width (PCD PCA width vs Hand width):\n"
        f"  RMSE:                {w_rmse:7.2f} mm\n"
        f"  MAPE:                {w_mape:7.2f} %\n"
        f"  R^2:                 {w_r2:7.3f}\n\n"
        "Notes:\n"
        "  - Point clouds are loaded from: data/kmelon_centered/S*/laser/fruit.ply (meters)\n"
        "  - PCA extents are converted to millimeters (meters * 1000)\n"
        "  - Hand 'height' is compared to PCA 'length' (largest extent)\n"
        "  - Hand 'width' is compared to PCA 'width' (second largest extent)\n"
        "  - Unhealthy samples are excluded (either S_name contains _unhealthy, or only S*_unhealthy folder exists)\n"
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Summary saved to: {summary_path}")

    # Save per-output-stem plot (PCD mode)
    stem_plot = output_path.parent / f"{output_path.stem}_plots.png"
    create_comparison_plots_pcd(df, stem_plot)
    print(f"Plots saved to: {stem_plot}")


def compare_measurements(hand_csv, gt_mesh_dir, output_csv):
    """Compare hand measurements with GT mesh measurements."""
    
    gt_mesh_dir = Path(gt_mesh_dir)
    
    # Load hand measurements
    hand_dict = load_hand_measurements(hand_csv)
    
    # Find all unique samples
    samples = set(sample for sample, _ in hand_dict.keys())
    print(f"[Mesh] Processing {len(samples)} unique samples")
    
    results = []
    
    for sample in tqdm(sorted(samples), desc="Processing samples"):
        # Try to load GT mesh
        mesh_path = gt_mesh_dir / "2_sfm" / "1_mesh" / f"S{sample}.ply"
        
        if not mesh_path.exists():
            continue
        
        try:
            # Load mesh
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if not mesh.has_vertices():
                continue
            
            # Compute PCA dimensions from mesh
            vertices = np.asarray(mesh.vertices)
            mesh_length, mesh_width, mesh_height = compute_pca_dimensions(vertices)
            
            if mesh_length is None:
                continue
            
            # Scale by 0.2 (based on user requirement)
            mesh_length *= 0.2
            mesh_width *= 0.2
            mesh_height *= 0.2
            
            # Find all dates for this sample
            sample_dates = [(s, d) for s, d in hand_dict.keys() if s == sample]
            
            if not sample_dates:
                continue
            
            # Get only the last date (latest measurement) for this sample
            # Sort dates and take the last one
            sorted_dates = sorted(sample_dates, key=lambda x: int(x[1]))
            last_sample, last_date = sorted_dates[-1]
            
            hand_length, hand_width = hand_dict[(last_sample, last_date)]
            
            length_diff = mesh_length - hand_length
            width_diff = mesh_width - hand_width
            
            length_diff_pct = (length_diff / hand_length * 100) if hand_length != 0 else None
            width_diff_pct = (width_diff / hand_width * 100) if hand_width != 0 else None
            
            result = {
                'sample': f"S{sample}",
                'date': last_date,
                'hand_length_mm': round(hand_length, 2),
                'hand_width_mm': round(hand_width, 2),
                'mesh_length_mm': round(mesh_length, 2),
                'mesh_width_mm': round(mesh_width, 2),
                'mesh_height_mm': round(mesh_height, 2),
                'length_diff_mm': round(length_diff, 2),
                'width_diff_mm': round(width_diff, 2),
                'length_diff_pct': round(length_diff_pct, 2) if length_diff_pct is not None else None,
                'width_diff_pct': round(width_diff_pct, 2) if width_diff_pct is not None else None,
            }
            results.append(result)
        
        except Exception as e:
            print(f"[Error] Failed to process sample {sample}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("[Warning] No matching samples found!")
        return
    
    # Calculate summary statistics
    print("\n" + "="*80)
    print("COMPARISON SUMMARY: Hand Measurements vs GT Mesh (PCA)")
    print("(Using only the last date's measurement for each sample)")
    print("="*80)
    print(f"Total samples compared: {len(df)}")
    
    print(f"\nLength Bias (Mesh - Hand):")
    print(f"  Mean difference:     {df['length_diff_mm'].mean():7.2f} mm")
    print(f"  Median difference:   {df['length_diff_mm'].median():7.2f} mm")
    print(f"  Std Dev:             {df['length_diff_mm'].std():7.2f} mm")
    print(f"  Mean difference %:   {df['length_diff_pct'].mean():7.2f} %")
    
    print(f"\nWidth Bias (Mesh - Hand):")
    print(f"  Mean difference:     {df['width_diff_mm'].mean():7.2f} mm")
    print(f"  Median difference:   {df['width_diff_mm'].median():7.2f} mm")
    print(f"  Std Dev:             {df['width_diff_mm'].std():7.2f} mm")
    print(f"  Mean difference %:   {df['width_diff_pct'].mean():7.2f} %")
    print("="*80)
    
    # Save results
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Save summary
    summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("COMPARISON SUMMARY: Hand Measurements vs GT Mesh (PCA)\n")
        f.write("(Using only the last date's measurement for each sample)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total samples compared: {len(df)}\n\n")
        
        f.write("Length Bias (Mesh - Hand):\n")
        f.write(f"  Mean difference:     {df['length_diff_mm'].mean():7.2f} mm\n")
        f.write(f"  Median difference:   {df['length_diff_mm'].median():7.2f} mm\n")
        f.write(f"  Std Dev:             {df['length_diff_mm'].std():7.2f} mm\n")
        f.write(f"  Mean difference %:   {df['length_diff_pct'].mean():7.2f} %\n\n")
        
        f.write("Width Bias (Mesh - Hand):\n")
        f.write(f"  Mean difference:     {df['width_diff_mm'].mean():7.2f} mm\n")
        f.write(f"  Median difference:   {df['width_diff_mm'].median():7.2f} mm\n")
        f.write(f"  Std Dev:             {df['width_diff_mm'].std():7.2f} mm\n")
        f.write(f"  Mean difference %:   {df['width_diff_pct'].mean():7.2f} %\n")
    
    print(f"Summary saved to: {summary_path}")
    
    # Create scatter plots
    create_comparison_plots(df, output_path.parent / f"{output_path.stem}_plots.png")


def create_comparison_plots(df, output_path):
    """Create scatter plots comparing hand vs mesh measurements."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Length comparison
    ax = axes[0, 0]
    ax.scatter(df['hand_length_mm'], df['mesh_length_mm'], alpha=0.6)
    ax.plot([df['hand_length_mm'].min(), df['hand_length_mm'].max()],
            [df['hand_length_mm'].min(), df['hand_length_mm'].max()],
            'r--', label='y=x')
    ax.set_xlabel('Hand Measured Length (mm)')
    ax.set_ylabel('Mesh PCA Length (mm)')
    ax.set_title('Length Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Width comparison
    ax = axes[0, 1]
    ax.scatter(df['hand_width_mm'], df['mesh_width_mm'], alpha=0.6)
    ax.plot([df['hand_width_mm'].min(), df['hand_width_mm'].max()],
            [df['hand_width_mm'].min(), df['hand_width_mm'].max()],
            'r--', label='y=x')
    ax.set_xlabel('Hand Measured Width (mm)')
    ax.set_ylabel('Mesh PCA Width (mm)')
    ax.set_title('Width Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Length difference histogram
    ax = axes[1, 0]
    ax.hist(df['length_diff_mm'], bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(df['length_diff_mm'].mean(), color='r', linestyle='--', 
               label=f"Mean: {df['length_diff_mm'].mean():.2f} mm")
    ax.set_xlabel('Length Difference (Mesh - Hand) [mm]')
    ax.set_ylabel('Frequency')
    ax.set_title('Length Bias Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Width difference histogram
    ax = axes[1, 1]
    ax.hist(df['width_diff_mm'], bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(df['width_diff_mm'].mean(), color='r', linestyle='--',
               label=f"Mean: {df['width_diff_mm'].mean():.2f} mm")
    ax.set_xlabel('Width Difference (Mesh - Hand) [mm]')
    ax.set_ylabel('Frequency')
    ax.set_title('Width Bias Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {output_path}")
    plt.close()


def create_comparison_plots_pcd(df: pd.DataFrame, output_path: Path) -> None:
    """Create scatter + histogram plots for hand vs point-cloud PCA comparisons."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Height (hand) vs PCA length
    ax = axes[0, 0]
    ax.scatter(df["hand_height_mm"], df["pcd_pca_length_mm"], alpha=0.6)
    lo = min(df["hand_height_mm"].min(), df["pcd_pca_length_mm"].min())
    hi = max(df["hand_height_mm"].max(), df["pcd_pca_length_mm"].max())
    ax.plot([lo, hi], [lo, hi], "r--", label="y=x")
    ax.set_xlabel("Hand Height (mm)")
    ax.set_ylabel("PCD PCA Length (mm)")
    ax.set_title("Height vs PCA Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Width (hand) vs PCA width
    ax = axes[0, 1]
    ax.scatter(df["hand_width_mm"], df["pcd_pca_width_mm"], alpha=0.6)
    lo = min(df["hand_width_mm"].min(), df["pcd_pca_width_mm"].min())
    hi = max(df["hand_width_mm"].max(), df["pcd_pca_width_mm"].max())
    ax.plot([lo, hi], [lo, hi], "r--", label="y=x")
    ax.set_xlabel("Hand Width (mm)")
    ax.set_ylabel("PCD PCA Width (mm)")
    ax.set_title("Width vs PCA Width")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Height diff histogram
    ax = axes[1, 0]
    ax.hist(df["height_diff_mm"], bins=30, alpha=0.7, edgecolor="black")
    ax.axvline(
        df["height_diff_mm"].mean(),
        color="r",
        linestyle="--",
        label=f"Mean: {df['height_diff_mm'].mean():.2f} mm",
    )
    ax.set_xlabel("Height Diff (PCD PCA length - Hand) [mm]")
    ax.set_ylabel("Frequency")
    ax.set_title("Height Bias Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Width diff histogram
    ax = axes[1, 1]
    ax.hist(df["width_diff_mm"], bins=30, alpha=0.7, edgecolor="black")
    ax.axvline(
        df["width_diff_mm"].mean(),
        color="r",
        linestyle="--",
        label=f"Mean: {df['width_diff_mm'].mean():.2f} mm",
    )
    ax.set_xlabel("Width Diff (PCD PCA width - Hand) [mm]")
    ax.set_ylabel("Frequency")
    ax.set_title("Width Bias Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare hand measurements vs GT mesh PCA measurements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python3 compare_hand_vs_mesh_size.py \\
      --gt_csv ./data/3DKmelonDemo/size_data.csv \\
      --gt_mesh_dir ./data/3DKmelonDemo \\
      --output ./comparison_hand_vs_mesh.csv
        """
    )
    
    parser.add_argument(
        '--gt_csv',
        type=str,
        required=False,
        help='[Mesh mode] Path to ground truth CSV with hand measurements'
    )
    
    parser.add_argument(
        '--gt_mesh_dir',
        type=str,
        required=False,
        help='[Mesh mode] Directory containing GT meshes (looks for 2_sfm/1_mesh/*.ply)'
    )

    parser.add_argument(
        '--hand_csv',
        type=str,
        required=False,
        help='[PCD mode] Path to phenobox_size_data.csv (hand height/width in mm, plus S_number mapping)',
    )

    parser.add_argument(
        '--pcd_root',
        type=str,
        required=False,
        help='[PCD mode] Root directory containing point clouds as <pcd_root>/S*/laser/fruit.ply',
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for comparison CSV'
    )
    
    args = parser.parse_args()

    # Choose mode
    if args.hand_csv and args.pcd_root:
        compare_hand_vs_pcd_bbox(hand_csv=args.hand_csv, pcd_root=args.pcd_root, output_csv=args.output)
        return

    if not (args.gt_csv and args.gt_mesh_dir):
        raise SystemExit("Provide either --hand_csv + --pcd_root (PCD mode), or --gt_csv + --gt_mesh_dir (mesh mode).")

    compare_measurements(hand_csv=args.gt_csv, gt_mesh_dir=args.gt_mesh_dir, output_csv=args.output)


if __name__ == '__main__':
    main()

