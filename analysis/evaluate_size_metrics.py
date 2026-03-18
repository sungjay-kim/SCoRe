#!/usr/bin/env python3
"""
Evaluate length and width metrics for fruit meshes.

This script:
1. Loads predicted meshes from test output
2. Computes length (longest axis) and width (longest perpendicular axis) using PCA
3. Compares against ground truth from size_data.csv
4. Outputs evaluation CSV with errors

Usage:
    python3 evaluate_size_metrics.py \
        --mesh_dir ./experiments/kmelon_latent32/RUN_DIR/inf/meshes \
        --gt_csv ./data/3DKmelonDemo/size_data.csv \
        --output ./experiments/kmelon_latent32/RUN_DIR/size_evaluation.csv
"""

import argparse
import pandas as pd
import numpy as np
import open3d as o3d
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import pickle


def compute_mesh_dimensions(mesh):
    """
    Compute length and width of a mesh using PCA.
    
    Length: extent along the first principal component (longest axis)
    Width: extent along the second principal component (longest perpendicular axis)
    
    Args:
        mesh: open3d.geometry.TriangleMesh
    
    Returns:
        length_mm: float, length in millimeters
        width_mm: float, width in millimeters
        dimensions: dict with all 3 dimensions and axes
    """
    if mesh is None or not mesh.has_vertices():
        return None, None, None
    
    # Get vertices
    vertices = np.asarray(mesh.vertices)
    
    if len(vertices) < 3:
        return None, None, None
    
    # Center the points
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    
    # Perform PCA to find principal axes
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
    
    dimensions = {
        'length_mm': length_mm,
        'width_mm': width_mm,
        'height_mm': height_mm,
        'principal_axes': pca.components_[sorted_indices],
        'extents_m': sorted_extents,
        'centroid': centroid,
        'explained_variance_ratio': pca.explained_variance_ratio_[sorted_indices]
    }
    
    return length_mm, width_mm, dimensions


def parse_mesh_filename(filename):
    """
    Parse mesh filename to extract sample and frame info.
    
    Expected formats:
        - {sample}_{frame}_mesh.ply
        - {sample}_{date}_{view}_mesh.ply
    
    Returns:
        sample: str (e.g., 'S01')
        frame: str (e.g., '20240615_0')
        date: str (e.g., '20240615')
    """
    stem = Path(filename).stem
    
    # Remove '_mesh' suffix if present
    if stem.endswith('_mesh'):
        stem = stem[:-5]
    
    parts = stem.split('_')
    
    if len(parts) >= 2:
        sample = parts[0]  # e.g., 'S01'
        # Try to extract date (format: YYYYMMDD)
        date = None
        for part in parts[1:]:
            if len(part) == 8 and part.isdigit():
                date = part
                break
        
        # Reconstruct frame_id (everything after sample)
        frame = '_'.join(parts[1:])
        
        return sample, frame, date
    
    return None, None, None


def train_correction_model(comparison_csv):
    """
    Train a linear regression model to correct PCA measurements to match hand measurements.
    
    Args:
        comparison_csv: Path to comparison CSV from compare_hand_vs_mesh_size.py
    
    Returns:
        dict with 'length_model' and 'width_model' (sklearn LinearRegression)
    """
    df = pd.read_csv(comparison_csv)
    
    if len(df) == 0:
        print("[Warning] No data in comparison CSV. Cannot train correction model.")
        return None
    
    # Train length correction: mesh_length -> hand_length
    X_length = df[['mesh_length_mm']].values
    y_length = df['hand_length_mm'].values
    length_model = LinearRegression()
    length_model.fit(X_length, y_length)
    
    # Train width correction: mesh_width -> hand_width
    X_width = df[['mesh_width_mm']].values
    y_width = df['hand_width_mm'].values
    width_model = LinearRegression()
    width_model.fit(X_width, y_width)
    
    # Print correction parameters
    print("\n[Correction Model] Trained from hand vs mesh comparison:")
    print(f"  Length: corrected = {length_model.coef_[0]:.4f} * measured + {length_model.intercept_:.4f}")
    print(f"  Width:  corrected = {width_model.coef_[0]:.4f} * measured + {width_model.intercept_:.4f}")
    
    return {
        'length_model': length_model,
        'width_model': width_model,
        'n_samples': len(df)
    }


def apply_correction(length_mm, width_mm, correction_model):
    """
    Apply correction model to PCA measurements.
    
    Args:
        length_mm: Measured length in mm
        width_mm: Measured width in mm
        correction_model: Dict with length_model and width_model
    
    Returns:
        corrected_length_mm, corrected_width_mm
    """
    if correction_model is None:
        return length_mm, width_mm
    
    corrected_length = correction_model['length_model'].predict([[length_mm]])[0]
    corrected_width = correction_model['width_model'].predict([[width_mm]])[0]
    
    return corrected_length, corrected_width


def load_ground_truth(csv_path):
    """
    Load ground truth size data.
    
    Expected columns: sample, date, length_mm, width_mm
    """
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required = ['sample', 'date']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in GT CSV: {missing}")
    
    # Handle different possible column names
    length_col = None
    width_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        # Length can be called 'length' or 'height' (melon height = length axis)
        if 'length' in col_lower or 'height' in col_lower:
            if length_col is None:  # Take the first match
                length_col = col
        if 'width' in col_lower:
            width_col = col
    
    if length_col is None or width_col is None:
        raise ValueError(f"Could not find length/width columns in GT CSV. Available: {list(df.columns)}\n"
                        f"Note: Looking for columns containing 'length', 'height', or 'width' (case-insensitive)")
    
    print(f"[GT] Using columns: length='{length_col}', width='{width_col}'")
    
    # Create lookup dictionary: (sample, date) -> (length, width)
    gt_dict = {}
    for _, row in df.iterrows():
        # Convert to string and strip whitespace
        sample = str(int(row['sample'])).strip()  # Remove any decimals from pandas reading
        date = str(int(row['date'])).strip()  # Ensure date is clean integer string
        length = float(row[length_col])
        width = float(row[width_col])
        
        # Skip entries with zero length or width (missing data)
        if length == 0 or width == 0:
            continue
            
        gt_dict[(sample, date)] = (length, width)
    
    print(f"[GT] Loaded {len(gt_dict)} ground truth entries")
    return gt_dict, length_col, width_col


def create_axis_arrow(origin, direction, length, color):
    """
    Create an arrow mesh to visualize an axis.
    
    Args:
        origin: 3D point where arrow starts
        direction: unit direction vector
        length: length of the arrow
        color: RGB color (0-1)
    
    Returns:
        arrow mesh
    """
    # Create a cylinder for the arrow shaft
    cylinder_radius = length * 0.01
    cylinder_height = length * 0.8
    
    # Create a cone for the arrow head
    cone_radius = length * 0.03
    cone_height = length * 0.2
    
    # Create cylinder and cone
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=cylinder_radius, 
        height=cylinder_height
    )
    cone = o3d.geometry.TriangleMesh.create_cone(
        radius=cone_radius, 
        height=cone_height
    )
    
    # Move cone to end of cylinder
    cone.translate([0, 0, cylinder_height/2 + cone_height/2])
    
    # Combine cylinder and cone
    arrow = cylinder + cone
    arrow.paint_uniform_color(color)
    
    # Align arrow with direction vector
    # Default arrow points along Z axis [0, 0, 1]
    default_dir = np.array([0, 0, 1])
    
    # Calculate rotation to align with target direction
    direction = direction / np.linalg.norm(direction)  # Ensure unit vector
    
    # Use Rodrigues' rotation formula
    if not np.allclose(default_dir, direction):
        if np.allclose(default_dir, -direction):
            # 180 degree rotation
            rotation_axis = np.array([1, 0, 0])
            angle = np.pi
        else:
            rotation_axis = np.cross(default_dir, direction)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.dot(default_dir, direction))
        
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        arrow.rotate(R, center=[0, 0, 0])
    
    # Translate to origin
    arrow.translate(origin)
    
    return arrow


def evaluate_size_metrics(mesh_dir, gt_csv, output_csv, mesh_format='ply', save_visualizations=False, correction_csv=None):
    """
    Evaluate length and width for all meshes and compare to ground truth.
    
    Args:
        mesh_dir: Directory containing mesh files
        gt_csv: Path to ground truth CSV
        output_csv: Path to save evaluation results
        mesh_format: Mesh file format (default: 'ply')
        save_visualizations: If True, save meshes with PCA axes overlayed
        correction_csv: Optional path to hand vs mesh comparison CSV for regression correction
    """
    mesh_dir = Path(mesh_dir)
    
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")
    
    # Load correction model if provided
    correction_model = None
    if correction_csv is not None:
        correction_csv_path = Path(correction_csv)
        if correction_csv_path.exists():
            correction_model = train_correction_model(correction_csv)
        else:
            print(f"[Warning] Correction CSV not found: {correction_csv}. Proceeding without correction.")
    
    # Load ground truth
    gt_dict, gt_length_col, gt_width_col = load_ground_truth(gt_csv)
    
    # Find all mesh files
    mesh_files = list(mesh_dir.glob(f"*.{mesh_format}"))
    print(f"[Eval] Found {len(mesh_files)} mesh files in {mesh_dir}")
    
    if len(mesh_files) == 0:
        print(f"[Eval] Warning: No mesh files found!")
        return
    
    # Create visualization directory if needed
    vis_dir = None
    if save_visualizations:
        vis_dir = Path(output_csv).parent / "axis_visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Eval] Saving axis visualizations to: {vis_dir}")
    
    # Results storage
    results = []
    
    # Process each mesh
    for mesh_path in tqdm(mesh_files, desc="Processing meshes"):
        # Parse filename
        sample, frame, date = parse_mesh_filename(mesh_path.name)
        
        if sample is None or date is None:
            print(f"[Eval] Warning: Could not parse filename: {mesh_path.name}")
            continue
        
        # Load mesh
        try:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        except Exception as e:
            print(f"[Eval] Failed to load mesh {mesh_path.name}: {e}")
            continue
        
        # Compute dimensions
        length_mm, width_mm, dims = compute_mesh_dimensions(mesh)
        
        if length_mm is None:
            print(f"[Eval] Failed to compute dimensions for {mesh_path.name}")
            continue
        
        # Apply correction if model is available
        length_mm_corrected, width_mm_corrected = apply_correction(length_mm, width_mm, correction_model)
        
        # Save visualization with axes if requested
        if save_visualizations and dims is not None:
            try:
                # Get centroid and axes from dimensions
                centroid = dims['centroid']
                axes = dims['principal_axes']  # Already sorted by extent
                extents_m = dims['extents_m']
                
                # Create arrows for length and width axes
                # Length (red) - longest axis
                length_arrow = create_axis_arrow(
                    origin=centroid,
                    direction=axes[0],
                    length=extents_m[0],
                    color=[1, 0, 0]  # Red
                )
                
                # Width (green) - second longest axis
                width_arrow = create_axis_arrow(
                    origin=centroid,
                    direction=axes[1],
                    length=extents_m[1],
                    color=[0, 1, 0]  # Green
                )
                
                # Combine mesh with both axes into a single file
                vis_mesh = o3d.geometry.TriangleMesh(mesh)
                vis_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
                
                # Combine all components
                combined = vis_mesh + length_arrow + width_arrow
                
                # Save combined visualization
                vis_filename = vis_dir / f"{mesh_path.stem}_with_axes.ply"
                o3d.io.write_triangle_mesh(str(vis_filename), combined)
                
            except Exception as e:
                print(f"[Eval] Failed to save visualization for {mesh_path.name}: {e}")
        
        # Look up ground truth from CSV (hand measurements)
        # Normalize sample ID: strip 'S' prefix if present to match CSV format
        sample_normalized = sample.lstrip('S') if sample.startswith('S') else sample
        gt_key = (sample_normalized, date)
        gt_length = gt_width = None
        length_error = width_error = None
        length_error_corrected = width_error_corrected = None
        has_gt = False
        
        if gt_key in gt_dict:
            gt_length, gt_width = gt_dict[gt_key]
            # Error using corrected values
            length_error_corrected = length_mm_corrected - gt_length
            width_error_corrected = width_mm_corrected - gt_width
            # Error using raw values (for comparison)
            length_error = length_mm - gt_length
            width_error = width_mm - gt_width
            has_gt = True
        
        # Store results
        result = {
            'sample': sample,
            'date': date,
            'frame': frame,
            'mesh_file': mesh_path.name,
            'pred_length_mm': round(length_mm, 2),
            'pred_width_mm': round(width_mm, 2),
            'pred_height_mm': round(dims['height_mm'], 2),
            'pred_length_corrected_mm': round(length_mm_corrected, 2) if correction_model else None,
            'pred_width_corrected_mm': round(width_mm_corrected, 2) if correction_model else None,
            'gt_length_mm': round(gt_length, 2) if gt_length is not None else None,
            'gt_width_mm': round(gt_width, 2) if gt_width is not None else None,
            'length_error_mm': round(length_error, 2) if length_error is not None else None,
            'width_error_mm': round(width_error, 2) if width_error is not None else None,
            'length_error_corrected_mm': round(length_error_corrected, 2) if length_error_corrected is not None else None,
            'width_error_corrected_mm': round(width_error_corrected, 2) if width_error_corrected is not None else None,
            'length_error_pct': round(length_error / gt_length * 100, 2) if gt_length and gt_length != 0 else None,
            'width_error_pct': round(width_error / gt_width * 100, 2) if gt_width and gt_width != 0 else None,
            'length_error_corrected_pct': round(length_error_corrected / gt_length * 100, 2) if length_error_corrected is not None and gt_length and gt_length != 0 else None,
            'width_error_corrected_pct': round(width_error_corrected / gt_width * 100, 2) if width_error_corrected is not None and gt_width and gt_width != 0 else None,
            'has_gt': has_gt
        }
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary statistics for samples with GT
    df_with_gt = df[df['has_gt'] == True]
    if len(df_with_gt) > 0:
        print("\n" + "="*80)
        print("SIZE EVALUATION SUMMARY (samples with GT)")
        print("="*80)
        print(f"Total samples evaluated: {len(df_with_gt)}")
        
        if correction_model:
            print(f"\n[RAW] Length Metrics (before correction):")
        else:
            print(f"\nLength Metrics:")
        print(f"  Mean Error:     {df_with_gt['length_error_mm'].mean():7.2f} mm")
        print(f"  Mean Abs Error: {df_with_gt['length_error_mm'].abs().mean():7.2f} mm")
        print(f"  Std Dev:        {df_with_gt['length_error_mm'].std():7.2f} mm")
        print(f"  RMSE:           {np.sqrt((df_with_gt['length_error_mm']**2).mean()):7.2f} mm")
        print(f"  Mean Error %:   {df_with_gt['length_error_pct'].mean():7.2f} %")
        
        if correction_model:
            print(f"\n[RAW] Width Metrics (before correction):")
        else:
            print(f"\nWidth Metrics:")
        print(f"  Mean Error:     {df_with_gt['width_error_mm'].mean():7.2f} mm")
        print(f"  Mean Abs Error: {df_with_gt['width_error_mm'].abs().mean():7.2f} mm")
        print(f"  Std Dev:        {df_with_gt['width_error_mm'].std():7.2f} mm")
        print(f"  RMSE:           {np.sqrt((df_with_gt['width_error_mm']**2).mean()):7.2f} mm")
        print(f"  Mean Error %:   {df_with_gt['width_error_pct'].mean():7.2f} %")
        
        if correction_model:
            print(f"\n[CORRECTED] Length Metrics (after regression correction):")
            print(f"  Mean Error:     {df_with_gt['length_error_corrected_mm'].mean():7.2f} mm")
            print(f"  Mean Abs Error: {df_with_gt['length_error_corrected_mm'].abs().mean():7.2f} mm")
            print(f"  Std Dev:        {df_with_gt['length_error_corrected_mm'].std():7.2f} mm")
            print(f"  RMSE:           {np.sqrt((df_with_gt['length_error_corrected_mm']**2).mean()):7.2f} mm")
            print(f"  Mean Error %:   {df_with_gt['length_error_corrected_pct'].mean():7.2f} %")
            
            print(f"\n[CORRECTED] Width Metrics (after regression correction):")
            print(f"  Mean Error:     {df_with_gt['width_error_corrected_mm'].mean():7.2f} mm")
            print(f"  Mean Abs Error: {df_with_gt['width_error_corrected_mm'].abs().mean():7.2f} mm")
            print(f"  Std Dev:        {df_with_gt['width_error_corrected_mm'].std():7.2f} mm")
            print(f"  RMSE:           {np.sqrt((df_with_gt['width_error_corrected_mm']**2).mean()):7.2f} mm")
            print(f"  Mean Error %:   {df_with_gt['width_error_corrected_pct'].mean():7.2f} %")
        
        print("="*80)
    
    # Print info about samples without GT
    df_no_gt = df[df['has_gt'] == False]
    if len(df_no_gt) > 0:
        print(f"\nSamples without GT: {len(df_no_gt)}")
        print(f"  These are likely historical samples - predictions available but no evaluation metrics")
    
    # Save results
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Save summary statistics
    if len(df_with_gt) > 0:
        summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("SIZE EVALUATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total samples evaluated (with GT): {len(df_with_gt)}\n")
            f.write(f"Total samples (no GT): {len(df_no_gt)}\n\n")
            
            f.write("Length Metrics:\n")
            f.write(f"  Mean Error:     {df_with_gt['length_error_mm'].mean():7.2f} mm\n")
            f.write(f"  Mean Abs Error: {df_with_gt['length_error_mm'].abs().mean():7.2f} mm\n")
            f.write(f"  Std Dev:        {df_with_gt['length_error_mm'].std():7.2f} mm\n")
            f.write(f"  RMSE:           {np.sqrt((df_with_gt['length_error_mm']**2).mean()):7.2f} mm\n")
            f.write(f"  Mean Error %:   {df_with_gt['length_error_pct'].mean():7.2f} %\n\n")
            
            f.write("Width Metrics:\n")
            f.write(f"  Mean Error:     {df_with_gt['width_error_mm'].mean():7.2f} mm\n")
            f.write(f"  Mean Abs Error: {df_with_gt['width_error_mm'].abs().mean():7.2f} mm\n")
            f.write(f"  Std Dev:        {df_with_gt['width_error_mm'].std():7.2f} mm\n")
            f.write(f"  RMSE:           {np.sqrt((df_with_gt['width_error_mm']**2).mean()):7.2f} mm\n")
            f.write(f"  Mean Error %:   {df_with_gt['width_error_pct'].mean():7.2f} %\n")
        
        print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate length and width metrics for fruit meshes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate size metrics for a test run
  python3 evaluate_size_metrics.py \\
      --mesh_dir ./experiments/kmelon_latent32/20251023_120000_test/inf/meshes \\
      --gt_csv ./data/3DKmelonDemo/size_data.csv \\
      --output ./experiments/kmelon_latent32/20251023_120000_test/size_evaluation.csv

  # With regression correction (reduces systematic bias)
  python3 evaluate_size_metrics.py \\
      --mesh_dir ./experiments/kmelon/inf/meshes \\
      --gt_csv ./data/3DKmelonDemo/size_data.csv \\
      --output ./experiments/kmelon/size_evaluation.csv \\
      --correction_csv ./comparison_hand_vs_mesh.csv

  # Save visualizations with PCA axes overlayed
  python3 evaluate_size_metrics.py \\
      --mesh_dir ./experiments/kmelon/inf/meshes \\
      --gt_csv ./data/3DKmelonDemo/size_data.csv \\
      --output ./experiments/kmelon/size_evaluation.csv \\
      --visualize
        """
    )
    
    parser.add_argument(
        '--mesh_dir',
        type=str,
        required=True,
        help='Directory containing predicted mesh files'
    )
    
    parser.add_argument(
        '--gt_csv',
        type=str,
        default='./data/3DKmelonDemo/size_data.csv',
        help='Path to ground truth CSV with size data (default: ./data/3DKmelonDemo/size_data.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for evaluation CSV'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='ply',
        choices=['ply', 'obj', 'stl'],
        help='Mesh file format (default: ply)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save meshes with PCA axes overlayed for visualization'
    )
    
    parser.add_argument(
        '--correction_csv',
        type=str,
        default=None,
        help='Path to hand vs mesh comparison CSV for regression correction (optional)'
    )
    
    args = parser.parse_args()
    
    evaluate_size_metrics(
        mesh_dir=args.mesh_dir,
        gt_csv=args.gt_csv,
        output_csv=args.output,
        mesh_format=args.format,
        save_visualizations=args.visualize,
        correction_csv=args.correction_csv
    )


if __name__ == '__main__':
    main()

