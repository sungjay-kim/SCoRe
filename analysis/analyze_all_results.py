#!/usr/bin/env python3
"""
Comprehensive analysis of shape completion and size evaluation results.

This script analyzes:
1. Shape completion performance (F1 score, RMSE) across encoders and training strategies
2. Size evaluation performance across configurations
3. Overfitting detection by analyzing errors vs. date (fruit growth stage)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def parse_run_directory(run_path):
    """
    Extract configuration info from run directory name.
    
    Example: 20251024_125916_train_orbbec_from_scratch
    Returns: {
        'timestamp': '20251024_125916',
        'encoder': 'minkowski',
        'training': 'from_scratch',
        'config_name': 'train_orbbec_from_scratch'
    }
    """
    parts = run_path.name.split('_', 2)
    if len(parts) < 3:
        return None
    
    timestamp = f"{parts[0]}_{parts[1]}"
    config_name = parts[2]
    
    # Determine encoder type
    if 'pce' in config_name:
        encoder = 'pce'
    elif 'super3d' in config_name:
        encoder = 'super3d'
    else:
        encoder = 'minkowski'
    
    # Determine training strategy
    if 'finetune' in config_name:
        training = 'finetune'
    elif 'from_scratch' in config_name or 'train_' in config_name:
        training = 'from_scratch'
    else:
        training = 'unknown'
    
    return {
        'timestamp': timestamp,
        'encoder': encoder,
        'training': training,
        'config_name': config_name
    }


def load_all_results(base_dir, latent_sizes=[16, 64, 128]):
    """
    Load all shape_completion_results.csv and size_evaluation.csv files.
    """
    base_path = Path(base_dir)
    
    shape_results = []
    size_results = []
    
    for latent_size in latent_sizes:
        latent_dir = base_path / f"kmelon_latent{latent_size}"
        
        if not latent_dir.exists():
            print(f"[Warning] Directory not found: {latent_dir}")
            continue
        
        # Find all run directories
        for run_dir in latent_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            # Parse configuration
            config = parse_run_directory(run_dir)
            if config is None:
                continue
            
            # Load shape completion results
            shape_csv = run_dir / "shape_completion_results.csv"
            if shape_csv.exists():
                try:
                    df_shape = pd.read_csv(shape_csv)
                    df_shape['latent_size'] = latent_size
                    df_shape['encoder'] = config['encoder']
                    df_shape['training'] = config['training']
                    df_shape['config_name'] = config['config_name']
                    df_shape['run_dir'] = run_dir.name
                    shape_results.append(df_shape)
                except Exception as e:
                    print(f"[Error] Failed to load {shape_csv}: {e}")
            
            # Load size evaluation results
            size_csv = run_dir / "size_evaluation.csv"
            if size_csv.exists():
                try:
                    df_size = pd.read_csv(size_csv)
                    df_size['latent_size'] = latent_size
                    df_size['encoder'] = config['encoder']
                    df_size['training'] = config['training']
                    df_size['config_name'] = config['config_name']
                    df_size['run_dir'] = run_dir.name
                    size_results.append(df_size)
                except Exception as e:
                    print(f"[Error] Failed to load {size_csv}: {e}")
    
    df_shape_all = pd.concat(shape_results, ignore_index=True) if shape_results else pd.DataFrame()
    df_size_all = pd.concat(size_results, ignore_index=True) if size_results else pd.DataFrame()
    
    print(f"\n[Data] Loaded {len(shape_results)} shape completion result files")
    print(f"[Data] Loaded {len(size_results)} size evaluation result files")
    
    return df_shape_all, df_size_all


def analyze_shape_completion(df_shape, output_dir):
    """
    Analyze shape completion results: F1 score, Chamfer distance, and Volume RMSE comparison.
    Only analyzes the LAST date for each sample (filters out earlier time-series data).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SHAPE COMPLETION ANALYSIS")
    print("="*80)
    
    # Filter to only valid entries (with ground truth)
    # Check if gt_volume_ml exists and is not null
    if 'gt_volume_ml' in df_shape.columns:
        df_valid = df_shape[df_shape['gt_volume_ml'].notna()].copy()
    else:
        df_valid = df_shape.copy()
    
    # Extract date from frame_id (format: SampleID_YYYYMMDD)
    if 'frame_id' in df_valid.columns:
        df_valid['date'] = df_valid['frame_id'].str.split('_').str[-1]
        df_valid['date'] = pd.to_datetime(df_valid['date'], format='%Y%m%d', errors='coerce')
    
    # Identify fruit/sample ID column
    if 'potato_id' in df_valid.columns:
        fruit_id_col = 'potato_id'
    elif 'kmelon_id' in df_valid.columns and df_valid['kmelon_id'].notna().any():
        fruit_id_col = 'kmelon_id'
    else:
        # Extract from frame_id if no dedicated ID column
        df_valid['fruit_id_extracted'] = df_valid['frame_id'].str.split('_').str[0]
        fruit_id_col = 'fruit_id_extracted'
    
    # Keep only the LAST date for each fruit (per configuration)
    print(f"\nTotal samples before filtering: {len(df_valid)}")
    
    df_latest = []
    for (latent_size, encoder, training, run_dir), group in df_valid.groupby(['latent_size', 'encoder', 'training', 'run_dir']):
        if 'date' in group.columns and group['date'].notna().any():
            # For each fruit, keep only the latest date
            latest_per_fruit = group.sort_values('date').groupby(fruit_id_col).tail(1)
            df_latest.append(latest_per_fruit)
        else:
            # No date info, keep all (shouldn't happen with proper data)
            df_latest.append(group)
    
    df_valid = pd.concat(df_latest, ignore_index=True) if df_latest else df_valid
    
    print(f"Samples after keeping only LAST date per fruit: {len(df_valid)}")
    print("[Info] Analyzing only the latest measurement for each fruit")
    
    # Convert chamfer_distance from meters to millimeters
    if 'chamfer_distance' in df_valid.columns:
        df_valid['chamfer_distance_mm'] = df_valid['chamfer_distance'] * 1000
    
    # Calculate volume error if both pred and gt volumes exist
    # Use mesh_volume_ml as the predicted volume
    if 'mesh_volume_ml' in df_valid.columns and 'gt_volume_ml' in df_valid.columns:
        df_valid['volume_error_ml'] = df_valid['mesh_volume_ml'] - df_valid['gt_volume_ml']
        df_valid['volume_error_percent'] = (df_valid['volume_error_ml'] / df_valid['gt_volume_ml']) * 100
    
    print(f"\nTotal samples with GT (latest dates only): {len(df_valid)}")
    
    # Group by configuration
    grouped = df_valid.groupby(['latent_size', 'encoder', 'training'])
    
    # Compute mean metrics
    agg_dict = {
        'f1': ['mean', 'std', 'count'],
        'chamfer_distance_mm': ['mean', 'std']
    }
    
    # Add precision and recall if available
    if 'precision' in df_valid.columns:
        agg_dict['precision'] = ['mean', 'std']
    if 'recall' in df_valid.columns:
        agg_dict['recall'] = ['mean', 'std']
    
    if 'volume_error_ml' in df_valid.columns:
        agg_dict['volume_error_ml'] = ['mean', 'std']
        agg_dict['volume_error_percent'] = ['mean', 'std']
    
    metrics = grouped.agg(agg_dict).round(4)
    
    print("\n" + "-"*80)
    print("MEAN METRICS BY CONFIGURATION")
    print("-"*80)
    print(metrics)
    
    # Create summary table for easier comparison
    summary_data = []
    for (latent_size, encoder, training), group in grouped:
        row = {
            'latent_size': latent_size,
            'encoder': encoder,
            'training': training,
            'f1_mean': group['f1'].mean(),
            'f1_std': group['f1'].std(),
            'chamfer_distance_mm_mean': group['chamfer_distance_mm'].mean(),
            'chamfer_distance_mm_std': group['chamfer_distance_mm'].std(),
            'n_samples': len(group)
        }
        
        # Add precision and recall if available
        if 'precision' in group.columns:
            row['precision_mean'] = group['precision'].mean()
            row['precision_std'] = group['precision'].std()
        if 'recall' in group.columns:
            row['recall_mean'] = group['recall'].mean()
            row['recall_std'] = group['recall'].std()
        
        # Add volume metrics if available
        if 'volume_error_ml' in group.columns:
            row['volume_mae_ml'] = group['volume_error_ml'].abs().mean()
            row['volume_rmse_ml'] = np.sqrt((group['volume_error_ml']**2).mean())
            row['volume_mape_percent'] = group['volume_error_percent'].abs().mean()
        
        # Add inference time if available (convert from ms to s)
        if 'total_time_ms' in group.columns:
            row['inference_time_mean_s'] = group['total_time_ms'].mean() / 1000.0
            row['inference_time_std_s'] = group['total_time_ms'].std() / 1000.0
        
        # Add timing breakdown if available
        if 'encoder_time_ms' in group.columns:
            row['encoder_time_mean_ms'] = group['encoder_time_ms'].mean()
            row['encoder_time_std_ms'] = group['encoder_time_ms'].std()
        
        if 'decoder_time_ms' in group.columns:
            row['decoder_time_mean_ms'] = group['decoder_time_ms'].mean()
            row['decoder_time_std_ms'] = group['decoder_time_ms'].std()
        
        if 'mesh_extraction_time_ms' in group.columns:
            row['mesh_extraction_time_mean_ms'] = group['mesh_extraction_time_ms'].mean()
            row['mesh_extraction_time_std_ms'] = group['mesh_extraction_time_ms'].std()
        
        summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values(['latent_size', 'encoder', 'training'])
    
    # Save summary
    summary_path = output_dir / "shape_completion_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[Saved] Summary table: {summary_path}")
    
    # Analysis 1: Encoder Comparison
    print("\n" + "-"*80)
    print("ANALYSIS 1: ENCODER COMPARISON")
    print("-"*80)
    
    encoder_comparison = df_summary.groupby('encoder').agg({
        'f1_mean': 'mean',
        'chamfer_distance_mm_mean': 'mean',
        'n_samples': 'sum'
    }).round(4)
    
    print("\nOverall Performance by Encoder:")
    print(encoder_comparison)
    
    best_f1_encoder = encoder_comparison['f1_mean'].idxmax()
    best_chamfer_encoder = encoder_comparison['chamfer_distance_mm_mean'].idxmin()
    
    print(f"\n✓ Best F1 Score: {best_f1_encoder} ({encoder_comparison.loc[best_f1_encoder, 'f1_mean']:.4f})")
    print(f"✓ Best Chamfer Distance: {best_chamfer_encoder} ({encoder_comparison.loc[best_chamfer_encoder, 'chamfer_distance_mm_mean']:.4f} mm)")
    
    # Volume metrics if available
    if 'volume_rmse_ml' in df_summary.columns:
        volume_comparison = df_summary.groupby('encoder').agg({
            'volume_mae_ml': 'mean',
            'volume_rmse_ml': 'mean',
            'volume_mape_percent': 'mean'
        }).round(4)
        
        best_volume_encoder = volume_comparison['volume_rmse_ml'].idxmin()
        print(f"✓ Best Volume RMSE: {best_volume_encoder} ({volume_comparison.loc[best_volume_encoder, 'volume_rmse_ml']:.4f} ml)")
    
    # Analysis 2: Training Strategy Comparison (Finetune vs From Scratch)
    print("\n" + "-"*80)
    print("ANALYSIS 2: TRAINING STRATEGY COMPARISON")
    print("-"*80)
    
    for encoder in df_summary['encoder'].unique():
        df_encoder = df_summary[df_summary['encoder'] == encoder]
        
        if 'finetune' in df_encoder['training'].values and 'from_scratch' in df_encoder['training'].values:
            finetune = df_encoder[df_encoder['training'] == 'finetune']
            from_scratch = df_encoder[df_encoder['training'] == 'from_scratch']
            
            print(f"\n{encoder.upper()} Encoder:")
            print(f"  Finetune      - F1: {finetune['f1_mean'].mean():.4f}, Chamfer: {finetune['chamfer_distance_mm_mean'].mean():.4f} mm")
            print(f"  From Scratch  - F1: {from_scratch['f1_mean'].mean():.4f}, Chamfer: {from_scratch['chamfer_distance_mm_mean'].mean():.4f} mm")
            
            f1_improvement = (finetune['f1_mean'].mean() - from_scratch['f1_mean'].mean()) / from_scratch['f1_mean'].mean() * 100
            chamfer_improvement = (from_scratch['chamfer_distance_mm_mean'].mean() - finetune['chamfer_distance_mm_mean'].mean()) / from_scratch['chamfer_distance_mm_mean'].mean() * 100
            
            print(f"  Improvement   - F1: {f1_improvement:+.2f}%, Chamfer: {chamfer_improvement:+.2f}%")
    
    # Visualization 1: F1 Score and Chamfer Distance Comparison
    n_plots = 3 if 'volume_rmse_ml' in df_summary.columns else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    # F1 Score
    df_plot = df_summary.copy()
    df_plot['config'] = df_plot['encoder'] + '_' + df_plot['training']
    
    for latent in sorted(df_plot['latent_size'].unique()):
        df_latent = df_plot[df_plot['latent_size'] == latent]
        x = range(len(df_latent))
        axes[0].bar([i + latent/200 for i in x], df_latent['f1_mean'], 
                   width=0.2, label=f'Latent {latent}', alpha=0.7)
    
    axes[0].set_xlabel('Configuration')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score Comparison by Configuration')
    axes[0].set_xticks(range(len(df_plot['config'].unique())))
    axes[0].set_xticklabels(df_plot['config'].unique(), rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Chamfer Distance
    for latent in sorted(df_plot['latent_size'].unique()):
        df_latent = df_plot[df_plot['latent_size'] == latent]
        x = range(len(df_latent))
        axes[1].bar([i + latent/200 for i in x], df_latent['chamfer_distance_mm_mean'], 
                   width=0.2, label=f'Latent {latent}', alpha=0.7)
    
    axes[1].set_xlabel('Configuration')
    axes[1].set_ylabel('Chamfer Distance (mm)')
    axes[1].set_title('Chamfer Distance Comparison by Configuration')
    axes[1].set_xticks(range(len(df_plot['config'].unique())))
    axes[1].set_xticklabels(df_plot['config'].unique(), rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Volume RMSE (if available)
    if 'volume_rmse_ml' in df_summary.columns:
        for latent in sorted(df_plot['latent_size'].unique()):
            df_latent = df_plot[df_plot['latent_size'] == latent]
            x = range(len(df_latent))
            axes[2].bar([i + latent/200 for i in x], df_latent['volume_rmse_ml'], 
                       width=0.2, label=f'Latent {latent}', alpha=0.7)
        
        axes[2].set_xlabel('Configuration')
        axes[2].set_ylabel('Volume RMSE (ml)')
        axes[2].set_title('Volume RMSE Comparison by Configuration')
        axes[2].set_xticks(range(len(df_plot['config'].unique())))
        axes[2].set_xticklabels(df_plot['config'].unique(), rotation=45, ha='right')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "shape_completion_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[Saved] Plot: {plot_path}")
    plt.close()
    
    return df_summary


def analyze_size_evaluation(df_size, output_dir):
    """
    Analyze size evaluation results: length/width errors and overfitting detection.
    Analyzes ALL dates (including time-series data) to detect overfitting patterns.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SIZE EVALUATION ANALYSIS")
    print("="*80)
    
    # Filter to only valid entries (with has_gt=True)
    if 'has_gt' in df_size.columns:
        df_valid = df_size[df_size['has_gt'] == True].copy()
    else:
        df_valid = df_size.copy()
    
    # Convert date to datetime
    df_valid['date'] = pd.to_datetime(df_valid['date'], format='%Y%m%d')
    
    print(f"\nTotal samples with GT: {len(df_valid)}")
    print(f"Date range: {df_valid['date'].min()} to {df_valid['date'].max()}")
    print("[Info] Analyzing ALL dates (including time-series) for overfitting detection")
    
    # Use corrected metrics if available
    if 'length_error_corrected_mm' in df_valid.columns and df_valid['length_error_corrected_mm'].notna().any():
        length_error_col = 'length_error_corrected_mm'
        width_error_col = 'width_error_corrected_mm'
        print("\n[Info] Using corrected errors for analysis")
    else:
        length_error_col = 'length_error_mm'
        width_error_col = 'width_error_mm'
        print("\n[Info] Using raw errors for analysis (no correction applied)")
    
    # Group by configuration
    grouped = df_valid.groupby(['latent_size', 'encoder', 'training'])
    
    # Compute mean metrics
    summary_data = []
    for (latent_size, encoder, training), group in grouped:
        summary_data.append({
            'latent_size': latent_size,
            'encoder': encoder,
            'training': training,
            'length_mae': group[length_error_col].abs().mean(),
            'length_rmse': np.sqrt((group[length_error_col]**2).mean()),
            'width_mae': group[width_error_col].abs().mean(),
            'width_rmse': np.sqrt((group[width_error_col]**2).mean()),
            'n_samples': len(group)
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values(['latent_size', 'encoder', 'training'])
    
    print("\n" + "-"*80)
    print("MEAN SIZE METRICS BY CONFIGURATION")
    print("-"*80)
    print(df_summary.round(2))
    
    # Save summary
    summary_path = output_dir / "size_evaluation_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[Saved] Summary table: {summary_path}")
    
    # Analysis 3: Overfitting Detection (Errors vs Date)
    print("\n" + "-"*80)
    print("ANALYSIS 3: OVERFITTING DETECTION (Errors vs Growth Stage)")
    print("-"*80)
    print("\nChecking if earlier dates (smaller fruits) show larger errors...")
    print("(This would indicate overfitting to larger fruits)")
    
    # Create date bins (early, mid, late season)
    date_min = df_valid['date'].min()
    date_max = df_valid['date'].max()
    date_range = (date_max - date_min).days
    
    # Define early, mid, late periods
    df_valid['period'] = pd.cut(df_valid['date'], 
                                 bins=3, 
                                 labels=['Early (Small)', 'Mid', 'Late (Large)'])
    
    print(f"\nDate periods:")
    print(f"  Early (Small): {date_min.strftime('%Y-%m-%d')} - {(date_min + pd.Timedelta(days=date_range//3)).strftime('%Y-%m-%d')}")
    print(f"  Mid:           {(date_min + pd.Timedelta(days=date_range//3)).strftime('%Y-%m-%d')} - {(date_min + pd.Timedelta(days=2*date_range//3)).strftime('%Y-%m-%d')}")
    print(f"  Late (Large):  {(date_min + pd.Timedelta(days=2*date_range//3)).strftime('%Y-%m-%d')} - {date_max.strftime('%Y-%m-%d')}")
    
    # Analyze errors by period
    overfitting_analysis = []
    
    for (latent_size, encoder, training), group in df_valid.groupby(['latent_size', 'encoder', 'training']):
        period_stats = group.groupby('period').agg({
            length_error_col: lambda x: x.abs().mean(),
            width_error_col: lambda x: x.abs().mean(),
            'sample': 'count'
        }).round(2)
        
        if len(period_stats) >= 2:
            # Check if early period has larger errors
            if 'Early (Small)' in period_stats.index and 'Late (Large)' in period_stats.index:
                early_length_mae = period_stats.loc['Early (Small)', length_error_col]
                late_length_mae = period_stats.loc['Late (Large)', length_error_col]
                early_width_mae = period_stats.loc['Early (Small)', width_error_col]
                late_width_mae = period_stats.loc['Late (Large)', width_error_col]
                
                overfitting_score = ((early_length_mae - late_length_mae) + (early_width_mae - late_width_mae)) / 2
                
                overfitting_analysis.append({
                    'latent_size': latent_size,
                    'encoder': encoder,
                    'training': training,
                    'early_length_mae': early_length_mae,
                    'late_length_mae': late_length_mae,
                    'early_width_mae': early_width_mae,
                    'late_width_mae': late_width_mae,
                    'overfitting_score': overfitting_score,
                    'interpretation': 'Overfitting' if overfitting_score > 5 else 'Good Generalization'
                })
    
    df_overfitting = pd.DataFrame(overfitting_analysis)
    df_overfitting = df_overfitting.sort_values('overfitting_score', ascending=False)
    
    print("\n" + "-"*80)
    print("OVERFITTING ANALYSIS RESULTS")
    print("-"*80)
    print("\nOverfitting Score = (Early MAE - Late MAE) averaged for length & width")
    print("Positive score = worse on small fruits (overfitting to large fruits)")
    print("Negative score = worse on large fruits (underfitting)")
    print()
    print(df_overfitting.round(2))
    
    # Save overfitting analysis
    overfitting_path = output_dir / "overfitting_analysis.csv"
    df_overfitting.to_csv(overfitting_path, index=False)
    print(f"\n[Saved] Overfitting analysis: {overfitting_path}")
    
    # Identify models with severe overfitting
    severe_overfitting = df_overfitting[df_overfitting['overfitting_score'] > 5]
    if len(severe_overfitting) > 0:
        print("\n⚠️  MODELS WITH SEVERE OVERFITTING DETECTED:")
        for _, row in severe_overfitting.iterrows():
            print(f"   - Latent {row['latent_size']}, {row['encoder']}, {row['training']}")
            print(f"     Score: {row['overfitting_score']:.2f} mm (Early: {row['early_length_mae']:.2f}/{row['early_width_mae']:.2f}, Late: {row['late_length_mae']:.2f}/{row['late_width_mae']:.2f})")
    else:
        print("\n✓ No severe overfitting detected in any configuration")
    
    # Visualization 1: Raw (Before Correction) Errors by Date
    if 'length_error_mm' in df_valid.columns and 'width_error_mm' in df_valid.columns:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Size Errors vs Date - BEFORE CORRECTION (Raw)', fontsize=14, fontweight='bold')
        
        encoders = ['minkowski', 'pce', 'super3d']
        
        for i, encoder in enumerate(encoders):
            df_encoder = df_valid[df_valid['encoder'] == encoder]
            
            if len(df_encoder) == 0:
                continue
            
            # Length errors (raw)
            for training in df_encoder['training'].unique():
                df_subset = df_encoder[df_encoder['training'] == training]
                
                # Group by date and compute mean absolute error
                date_errors = df_subset.groupby('date').agg({
                    'length_error_mm': lambda x: x.abs().mean(),
                    'width_error_mm': lambda x: x.abs().mean()
                }).reset_index()
                
                axes[0, i].plot(date_errors['date'], date_errors['length_error_mm'], 
                              marker='o', label=training, alpha=0.7, linewidth=2)
            
            axes[0, i].set_xlabel('Date')
            axes[0, i].set_ylabel('Length MAE (mm)')
            axes[0, i].set_title(f'{encoder.upper()} - Length Errors (Raw)')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].tick_params(axis='x', rotation=45)
            
            # Width errors (raw)
            for training in df_encoder['training'].unique():
                df_subset = df_encoder[df_encoder['training'] == training]
                
                date_errors = df_subset.groupby('date').agg({
                    'length_error_mm': lambda x: x.abs().mean(),
                    'width_error_mm': lambda x: x.abs().mean()
                }).reset_index()
                
                axes[1, i].plot(date_errors['date'], date_errors['width_error_mm'], 
                              marker='o', label=training, alpha=0.7, linewidth=2)
            
            axes[1, i].set_xlabel('Date')
            axes[1, i].set_ylabel('Width MAE (mm)')
            axes[1, i].set_title(f'{encoder.upper()} - Width Errors (Raw)')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = output_dir / "size_errors_by_date_raw.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n[Saved] Plot (raw errors): {plot_path}")
        plt.close()
    
    # Visualization 2: Corrected Errors by Date (or raw if no correction available)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    correction_status = "AFTER CORRECTION" if length_error_col == 'length_error_corrected_mm' else "RAW (no correction available)"
    fig.suptitle(f'Size Errors vs Date - {correction_status}', fontsize=14, fontweight='bold')
    
    encoders = ['minkowski', 'pce', 'super3d']
    
    for i, encoder in enumerate(encoders):
        df_encoder = df_valid[df_valid['encoder'] == encoder]
        
        if len(df_encoder) == 0:
            continue
        
        # Length errors
        for training in df_encoder['training'].unique():
            df_subset = df_encoder[df_encoder['training'] == training]
            
            # Group by date and compute mean absolute error
            date_errors = df_subset.groupby('date').agg({
                length_error_col: lambda x: x.abs().mean(),
                width_error_col: lambda x: x.abs().mean()
            }).reset_index()
            
            axes[0, i].plot(date_errors['date'], date_errors[length_error_col], 
                          marker='o', label=training, alpha=0.7, linewidth=2)
        
        axes[0, i].set_xlabel('Date')
        axes[0, i].set_ylabel('Length MAE (mm)')
        title_suffix = "(Corrected)" if length_error_col == 'length_error_corrected_mm' else "(Raw)"
        axes[0, i].set_title(f'{encoder.upper()} - Length Errors {title_suffix}')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].tick_params(axis='x', rotation=45)
        
        # Width errors
        for training in df_encoder['training'].unique():
            df_subset = df_encoder[df_encoder['training'] == training]
            
            date_errors = df_subset.groupby('date').agg({
                length_error_col: lambda x: x.abs().mean(),
                width_error_col: lambda x: x.abs().mean()
            }).reset_index()
            
            axes[1, i].plot(date_errors['date'], date_errors[width_error_col], 
                          marker='o', label=training, alpha=0.7, linewidth=2)
        
        axes[1, i].set_xlabel('Date')
        axes[1, i].set_ylabel('Width MAE (mm)')
        axes[1, i].set_title(f'{encoder.upper()} - Width Errors {title_suffix}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = output_dir / "size_errors_by_date_corrected.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[Saved] Plot (corrected errors): {plot_path}")
    plt.close()
    
    return df_summary, df_overfitting


def analyze_inference_time(df_shape, output_dir):
    """
    Analyze inference time performance across configurations.
    Breaks down timing into encoder, decoder, and mesh extraction components.
    Uses timing columns from the data (converts ms to seconds for reporting).
    """
    output_dir = Path(output_dir)
    
    print("\n" + "="*80)
    print("INFERENCE TIME ANALYSIS")
    print("="*80)
    
    if 'total_time_ms' not in df_shape.columns:
        print("\n[Warning] No total_time_ms column found in data")
        return None
    
    # Filter valid entries and convert ms to seconds
    df_valid = df_shape[df_shape['total_time_ms'].notna()].copy()
    df_valid['inference_time_s'] = df_valid['total_time_ms'] / 1000.0
    
    # Convert component times if available
    if 'encoder_time_ms' in df_valid.columns:
        df_valid['encoder_time_s'] = df_valid['encoder_time_ms'] / 1000.0
    if 'decoder_time_ms' in df_valid.columns:
        df_valid['decoder_time_s'] = df_valid['decoder_time_ms'] / 1000.0
    if 'mesh_extraction_time_ms' in df_valid.columns:
        df_valid['mesh_extraction_time_s'] = df_valid['mesh_extraction_time_ms'] / 1000.0
    
    if len(df_valid) == 0:
        print("\n[Warning] No inference time data available")
        return None
    
    print(f"\nTotal samples with inference time: {len(df_valid)}")
    
    # Group by configuration
    grouped = df_valid.groupby(['latent_size', 'encoder', 'training'])
    
    # Compute statistics
    summary_data = []
    for (latent_size, encoder, training), group in grouped:
        row = {
            'latent_size': latent_size,
            'encoder': encoder,
            'training': training,
            'total_time_mean_ms': group['total_time_ms'].mean(),
            'total_time_std_ms': group['total_time_ms'].std(),
            'total_time_median_ms': group['total_time_ms'].median(),
            'n_samples': len(group)
        }
        
        # Add component timing breakdown if available
        if 'encoder_time_ms' in group.columns:
            row['encoder_time_mean_ms'] = group['encoder_time_ms'].mean()
            row['encoder_time_std_ms'] = group['encoder_time_ms'].std()
        
        if 'decoder_time_ms' in group.columns:
            row['decoder_time_mean_ms'] = group['decoder_time_ms'].mean()
            row['decoder_time_std_ms'] = group['decoder_time_ms'].std()
        
        if 'mesh_extraction_time_ms' in group.columns:
            row['mesh_extraction_time_mean_ms'] = group['mesh_extraction_time_ms'].mean()
            row['mesh_extraction_time_std_ms'] = group['mesh_extraction_time_ms'].std()
        
        # Calculate percentage breakdown if components available
        if all(k in row for k in ['encoder_time_mean_ms', 'decoder_time_mean_ms', 'mesh_extraction_time_mean_ms']):
            total = row['total_time_mean_ms']
            if total > 0:
                row['encoder_time_percent'] = (row['encoder_time_mean_ms'] / total) * 100
                row['decoder_time_percent'] = (row['decoder_time_mean_ms'] / total) * 100
                row['mesh_extraction_time_percent'] = (row['mesh_extraction_time_mean_ms'] / total) * 100
        
        summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values(['latent_size', 'encoder', 'training'])
    
    print("\n" + "-"*80)
    print("INFERENCE TIME BY CONFIGURATION (milliseconds)")
    print("-"*80)
    print(df_summary.round(2))
    
    # Save summary
    summary_path = output_dir / "inference_time_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[Saved] Inference time summary: {summary_path}")
    
    # Analysis: Encoder comparison
    print("\n" + "-"*80)
    print("INFERENCE TIME BY ENCODER TYPE")
    print("-"*80)
    
    agg_dict = {
        'total_time_mean_ms': 'mean',
        'n_samples': 'sum'
    }
    
    # Add component times if available
    if 'encoder_time_mean_ms' in df_summary.columns:
        agg_dict['encoder_time_mean_ms'] = 'mean'
    if 'decoder_time_mean_ms' in df_summary.columns:
        agg_dict['decoder_time_mean_ms'] = 'mean'
    if 'mesh_extraction_time_mean_ms' in df_summary.columns:
        agg_dict['mesh_extraction_time_mean_ms'] = 'mean'
    
    encoder_comparison = df_summary.groupby('encoder').agg(agg_dict).round(2)
    
    print(encoder_comparison)
    
    fastest_encoder = encoder_comparison['total_time_mean_ms'].idxmin()
    print(f"\n✓ Fastest encoder: {fastest_encoder} ({encoder_comparison.loc[fastest_encoder, 'total_time_mean_ms']:.2f} ms)")
    
    # Timing breakdown analysis
    if 'encoder_time_mean_ms' in df_summary.columns:
        print("\n" + "-"*80)
        print("TIMING BREAKDOWN (Mean across all configurations)")
        print("-"*80)
        
        for encoder in df_summary['encoder'].unique():
            df_enc = df_summary[df_summary['encoder'] == encoder]
            enc_time = df_enc['encoder_time_mean_ms'].mean()
            dec_time = df_enc['decoder_time_mean_ms'].mean() if 'decoder_time_mean_ms' in df_enc.columns else 0
            mesh_time = df_enc['mesh_extraction_time_mean_ms'].mean() if 'mesh_extraction_time_mean_ms' in df_enc.columns else 0
            total = df_enc['total_time_mean_ms'].mean()
            
            print(f"\n{encoder.upper()}:")
            print(f"  Encoder:         {enc_time:7.2f} ms ({enc_time/total*100:5.1f}%)")
            print(f"  Decoder:         {dec_time:7.2f} ms ({dec_time/total*100:5.1f}%)")
            print(f"  Mesh Extraction: {mesh_time:7.2f} ms ({mesh_time/total*100:5.1f}%)")
            print(f"  Total:           {total:7.2f} ms")
    
    # Visualization: Inference time comparison (total time only)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # By configuration
    df_plot = df_summary.copy()
    df_plot['config'] = df_plot['encoder'] + '_' + df_plot['training']
    df_plot['total_time_mean_s'] = df_plot['total_time_mean_ms'] / 1000.0
    
    for latent in sorted(df_plot['latent_size'].unique()):
        df_latent = df_plot[df_plot['latent_size'] == latent]
        x = range(len(df_latent))
        axes[0].bar([i + latent/200 for i in x], df_latent['total_time_mean_s'], 
                   width=0.2, label=f'Latent {latent}', alpha=0.7)
    
    axes[0].set_xlabel('Configuration')
    axes[0].set_ylabel('Total Inference Time (s)')
    axes[0].set_title('Total Inference Time by Configuration')
    axes[0].set_xticks(range(len(df_plot['config'].unique())))
    axes[0].set_xticklabels(df_plot['config'].unique(), rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # By encoder (aggregated)
    encoder_stats = df_summary.groupby('encoder').agg({
        'total_time_mean_ms': 'mean'
    }).reset_index()
    encoder_stats['total_time_mean_s'] = encoder_stats['total_time_mean_ms'] / 1000.0
    
    axes[1].bar(encoder_stats['encoder'], encoder_stats['total_time_mean_s'], alpha=0.7)
    axes[1].set_xlabel('Encoder')
    axes[1].set_ylabel('Mean Total Inference Time (s)')
    axes[1].set_title('Mean Total Inference Time by Encoder')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "inference_time_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[Saved] Plot: {plot_path}")
    plt.close()
    
    return df_summary


def generate_comprehensive_report(df_shape_summary, df_size_summary, df_overfitting, df_inference_summary, output_dir):
    """
    Generate a comprehensive text report summarizing all findings.
    """
    output_dir = Path(output_dir)
    report_path = output_dir / "comprehensive_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("Analysis Date: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("="*80 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*80 + "\n\n")
        
        # Best encoder for shape completion
        encoder_perf = df_shape_summary.groupby('encoder').agg({
            'f1_mean': 'mean',
            'chamfer_distance_mm_mean': 'mean'
        }).round(4)
        best_f1_encoder = encoder_perf['f1_mean'].idxmax()
        best_chamfer_encoder = encoder_perf['chamfer_distance_mm_mean'].idxmin()
        
        f.write(f"1. ENCODER PERFORMANCE\n")
        f.write(f"   Best F1 Score: {best_f1_encoder.upper()} ({encoder_perf.loc[best_f1_encoder, 'f1_mean']:.4f})\n")
        f.write(f"   Best Chamfer Distance: {best_chamfer_encoder.upper()} ({encoder_perf.loc[best_chamfer_encoder, 'chamfer_distance_mm_mean']:.4f} mm)\n")
        
        # Volume RMSE if available
        if 'volume_rmse_ml' in df_shape_summary.columns:
            volume_perf = df_shape_summary.groupby('encoder')['volume_rmse_ml'].mean()
            best_volume_encoder = volume_perf.idxmin()
            f.write(f"   Best Volume RMSE: {best_volume_encoder.upper()} ({volume_perf[best_volume_encoder]:.4f} ml)\n")
        
        f.write("\n")
        
        # Finetune improvement
        f.write(f"2. FINETUNING EFFECTIVENESS\n")
        for encoder in ['minkowski', 'pce']:
            if encoder in df_shape_summary['encoder'].values:
                ft = df_shape_summary[(df_shape_summary['encoder'] == encoder) & 
                                     (df_shape_summary['training'] == 'finetune')]['f1_mean'].mean()
                fs = df_shape_summary[(df_shape_summary['encoder'] == encoder) & 
                                     (df_shape_summary['training'] == 'from_scratch')]['f1_mean'].mean()
                
                if not np.isnan(ft) and not np.isnan(fs):
                    improvement = (ft - fs) / fs * 100
                    f.write(f"   {encoder.upper()}: {improvement:+.2f}% F1 improvement\n")
        
        f.write("\n")
        
        # Inference time
        if df_inference_summary is not None:
            f.write(f"3. INFERENCE TIME\n")
            inference_by_encoder = df_inference_summary.groupby('encoder')['total_time_mean_ms'].mean()
            fastest_encoder = inference_by_encoder.idxmin()
            f.write(f"   Fastest encoder: {fastest_encoder.upper()} ({inference_by_encoder[fastest_encoder]:.2f} ms)\n\n")
            
            # Detailed timing breakdown by encoder
            f.write("   Timing Breakdown by Encoder:\n")
            for encoder in inference_by_encoder.index:
                df_enc = df_inference_summary[df_inference_summary['encoder'] == encoder]
                total = df_enc['total_time_mean_ms'].mean()
                f.write(f"   {encoder.upper()}: {total:.2f} ms total\n")
                
                if 'encoder_time_mean_ms' in df_enc.columns:
                    enc_time = df_enc['encoder_time_mean_ms'].mean()
                    dec_time = df_enc['decoder_time_mean_ms'].mean() if 'decoder_time_mean_ms' in df_enc.columns else 0
                    mesh_time = df_enc['mesh_extraction_time_mean_ms'].mean() if 'mesh_extraction_time_mean_ms' in df_enc.columns else 0
                    
                    f.write(f"      - Encoder:         {enc_time:7.2f} ms ({enc_time/total*100:5.1f}%)\n")
                    f.write(f"      - Decoder:         {dec_time:7.2f} ms ({dec_time/total*100:5.1f}%)\n")
                    f.write(f"      - Mesh Extraction: {mesh_time:7.2f} ms ({mesh_time/total*100:5.1f}%)\n")
            
            f.write("\n")
        
        # Overfitting
        f.write(f"4. OVERFITTING ASSESSMENT\n")
        severe_overfitting = df_overfitting[df_overfitting['overfitting_score'] > 5]
        if len(severe_overfitting) > 0:
            f.write(f"   ⚠️  {len(severe_overfitting)} configuration(s) show severe overfitting\n")
            for _, row in severe_overfitting.iterrows():
                f.write(f"      - Latent {row['latent_size']}, {row['encoder']}, {row['training']}\n")
        else:
            f.write(f"   ✓ No severe overfitting detected\n")
        
        f.write("\n\n")
        
        # Detailed Results
        f.write("DETAILED RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Shape Completion Performance:\n")
        f.write(df_shape_summary.to_string())
        f.write("\n\n")
        
        f.write("Size Evaluation Performance:\n")
        f.write(df_size_summary.to_string())
        f.write("\n\n")
        
        if df_inference_summary is not None:
            f.write("Inference Time Performance:\n")
            f.write(df_inference_summary.to_string())
            f.write("\n\n")
        
        f.write("Overfitting Analysis:\n")
        f.write(df_overfitting.to_string())
        f.write("\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Based on the analysis:\n\n")
        
        if best_f1_encoder == 'minkowski':
            f.write("1. Minkowski encoder shows superior performance for shape completion\n")
        else:
            f.write(f"1. {best_f1_encoder.upper()} encoder shows best performance, consider it for production\n")
        
        # Check if finetuning helps
        ft_helps = []
        for encoder in ['minkowski', 'pce']:
            if encoder in df_shape_summary['encoder'].values:
                ft_scores = df_shape_summary[(df_shape_summary['encoder'] == encoder) & 
                                            (df_shape_summary['training'] == 'finetune')]['f1_mean']
                fs_scores = df_shape_summary[(df_shape_summary['encoder'] == encoder) & 
                                            (df_shape_summary['training'] == 'from_scratch')]['f1_mean']
                
                if len(ft_scores) > 0 and len(fs_scores) > 0:
                    if ft_scores.mean() > fs_scores.mean():
                        ft_helps.append(encoder)
        
        if len(ft_helps) > 0:
            f.write(f"2. Finetuning improves performance for {', '.join(ft_helps).upper()}\n")
        else:
            f.write("2. Finetuning does not show consistent improvement\n")
        
        if len(severe_overfitting) > 0:
            f.write("3. Address overfitting by including more small fruit examples in training data\n")
        else:
            f.write("3. Model generalization across fruit sizes is satisfactory\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
    
    print(f"\n[Saved] Comprehensive report: {report_path}")
    
    return report_path


def main():
    # Configuration
    base_dir = Path("./experiments")
    output_dir = Path("./analysis")
    latent_sizes = [16, 32, 64, 128]  # All latent sizes
    
    print("="*80)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing latent sizes: {latent_sizes}")
    print(f"Output directory: {output_dir}")
    
    # Load all results
    df_shape, df_size = load_all_results(base_dir, latent_sizes)
    
    if len(df_shape) == 0 and len(df_size) == 0:
        print("\n[Error] No data found! Check that result files exist.")
        return
    
    # Analyze shape completion
    df_shape_summary = None
    if len(df_shape) > 0:
        df_shape_summary = analyze_shape_completion(df_shape, output_dir)
    else:
        print("\n[Warning] No shape completion data found")
    
    # Analyze size evaluation
    df_size_summary = None
    df_overfitting = None
    if len(df_size) > 0:
        df_size_summary, df_overfitting = analyze_size_evaluation(df_size, output_dir)
    else:
        print("\n[Warning] No size evaluation data found")
    
    # Analyze inference time
    df_inference_summary = None
    if len(df_shape) > 0:
        df_inference_summary = analyze_inference_time(df_shape, output_dir)
    
    # Generate comprehensive report
    if df_shape_summary is not None and df_size_summary is not None:
        generate_comprehensive_report(df_shape_summary, df_size_summary, df_overfitting, df_inference_summary, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - shape_completion_summary.csv (includes volume RMSE)")
    print("  - shape_completion_comparison.png (includes volume RMSE plot)")
    print("  - size_evaluation_summary.csv")
    print("  - size_errors_by_date_raw.png (before correction)")
    print("  - size_errors_by_date_corrected.png (after correction)")
    print("  - overfitting_analysis.csv")
    print("  - inference_time_summary.csv")
    print("  - inference_time_comparison.png")
    print("  - comprehensive_analysis_report.txt")


if __name__ == '__main__':
    main()

