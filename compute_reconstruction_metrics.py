import open3d as o3d
import os 
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from pathlib import Path

# Add root directory to python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _repo_bootstrap  # noqa: F401  (binds the repo root as `corepp`)
from dataloaders import dataset_layout as layout

from tqdm import tqdm
from corepp.metrics_3d import chamfer_distance, precision_recall


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Compute reconstruction metrics and save figures/CSVs")
    arg_parser.add_argument(
        "--ground-truth-dir",
        "-gt",
        dest="gt_dir",
        required=True,
    )
    arg_parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default=None,
        help="Directory to save metrics (default: <prediction-dir>/../metrics)",
    )
    arg_parser.add_argument(
        "--report-threshold",
        dest="report_t",
        type=float,
        default=0.005,
        help="Threshold to report PR/F1 values (meters)",
    )
    arg_parser.add_argument(
        "--prediction-dir",
        "-pt",
        dest="pt_dir",
        required=True,
    )

    args = arg_parser.parse_args()

    # Prepare output directory
    pt_dir = args.pt_dir
    if args.out_dir is None:
        base = os.path.dirname(os.path.normpath(pt_dir))
        out_dir = os.path.join(base, 'metrics')
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(out_dir, 'reconstruction_metrics.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Starting reconstruction metrics computation")
    logger.info("=" * 80)
    logger.info(f"Ground truth directory: {args.gt_dir}")
    logger.info(f"Prediction directory: {pt_dir}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Report threshold: {args.report_t} meters")
    logger.info(f"Log file: {log_file}")

    cd = chamfer_distance.ChamferDistance()
    pr = precision_recall.PrecisionRecall(0.001, 0.01, 50)
    logger.info("Initialized Chamfer Distance and Precision-Recall metrics")

    # Formatted per fruit below; layout.gt_cloud resolves either dataset layout.
    gt_dir = layout.gt_cloud(args.gt_dir, '{}')

    names = []
    skipped_not_mesh = 0
    skipped_empty = 0
    skipped_no_gt = 0
    processed = 0

    logger.info("Starting to process prediction files...")
    if not os.path.isdir(pt_dir):
        logger.error(f"Prediction directory does not exist: {pt_dir}")
        logger.error("Nothing to evaluate. Did you run reconstruction first (reconstruct_deep_sdf.py)?")
        raise SystemExit(2)

    pred_files = sorted(os.listdir(pt_dir))
    mesh_candidates = [f for f in pred_files if f.lower().endswith(('.ply', '.obj', '.stl', '.off'))]
    if len(mesh_candidates) == 0:
        logger.warning("No mesh files found in prediction directory; nothing to evaluate.")
        logger.warning(f"Prediction directory: {pt_dir}")
        logger.warning("If you expected meshes here, check your reconstruction output path.")
        # Write a minimal summary so downstream scripts don't crash.
        summary = {
            'cd_mean': float('nan'),
            'threshold_used': float(args.report_t),
            'precision_at_t': float('nan'),
            'recall_at_t': float('nan'),
            'f1_at_t': float('nan'),
            'auc_precision': float('nan'),
            'auc_recall': float('nan'),
            'auc_f1': float('nan'),
            'num_samples': 0,
            'note': 'No prediction meshes found; nothing evaluated.'
        }
        summary_path = os.path.join(out_dir, 'summary.json')
        with open(summary_path, 'w') as jf:
            json.dump(summary, jf, indent=2)
        logger.info(f"Saved empty summary to {summary_path}")
        raise SystemExit(0)

    for fname in tqdm(sorted(os.listdir(pt_dir))):
        if not fname.lower().endswith(('.ply', '.obj', '.stl', '.off')):
            skipped_not_mesh += 1
            continue
        pred_path = os.path.join(pt_dir, fname)
        pt_mesh = o3d.io.read_triangle_mesh(pred_path)
        if pt_mesh.is_empty():
            logger.warning(f"Skipping {fname}: empty mesh")
            skipped_empty += 1
            continue
        gt_path = gt_dir.format(fname[:-4])
        if not os.path.exists(gt_path):
            logger.warning(f"Skipping {fname}: ground truth not found at {gt_path}")
            skipped_no_gt += 1
            continue
        gt_pcd = o3d.io.read_point_cloud(gt_path)
        cd.update(gt_pcd, pt_mesh)
        pr.update(gt_pcd, pt_mesh)
        names.append(os.path.splitext(fname)[0])
        processed += 1

    logger.info(f"Processing complete: {processed} samples processed")
    logger.info(f"Skipped: {skipped_not_mesh} non-mesh files, {skipped_empty} empty meshes, {skipped_no_gt} missing ground truth")

    # Compute summary metrics
    logger.info("Computing summary metrics...")
    if processed == 0:
        logger.warning("0 samples processed; skipping metric computation to avoid errors.")
        summary = {
            'cd_mean': float('nan'),
            'threshold_used': float(args.report_t),
            'precision_at_t': float('nan'),
            'recall_at_t': float('nan'),
            'f1_at_t': float('nan'),
            'auc_precision': float('nan'),
            'auc_recall': float('nan'),
            'auc_f1': float('nan'),
            'num_samples': 0,
            'note': 'No valid mesh/GT pairs were processed.'
        }
        summary_path = os.path.join(out_dir, 'summary.json')
        with open(summary_path, 'w') as jf:
            json.dump(summary, jf, indent=2)
        logger.info(f"Saved empty summary to {summary_path}")
        raise SystemExit(0)

    cd_mean = cd.compute(print_output=False)
    pr_vals, re_vals, f1_vals = pr.compute_at_all_thresholds()
    t_arr = pr.thresholds
    pr_at, re_at, f1_at, t_used = pr.compute_at_threshold(args.report_t, print_output=False)
    pr_area, re_area, f1_area = pr.compute_auc()
    logger.info(f"Chamfer Distance (mean): {cd_mean:.6f} meters")
    logger.info(f"At threshold {t_used:.4f}m - Precision: {pr_at:.2f}%, Recall: {re_at:.2f}%, F1: {f1_at:.2f}%")
    logger.info(f"AUC - Precision: {pr_area:.2f}, Recall: {re_area:.2f}, F1: {f1_area:.2f}")

    # Save CSV with per-sample Chamfer distances
    if len(names) == len(cd.cd_array) and len(names) > 0:
        import csv
        csv_path = os.path.join(out_dir, 'per_sample_chamfer.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sample_id', 'chamfer_distance'])
            for n, v in zip(names, cd.cd_array):
                w.writerow([n, float(v)])
        logger.info(f"Saved per-sample Chamfer distances to {csv_path}")

    # Save JSON summary
    summary = {
        'cd_mean': float(cd_mean),
        'threshold_used': float(t_used),
        'precision_at_t': float(pr_at),
        'recall_at_t': float(re_at),
        'f1_at_t': float(f1_at),
        'auc_precision': float(pr_area),
        'auc_recall': float(re_area),
        'auc_f1': float(f1_area),
        'num_samples': int(len(names))
    }
    summary_path = os.path.join(out_dir, 'summary.json')
    with open(summary_path, 'w') as jf:
        json.dump(summary, jf, indent=2)
    logger.info(f"Saved summary metrics to {summary_path}")

    # Save curves as SVG (efficient and scalable)
    logger.info("Generating and saving visualization plots...")
    def _save_svg(fig, path):
        fig.savefig(path, bbox_inches='tight', dpi=150, format='svg')
        plt.close(fig)

    # Precision/Recall/F1 vs threshold
    fig1 = plt.figure()
    plt.plot(t_arr, pr_vals, label='Precision')
    plt.plot(t_arr, re_vals, label='Recall')
    plt.plot(t_arr, f1_vals, label='F1')
    plt.axvline(t_used, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Threshold (m)')
    plt.ylabel('Score (%)')
    plt.title('PR/F1 vs Threshold')
    plt.legend()
    pr_f1_path = os.path.join(out_dir, 'pr_f1_vs_threshold.svg')
    _save_svg(fig1, pr_f1_path)
    logger.info(f"Saved PR/F1 vs threshold plot to {pr_f1_path}")

    # PR curve (Recall vs Precision)
    fig2 = plt.figure()
    plt.plot(re_vals, pr_vals)
    plt.xlabel('Recall (%)')
    plt.ylabel('Precision (%)')
    plt.title('Precision-Recall Curve')
    pr_curve_path = os.path.join(out_dir, 'pr_curve.svg')
    _save_svg(fig2, pr_curve_path)
    logger.info(f"Saved PR curve to {pr_curve_path}")

    # Chamfer histogram
    if len(cd.cd_array) > 0:
        fig3 = plt.figure()
        plt.hist(np.asarray(cd.cd_array), bins=30)
        plt.xlabel('Chamfer Distance (m)')
        plt.ylabel('Count')
        plt.title('Chamfer Distance Distribution')
        chamfer_hist_path = os.path.join(out_dir, 'chamfer_histogram.svg')
        _save_svg(fig3, chamfer_hist_path)
        logger.info(f"Saved Chamfer histogram to {chamfer_hist_path}")

    # Also save arrays for reproducibility
    curves_path = os.path.join(out_dir, 'curves.npz')
    np.savez_compressed(curves_path, thresholds=t_arr, precision=np.array(pr_vals), recall=np.array(re_vals), f1=np.array(f1_vals))
    logger.info(f"Saved curve arrays to {curves_path}")

    logger.info("=" * 80)
    logger.info("Reconstruction metrics computation completed successfully!")
    logger.info("=" * 80)
