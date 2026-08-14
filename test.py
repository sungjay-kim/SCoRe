#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _repo_bootstrap  # noqa: F401  (binds the repo root as `corepp`)
from re import I
from numpy.core.fromnumeric import mean
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
from tqdm import tqdm

from torchvision.transforms.transforms import Compose

import deepsdf.deep_sdf as deep_sdf
import deepsdf.deep_sdf.workspace as ws
import deepsdf.deep_sdf.o3d_utils as o3d_utils


from sdfrenderer.grid import Grid3D
from dataloaders.transforms import Pad
from dataloaders.cameralaser_w_masks import MaskedCameraLaserData

from networks.models import (
    Encoder,
    EncoderBig,
    ERFNetEncoder,
    EncoderBigPooled,
    EncoderPooled,
    DoubleEncoder,
    PointCloudEncoder,
    PointCloudEncoderMaxMean,
    PointCloudEncoderMaxMeanMasked,
    PointCloudEncoderLarge,
    PointCloudEncoderMaxMeanLarge,
    PointCloudEncoderMaxMeanLargeMasked,
    PointNetPlusPlusEncoder,
    FoldNetEncoder,
    MinkLatentEncoder,
)
import networks.utils as net_utils

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

import time
import json

from utils import sdf2mesh_cuda, mesh_volume_ml, tensor_dict_2_float_dict
from metrics_3d import chamfer_distance, precision_recall
from temporal.core import apply_test_occlusion, stable_sample_seed
from dataloaders import dataset_layout as layout


def _safe_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def write_metrics_summary_from_results_csv(
    results_csv_path: Path,
    output_json_path: Path,
    valid_mesh_filter: bool = True,
) -> dict:
    """
    Aggregate metrics from `shape_completion_results.csv` into a JSON summary.
    Uses only values already written to the CSV (no recomputation).
    """
    df = pd.read_csv(results_csv_path)

    # Candidate numeric metrics typically present in shape_completion_results.csv
    metric_cols = [
        "chamfer_distance",
        "auc_precision",
        "auc_recall",
        "auc_f1",
        "precision_5mm",
        "recall_5mm",
        "f1_5mm",
        "mesh_volume_ml",
        "gt_volume_ml",
        "sfm_volume_ml",
        "encoder_time_ms",
        "decoder_time_ms",
        "mesh_extraction_time_ms",
        "total_time_ms",
    ]

    # Build a numeric view
    numeric = {c: _safe_numeric_series(df, c) for c in metric_cols}
    numeric_df = pd.DataFrame(numeric)

    df_all = numeric_df.copy()
    df_valid = df_all
    if valid_mesh_filter and "mesh_volume_ml" in df_all.columns:
        # Mesh volume = 0 is used when mesh extraction fails / mesh is non-watertight in this script.
        df_valid = df_all[df_all["mesh_volume_ml"].fillna(0) > 0].copy()

    # Columns that are only populated for last-date (when skip_metrics_intermediate=True).
    # For these, zeros represent "skipped" not "actual zero performance" — filter them out.
    _last_date_only_cols = {"chamfer_distance", "auc_precision", "auc_recall", "auc_f1",
                             "precision_5mm", "recall_5mm", "f1_5mm"}
    def summarize(sub: pd.DataFrame) -> dict:
        out = {"num_samples": int(len(sub))}
        for c in metric_cols:
            if c not in sub.columns:
                continue
            s = sub[c].dropna()
            # For chamfer/PR/F1 columns, drop rows where value==0 (those are skipped intermediate frames)
            if c in _last_date_only_cols:
                s = s[s != 0]
            if len(s) == 0:
                out[c] = {"count": 0}
                continue
            out[c] = {
                "count": int(s.shape[0]),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
                "median": float(s.median()),
                "min": float(s.min()),
                "max": float(s.max()),
            }

        # Volume RMSE between predicted mesh volume and GT volume (last-date rows only when available)
        # Uses the CSV directly so we can group by fruit_id + frame_id.
        try:
            fid_col = 'kmelon_id' if 'kmelon_id' in df.columns else ('potato_id' if 'potato_id' in df.columns else None)
            if fid_col and 'frame_id' in df.columns and 'gt_volume_ml' in df.columns and 'mesh_volume_ml' in df.columns:
                sub_df = df.loc[sub.index] if set(sub.index).issubset(df.index) else df
                valid_rows = sub_df[(sub_df['mesh_volume_ml'].fillna(0) > 0)].copy()
                if len(valid_rows):
                    valid_rows['__f'] = pd.to_numeric(valid_rows['frame_id'].astype(str).str[:8], errors='coerce')
                    last_idx = valid_rows.groupby(fid_col)['__f'].idxmax()
                    last = valid_rows.loc[last_idx]
                    mesh_vol_last = pd.to_numeric(last['mesh_volume_ml'], errors='coerce')
                    gt_vol_last = pd.to_numeric(last['gt_volume_ml'], errors='coerce')
                    diff = (mesh_vol_last - gt_vol_last).dropna()
                    if len(diff) > 0:
                        rmse = float(np.sqrt(np.mean(np.square(diff.to_numpy()))))
                        mae = float(np.mean(np.abs(diff.to_numpy())))
                        bias = float(np.mean(diff.to_numpy()))
                        pe = (diff.abs() / gt_vol_last).dropna() * 100.0
                        out["volume_rmse_ml"] = {"count": int(diff.shape[0]), "rmse": rmse,
                                                  "mae": mae, "bias": bias,
                                                  "mape_pct": float(pe.mean()) if len(pe) else None}
                    else:
                        out["volume_rmse_ml"] = {"count": 0}
        except Exception:
            pass

        # Convenience throughput
        if "total_time_ms" in sub.columns:
            tt = sub["total_time_ms"].dropna()
            if len(tt) > 0 and float(tt.mean()) > 0:
                out["throughput_samples_per_sec"] = float(1000.0 / float(tt.mean()))
        return out

    summary = {
        "results_csv": str(results_csv_path),
        "generated_at_unix_s": float(time.time()),
        "all_rows": summarize(df_all),
        "valid_mesh_rows": summarize(df_valid),
        "notes": {
            "valid_mesh_rows_filter": "mesh_volume_ml > 0" if valid_mesh_filter else "disabled",
            "source": "Aggregated from shape_completion_results.csv (no recomputation).",
        },
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary

def sample_mesh(mesh, num_points):
    if mesh is None:
        return None
    try:
        return mesh.sample_points_uniformly(number_of_points=num_points)
    except Exception:
        return None


def colorize_point_cloud(pcd, gt_pcd, cmap_name='YlOrRd', max_distance=None, return_distances=False):
    """
    Colorize predicted point cloud based on per-point distance to ground truth.
    
    Args:
        pcd: Predicted point cloud (o3d.geometry.PointCloud)
        gt_pcd: Ground truth point cloud (o3d.geometry.PointCloud)
        cmap_name: Matplotlib colormap name. Options:
                   - 'YlOrRd': Yellow (close) to Red (far) - warm gradient
                   - 'jet': Blue (close) to Red (far) - full spectrum
                   - 'viridis': Purple (close) to Yellow (far) - perceptually uniform
                   - 'RdYlGn_r': Red (far) to Green (close) - traffic light style
                   - 'plasma', 'inferno', 'turbo', etc.
        max_distance: Maximum distance for color scale (None = auto from data)
        return_distances: If True, also return the distance array
    
    Returns:
        colored point cloud (and optionally distance array)
    
    Note: This computes one-directional distance (pred→GT), which is appropriate
          for visualizing prediction errors. Full chamfer distance (bidirectional)
          is computed separately for the numeric metric.
    """
    if pcd is None or gt_pcd is None:
        return (None, None) if return_distances else None
    try:
        cmap = plt.get_cmap(cmap_name)
        
        # Compute per-point distance from predicted to GT
        dist = np.asarray(pcd.compute_point_cloud_distance(gt_pcd))
        
        # Store original distances for statistics
        dist_original = dist.copy()
        
        # Normalize distances for colormap
        dist_min = dist.min()
        if max_distance is not None:
            # Use specified max distance for consistent color scale across samples
            dist_normalized = np.clip((dist - dist_min) / (max_distance - dist_min), 0, 1)
        else:
            # Auto-scale based on this sample
            dist_max = dist.max()
            if dist_max > dist_min:
                dist_normalized = (dist - dist_min) / (dist_max - dist_min)
            else:
                dist_normalized = np.zeros_like(dist)
        
        # Apply colormap
        colors = cmap(dist_normalized)[:, :3]  # Take only RGB, drop alpha
        
        # Create colored point cloud
        colored = o3d.geometry.PointCloud(pcd)
        colored.colors = o3d.utility.Vector3dVector(colors)
        
        if return_distances:
            return colored, dist_original
        return colored
    except Exception as e:
        print(f"Failed to colorize point cloud: {e}")
        return (None, None) if return_distances else None


def save_colorbar_legend(output_path, cmap_name='jet', min_val=0.0, max_val=0.01, 
                         title="Distance to Ground Truth (m)", figsize=(2, 6), dpi=150):
    """
    Save a colorbar legend as an image file for reference.
    
    Args:
        output_path: Path to save the colorbar image
        cmap_name: Name of the colormap used
        min_val: Minimum value of the color scale
        max_val: Maximum value of the color scale
        title: Title for the colorbar
        figsize: Size of the figure (width, height)
        dpi: Resolution of the saved image
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(right=0.5)
        
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
        
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                         cax=ax, orientation='vertical')
        cb.set_label(title, rotation=270, labelpad=20)
        
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Colorbar legend saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save colorbar legend: {e}")


cd = chamfer_distance.ChamferDistance()
pr = precision_recall.PrecisionRecall(0.001, 0.01, 10)

torch.autograd.set_detect_anomaly(True)


def main_function(decoder, pretrain, cfg, latent_size, checkpoint_dir_override=None, data_dir_override=None, occlusion_pct=0, occlusion_mode="patch", last_frame_only=False, output_dir_override=None, save_latents=False, min_n_points=0, n_points_cache=None, threshold_override=None, drop_first_date_per_fruit=False):
    torch.manual_seed(133)
    np.random.seed(133)
    torch.cuda.manual_seed_all(133)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    df = pd.read_csv("./data/3DKmelonDemo/ground_truth.csv")
    columns = ['kmelon_id',
                'frame_id',
                'vertical_pos',
                'cultivar',
                'weight_g',
                'gt_volume_ml',
                'sfm_volume_ml',
                'mesh_volume_ml',
                'mlp_vol_ml',
                'mesh_height_mm',
                'mesh_width_mm',
                'chamfer_distance',
                'auc_precision',
                'auc_recall',
                'auc_f1',
                'precision_5mm',
                'recall_5mm',
                'f1_5mm',
                'latent_norm',
                'latent_mean',
                'latent_std',
                'encoder_time_ms',
                'decoder_time_ms',
                'mesh_extraction_time_ms',
                'total_time_ms']
    save_df = pd.DataFrame(columns=columns)
    _rows: list = []  # accumulate per-sample dicts; build save_df once at end (avoids O(n^2) concat)

    exec_time = []

    with open(cfg) as json_file:
        param = json.load(json_file)

    if checkpoint_dir_override:
        param['checkpoint_dir'] = checkpoint_dir_override
    
    if data_dir_override:
        param['data_dir'] = data_dir_override
        print(f"[test] Overriding data_dir: {data_dir_override}")

    # Select the appropriate ground truth CSV based on species, fallback to potato if unknown
    try:
        species = str(param.get('species', '')).lower()
        if 'kmelon' in species:
            df = pd.read_csv("./data/3DKmelonDemo/ground_truth.csv")
        else:
            df = pd.read_csv("./data/3DPotatoTwinDemo/ground_truth.csv")
    except Exception as ex:
        print('Failed to load ground truth CSV based on species; using default. Reason:', ex)

    device = 'cuda'

    checkpoint_dir = Path(param['checkpoint_dir']).resolve()

    if output_dir_override is not None:
        run_root = Path(output_dir_override).resolve()
    else:
        run_root = checkpoint_dir.parent.parent
    run_root.mkdir(parents=True, exist_ok=True)

    # Save test config for reproducibility
    cfg_fname = cfg.split('/')[-1].replace('.json', '')
    test_config_save_path = run_root / f"{cfg_fname}_test.json"
    with open(test_config_save_path, 'w') as f:
        json.dump(param, f, indent=4)
    print(f"[test] Config saved to: {test_config_save_path}")

    occ_suffix = f"_occ{occlusion_pct}" if occlusion_pct > 0 else ""
    eval_root = run_root / "eval"
    sweep_dir = eval_root / "shape_completion_results"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_csv_path = sweep_dir / f"shape_completion_results{occ_suffix}.csv"
    if occlusion_pct > 0:
        print(f"[test] Occlusion level: {occlusion_pct}%")

    threshold = param['threshold'] if threshold_override is None else float(threshold_override)
    if threshold_override is not None:
        print(f"[test] threshold override: {param['threshold']} -> {threshold}")
    grid_density = param['grid_density']
    precision = torch.float32

    save_mesh = bool(param.get('save_mesh', False))
    save_pcd = bool(param.get('save_pcd', False))
    save_colored_mesh = bool(param.get('save_colored_mesh', False))
    save_mesh_format = str(param.get('save_mesh_format', 'ply'))
    pcd_num_points = int(param.get('pcd_num_points', grid_density ** 3))

    inf_dir = run_root / "inf"
    inf_dir.mkdir(parents=True, exist_ok=True)

    mesh_dir = inf_dir / "meshes"
    pcd_dir = inf_dir / "pointclouds"
    colored_pcd_dir = inf_dir / "colored_pointclouds"
    if save_mesh:
        mesh_dir.mkdir(parents=True, exist_ok=True)
    if save_pcd:
        pcd_dir.mkdir(parents=True, exist_ok=True)
    if save_colored_mesh:
        colored_pcd_dir.mkdir(parents=True, exist_ok=True)
        
        # Save colorbar legend for reference
        cmap_name = param.get('visualization_colormap', 'jet')
        max_dist = param.get('visualization_max_distance', 0.02)  # Default 2cm for scale
        legend_path = colored_pcd_dir / f"colorbar_legend_{cmap_name}.png"
        save_colorbar_legend(legend_path, cmap_name=cmap_name, 
                           min_val=0.0, max_val=max_dist,
                           title="Distance to GT (m)")
        print(f"[Visualization] Using colormap: '{cmap_name}' with max distance: {max_dist}m")

    # define encoder
    if param['encoder'] == 'big':
        encoder = EncoderBig(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'small_pool':
        encoder = EncoderPooled(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'erfnet':
        encoder = ERFNetEncoder(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'pool':
        encoder = EncoderBigPooled(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'double':
        encoder = DoubleEncoder(out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'point_cloud':
        encoder = PointCloudEncoder(in_channels=3, out_channels=latent_size).to(device)
    elif param['encoder'] == 'point_cloud_maxmean':
        encoder = PointCloudEncoderMaxMean(in_channels=3, out_channels=latent_size).to(device)
    elif param['encoder'] == 'point_cloud_maxmean_masked':
        encoder = PointCloudEncoderMaxMeanMasked(
            in_channels=3, out_channels=latent_size,
            volume_head=bool(param.get('volume_head', False)),
            nll_head=bool(param.get('nll_super_loss', False)),
        ).to(device)
    elif param['encoder'] == 'point_cloud_large':
        encoder = PointCloudEncoderLarge(
            in_channels=3, out_channels=latent_size,
            volume_head=bool(param.get('volume_head', False)),
        ).to(device)
    elif param['encoder'] == 'point_cloud_maxmean_large':
        encoder = PointCloudEncoderMaxMeanLarge(
            in_channels=3, out_channels=latent_size,
            volume_head=bool(param.get('volume_head', False)),
        ).to(device)
    elif param['encoder'] == 'point_cloud_maxmean_large_masked':
        encoder = PointCloudEncoderMaxMeanLargeMasked(
            in_channels=3, out_channels=latent_size,
            volume_head=bool(param.get('volume_head', False)),
            nll_head=bool(param.get('nll_super_loss', False)),
        ).to(device)
    elif param['encoder'] == 'pointnet2':
        encoder = PointNetPlusPlusEncoder(in_channels=3, out_channels=latent_size).to(device)
    elif param['encoder'] == 'foldnet':
        encoder = FoldNetEncoder(in_channels=3, out_channels=latent_size).to(device)
    elif param['encoder'] == 'minkowski':
        voxel = float(param.get('mink_voxel_size', 0.01))
        normalize = bool(param.get('mink_normalize', True))
        encoder = MinkLatentEncoder(
            in_channels=3,
            out_channels=latent_size,
            voxel_size=voxel,
            normalize_input=normalize,
        ).to(device)
    else:
        encoder = Encoder(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_field = param.get('checkpoint_file', '')
    ckpt_path = Path(ckpt_field)

    if ckpt_path.is_absolute():
        ckpt = ckpt_path
    else:
        ckpt = checkpoint_dir / ckpt_path.name

    if not ckpt.exists():
        candidates = set()

        if ckpt_path.name.startswith('_'):
            candidates.add(checkpoint_dir / f"checkpoints{ckpt_path.name}")
            candidates.add(checkpoint_dir / ckpt_path.name.lstrip('_'))

        stem = ckpt_path.stem.lstrip('_')
        if stem:
            pattern = f"*{stem}*.pt"
            candidates.update(checkpoint_dir.glob(pattern))

        candidates = [c for c in candidates if c.exists()]

        if len(candidates) == 1:
            ckpt = candidates[0]
            print(f"[test] Resolved checkpoint to {ckpt}")
        elif len(candidates) > 1:
            candidate_list = '\n'.join(str(c) for c in sorted(candidates))
            raise FileNotFoundError(
                f"Multiple checkpoint candidates found for '{ckpt_field}'. Specify 'checkpoint_file' explicitly.\nCandidates:\n{candidate_list}"
            )
        else:
            available = '\n'.join(str(c) for c in checkpoint_dir.glob('*.pt'))
            raise FileNotFoundError(
                f"Checkpoint '{ckpt_field}' not found in {checkpoint_dir}. Available files:\n{available}"
            )
    # import ipdb;ipdb.set_trace()
    encoder.load_state_dict(torch.load(ckpt)['encoder_state_dict'])
    decoder.load_state_dict(torch.load(ckpt)['decoder_state_dict'])
    encoder.eval()
    decoder.eval()

    ##############################
    #  TESTING LOOP STARTS HERE  #
    ##############################

    decoder.to(device)

    # transformations
    tfs = [Pad(size=param["input_size"])]
    tf = Compose(tfs)

    cl_dataset = MaskedCameraLaserData(data_source=param["data_dir"],
                                        tf=tf, 
                                        color_tf = None,
                                        pretrain=pretrain,
                                        pad_size=param.get("input_size", 1024),
                                        detection_input=param.get("detection_input", "mask"),
                                        normalize_depth=param.get("normalize_depth", True),
                                        depth_min=param.get("depth_min", 400),
                                        depth_max=param.get("depth_max", 900),
                                        supervised_3d=param.get("supervised_3d", True),
                                        sdf_loss=param.get("3D_loss", False),
                                        grid_density=param.get("grid_density", 50),
                                        split=param.get('eval_split', 'test'),
                                        overfit=False,
                                        species=param.get("species", "kmelon"),
                                        use_phenobox_pcd=bool(param.get("use_phenobox_pcd", False)),
                                        phenobox_subdir=param.get("phenobox_subdir", "phenobox/pointclouds"),
                                        use_orbbec_pcd=bool(param.get("use_orbbec_pcd", False)),
                                        orbbec_pcd_suffix=param.get("orbbec_pcd_suffix", "_preprocessed"),
                                        split_json_path=param.get("split_json", None),
                                        require_dataset_json=param.get("require_dataset_json", True),
                                        temporal_loading=bool(param.get("temporal_loading", False))
                                        )

    # Optional: drop the EARLIEST observation date per fruit_id BEFORE other filters.
    # First-date frames are typically infant fruits (very small ellipsoid, noisy).
    if drop_first_date_per_fruit:
        import re as _re_fd
        _date_re_fd = _re_fd.compile(r"^(\d{8})")
        _earliest_fd = {}
        for _f in getattr(cl_dataset, "files", []):
            _fid = layout.fruit_id_from_path(_f)
            _m = _date_re_fd.match(os.path.basename(_f))
            if not _m: continue
            _d = _m.group(1)
            if _fid not in _earliest_fd or _d < _earliest_fd[_fid]:
                _earliest_fd[_fid] = _d
        kept_fd = []
        for _f in cl_dataset.files:
            _fid = layout.fruit_id_from_path(_f)
            _m = _date_re_fd.match(os.path.basename(_f))
            if _m and _m.group(1) == _earliest_fd.get(_fid):
                continue  # drop earliest-date frame
            kept_fd.append(_f)
        n_before_fd = len(cl_dataset.files)
        cl_dataset.files = kept_fd
        print(f"[test] drop_first_date_per_fruit: {n_before_fd} -> {len(kept_fd)} frames "
              f"(dropped {n_before_fd - len(kept_fd)} fruits' first observation)")

    # Optional: keep only the last (latest-date) frame per fruit_id, matching
    # the temporal evaluation protocol (occlusion sweep is then comparable
    # 1-to-1 between static and temporal runs).
    if last_frame_only:
        import re as _re_lf
        _date_re_lf = _re_lf.compile(r"^(\d{8})")
        _latest_lf = {}
        for _f in getattr(cl_dataset, "files", []):
            _fid = layout.fruit_id_from_path(_f)
            _m = _date_re_lf.match(os.path.basename(_f))
            if not _m:
                continue
            _d = _m.group(1)
            if _d > _latest_lf.get(_fid, ""):
                _latest_lf[_fid] = _d
        kept = []
        for _f in cl_dataset.files:
            _fid = layout.fruit_id_from_path(_f)
            _m = _date_re_lf.match(os.path.basename(_f))
            if _m and _m.group(1) == _latest_lf.get(_fid):
                kept.append(_f)
        n_before = len(cl_dataset.files)
        cl_dataset.files = kept
        print(f"[test] last_frame_only=True: {n_before} -> {len(kept)} frames "
              f"({len(_latest_lf)} fruits)")

    # Optional: drop low-info frames whose input PCD has fewer than min_n_points real points.
    # Applied to TEST split only (val unchanged). Mirrors temporal/test.py semantics.
    _min_n = int(min_n_points or 0)
    if _min_n > 0:
        orbbec_suf = param.get("orbbec_pcd_suffix", "_preprocessed")
        n_pts_lookup = {}
        if n_points_cache and os.path.isfile(n_points_cache):
            _c = pd.read_csv(n_points_cache)
            n_pts_lookup = {(str(r["fruit_id"]), str(r["date"])): int(r["n_points"]) for _, r in _c.iterrows()}
            print(f"[test] loaded n_points cache: {len(n_pts_lookup)} entries from {n_points_cache}")
        kept_n = []
        n_drop = 0
        import re as _re_np
        _date_re_np = _re_np.compile(r"^(\d{8})")
        for _f in cl_dataset.files:
            _fid = layout.fruit_id_from_path(_f)
            _base = os.path.splitext(os.path.basename(_f))[0]
            _frame = _base.replace(orbbec_suf, '') if _base.endswith(orbbec_suf) else _base
            _m = _date_re_np.match(_frame)
            _date = _m.group(1) if _m else ''
            n_pts = n_pts_lookup.get((_fid, _date))
            if n_pts is None:
                try:
                    n_pts = int(np.asarray(o3d.io.read_point_cloud(_f).points).shape[0])
                except Exception:
                    n_pts = -1
            if n_pts >= _min_n:
                kept_n.append(_f)
            else:
                n_drop += 1
        n_before_n = len(cl_dataset.files)
        cl_dataset.files = kept_n
        print(f"[test] min_n_points={_min_n}: kept {len(kept_n)}/{n_before_n} frames "
              f"(dropped {n_drop} = {100*n_drop/max(n_before_n,1):.1f}%)")

    _nw = int(param.get("test_num_workers", 0))
    dataset = DataLoader(cl_dataset, batch_size=1, shuffle=False,
                         num_workers=_nw, pin_memory=True,
                         persistent_workers=(_nw > 0))

    with torch.no_grad():
        
        # GPU Warm-up: Run a dummy inference to trigger CUDA initialization
        # This ensures the first real sample has accurate timing
        print("\n[GPU Warm-up] Running dummy inference to initialize CUDA context...")
        try:
            dummy_item = next(iter(dataset))
            if param['encoder'] == 'minkowski':
                dummy_input = dummy_item['partial_pcd_world'].to(device)
                dummy_centers = dummy_item['bbox_center_tensor'].to(device)
                dummy_scales = dummy_item['bbox_scale_tensor'].to(device)
                dummy_latent = encoder(dummy_input, dummy_centers, dummy_scales)
            elif param['encoder'] in ['point_cloud', 'point_cloud_maxmean', 'point_cloud_maxmean_masked', 'point_cloud_large', 'point_cloud_maxmean_large', 'point_cloud_maxmean_large_masked', 'pointnet2', 'foldnet']:
                dummy_input = dummy_item['partial_pcd'].permute(0, 2, 1).to(device)
                dummy_latent = encoder(dummy_input)
            else:
                dummy_input = torch.cat((dummy_item['rgb'], dummy_item['depth']), 1).to(device)
                dummy_latent = encoder(dummy_input)
            
            # Warm up decoder
            dummy_box = tensor_dict_2_float_dict(dummy_item['bbox']) if 'bbox' in dummy_item else dataset.dataset.box
            dummy_grid = Grid3D(grid_density, device, precision, bbox=dummy_box)
            dummy_dec_input = torch.cat([dummy_latent.expand(dummy_grid.points.size(0), -1),
                                         dummy_grid.points], dim=1).to(dummy_latent.device, dummy_latent.dtype)
            dummy_sdf = decoder(dummy_dec_input)
            torch.cuda.synchronize()
            
            # Warm up mesh extraction (marching cubes)
            try:
                dummy_mesh = sdf2mesh_cuda(dummy_sdf, dummy_grid.points, t=threshold)
            except Exception:
                pass  # Warm-up failed, but not critical
            
            print("[GPU Warm-up] ✓ Complete (encoder + decoder + mesh extraction).")
            print("[GPU Warm-up] All subsequent samples will have accurate timing.\n")
        except Exception as ex:
            print(f"[GPU Warm-up] Warning: Warm-up failed ({ex}). First sample may have initialization overhead.")
        
        # Reset dataset iterator for actual testing
        _nw = int(param.get("test_num_workers", 0))
        dataset = DataLoader(cl_dataset, batch_size=1, shuffle=False,
                             num_workers=_nw, pin_memory=True,
                             persistent_workers=(_nw > 0))

        # Pre-compute latest date per fruit_id so intermediate frames can skip expensive metrics
        import re as _re
        _date_re = _re.compile(r"^(\d{8})")
        _latest_date_per_fruit = {}
        for _f in getattr(cl_dataset, "files", []):
            _fid = layout.fruit_id_from_path(_f)
            _base = os.path.basename(_f)
            _m = _date_re.match(_base)
            if not _m: continue
            _d = _m.group(1)
            if _d > _latest_date_per_fruit.get(_fid, ""):
                _latest_date_per_fruit[_fid] = _d
        # Default True: chamfer/PR/F1 only computed for latest-date frame per fruit.
        # Reason: GT is a single per-fruit laser PLY (final destructive scan), so comparing
        # intermediate-date predictions against final-state GT is meaningless (size mismatch).
        # Volume metrics are still computed for intermediate frames (they're scored against
        # corrected ellipsoid GT in eval_unified.py).
        _skip_metrics_intermediate = bool(param.get("skip_metrics_intermediate", True))
        if _skip_metrics_intermediate:
            print(f"[test] skip_metrics_intermediate=True (default). chamfer/PR only for latest-date per fruit ({len(_latest_date_per_fruit)} fruits).")

        for n_iter, item in enumerate(tqdm(iter(dataset))):
            volume, chamfer_distance, prec, rec, f1 = 0, 0, 0, 0, 0
            if 'bbox' in item:
                box = tensor_dict_2_float_dict(item['bbox'])
            else:
                try:
                    box = tensor_dict_2_float_dict(dataset.dataset.box)
                except Exception:
                    box = dataset.dataset.box

            cs = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
            gt = o3d.geometry.PointCloud()
            # Determine early whether this frame needs chamfer/PR metrics (so gt load can be skipped)
            _fd_early = _date_re.match(item['frame_id'][0])
            _fd_str = _fd_early.group(1) if _fd_early else ""
            _fid_early = item['fruit_id'][0]
            _skip_gt = _skip_metrics_intermediate and (_fd_str != _latest_date_per_fruit.get(_fid_early, _fd_str))
            if not _skip_gt:
                try:
                    if 'target_pcd' in item:
                        gt_np = item['target_pcd'][0].numpy()
                        bbox_center = item['bbox_center_tensor'][0].cpu().numpy()
                        bbox_scale = item['bbox_scale_tensor'][0].cpu().numpy()
                        gt_np = gt_np * bbox_scale + bbox_center
                    else:
                        gt_path = layout.gt_cloud(param["data_dir"], item['fruit_id'][0])
                        gt_pcd = o3d.io.read_point_cloud(gt_path)
                        gt_np = np.asarray(gt_pcd.points)
                    gt.points = o3d.utility.Vector3dVector(gt_np)
                except Exception as ex:
                    print('Failed to load GT point cloud:', ex)
                    gt.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            else:
                gt.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))

            start_total = time.time()

            # Apply test-time occlusion if requested
            if occlusion_pct > 0 and param['encoder'] in ['point_cloud', 'point_cloud_maxmean', 'point_cloud_maxmean_masked', 'point_cloud_large', 'point_cloud_maxmean_large', 'point_cloud_maxmean_large_masked', 'pointnet2', 'foldnet']:
                occ_rate = occlusion_pct / 100.0
                pcd = item['partial_pcd'][0]  # Remove batch dim: (N, 3)
                # Stable, dataset-order-independent seed so the same sample at
                # the same ratio gets the same occlusion across static/temporal
                # runs and across split changes.
                _occ_seed = stable_sample_seed(item['fruit_id'][0])
                pcd = apply_test_occlusion(
                    pcd,
                    occlusion_frac=occ_rate,
                    min_points=int(param.get("occlusion_min_points", 64)),
                    seed=_occ_seed,
                    mode=occlusion_mode,
                )
                item['partial_pcd'] = pcd.unsqueeze(0)  # Add batch dim back

            # Encoder timing
            start_encoder = time.time()
            # When the encoder has a vol_head MLP (volume_head=True at training), also extract
            # the MLP-predicted log_vol so we can save mlp_vol_ml in the eval CSV alongside
            # the decoder-based mesh_volume_ml. Allows comparing both inference paths.
            log_vol_mlp = None
            _use_mlp_path = bool(param.get('volume_head', False)) and hasattr(encoder, 'encode_with_vol')

            # unpacking inputs
            if param['encoder'] not in ['point_cloud', 'point_cloud_maxmean', 'point_cloud_maxmean_masked', 'point_cloud_large', 'point_cloud_maxmean_large', 'point_cloud_maxmean_large_masked', 'pointnet2', 'foldnet', 'minkowski']:
                encoder_input = torch.cat((item['rgb'], item['depth']), 1).to(device)
                latent = encoder(encoder_input)
            elif param['encoder'] == 'minkowski':
                partial_points = item['partial_pcd_world'].to(device)
                bbox_centers = item['bbox_center_tensor'].to(device)
                bbox_scales = item['bbox_scale_tensor'].to(device)
                latent = encoder(partial_points, bbox_centers, bbox_scales)
            else:
                if isinstance(item['partial_pcd'], list):
                    # Handle list of tensors with variable sizes - resample to fixed size
                    target_points = param["input_size"]
                    resampled_pcds = []
                    for p in item['partial_pcd']:
                        num_points = p.shape[0]
                        if num_points > target_points:
                            indices = torch.randperm(num_points)[:target_points]
                            resampled = p[indices]
                        elif num_points < target_points:
                            indices = torch.randint(0, num_points, (target_points,))
                            resampled = p[indices]
                        else:
                            resampled = p
                        resampled_pcds.append(resampled.permute(1, 0))
                    encoder_input = torch.stack(resampled_pcds, dim=0).to(device)
                else:
                    encoder_input = item['partial_pcd'].permute(0, 2, 1).to(device)
                if param['encoder'] in ('point_cloud_maxmean_masked', 'point_cloud_maxmean_large_masked') and 'partial_pcd_real_mask' in item:
                    real_mask = item['partial_pcd_real_mask'].to(device)
                    if _use_mlp_path:
                        latent, log_vol_mlp = encoder.encode_with_vol(encoder_input, mask=real_mask)
                    else:
                        latent = encoder(encoder_input, mask=real_mask)
                else:
                    if _use_mlp_path:
                        latent, log_vol_mlp = encoder.encode_with_vol(encoder_input)
                    else:
                        latent = encoder(encoder_input)
            torch.cuda.synchronize()  # Ensure GPU operations complete
            encoder_time = (time.time() - start_encoder) * 1000  # Convert to ms

            # Convert MLP-head log_vol to ml (per-sample) for CSV row dump
            mlp_vol_ml = None
            if log_vol_mlp is not None:
                try:
                    mlp_vol_ml = torch.exp(log_vol_mlp).detach().cpu().numpy().reshape(-1).tolist()
                except Exception:
                    mlp_vol_ml = None

            # Optional: dump full latent vector to disk for downstream analysis
            # (norm/mean/std are now stored in the per-frame CSV, so this is off by default)
            if bool(param.get('save_latent_codes', False)):
                latent_save = latent.detach().to('cpu').squeeze()
                save_path = os.path.join(os.path.dirname(pretrain), "encoder")
                os.makedirs(save_path, exist_ok=True)
                torch.save(latent_save, os.path.join(save_path, item['frame_id'][0] + ".pth"))

            # Sweep-time latent dump: per (sample, occlusion level), into the
            # run's output dir so we can analyze fallback / regression-to-mean.
            if save_latents:
                lat_dir = run_root / "latents"
                lat_dir.mkdir(parents=True, exist_ok=True)
                _suffix = f"_occ{occlusion_pct}" if occlusion_pct > 0 else ""
                _fid = item['fruit_id'][0]
                _frm = item['frame_id'][0]
                np.save(
                    str(lat_dir / f"{_fid}_{_frm}{_suffix}.npy"),
                    latent.detach().to('cpu').squeeze().numpy(),
                )

            # Decoder timing
            start_decoder = time.time()
            grid_3d = Grid3D(grid_density, device, precision, bbox=box)
            deepsdf_input = torch.cat([latent.expand(grid_3d.points.size(0), -1),
                                        grid_3d.points], dim=1).to(latent.device, latent.dtype)
            pred_sdf = decoder(deepsdf_input)
            torch.cuda.synchronize()  # Ensure GPU operations complete
            decoder_time = (time.time() - start_decoder) * 1000  # Convert to ms

            # Mesh extraction timing
            start_mesh = time.time()
            mesh = None
            mesh_h_mm = 0.0
            mesh_w_mm = 0.0
            try:
                mesh = sdf2mesh_cuda(
                    pred_sdf,
                    grid_3d.points,
                    t=threshold
                )
                if mesh is not None:
                    volume = mesh_volume_ml(mesh) / 1e6
                    _verts = np.asarray(mesh.vertices)
                    if len(_verts) > 3:
                        _ext = (_verts.max(0) - _verts.min(0)) * 1000.0  # mm
                        # Z-up convention: height=z extent, width=max(x,y) extent
                        mesh_h_mm = float(_ext[2])
                        mesh_w_mm = float(max(_ext[0], _ext[1]))
            except Exception as ex:
                print('Failed to create mesh:', ex)
            mesh_time = (time.time() - start_mesh) * 1000  # Convert to ms

            inference_time = time.time() - start_total
            exec_time.append(inference_time)

            chamfer_distance = 0
            prec = rec = f1 = 0
            prec_5mm = rec_5mm = f1_5mm = 0
            # Decide whether to compute expensive metrics for this frame
            _frame_date = _date_re.match(item['frame_id'][0])
            _frame_date_str = _frame_date.group(1) if _frame_date else ""
            _fid_cur = item['fruit_id'][0]
            _is_latest = (not _skip_metrics_intermediate) or (_frame_date_str == _latest_date_per_fruit.get(_fid_cur, _frame_date_str))
            if mesh is not None and _is_latest:
                cd.reset()
                cd.update(gt, mesh)
                chamfer_distance = cd.compute(print_output=False)

                pr.reset()
                pr.update(gt, mesh)
                prec, rec, f1 = pr.compute_auc()
                prec_5mm, rec_5mm, f1_5mm, _ = pr.compute_at_threshold(0.005, print_output=False)
            elif mesh is not None and not _is_latest:
                # Intermediate frame: keep mesh_volume computation already done above; leave metrics as 0
                pass

                # Include fruit_id to avoid overwriting across samples
                fruit_id = item['fruit_id'][0]
                frame_id = item['frame_id'][0]
                unique_id = f"{fruit_id}_{frame_id}"
                
                if save_mesh:
                    mesh_filename = mesh_dir / f"{unique_id}_mesh.{save_mesh_format}"
                    try:
                        mesh.compute_vertex_normals()
                        o3d.io.write_triangle_mesh(str(mesh_filename), mesh)
                    except Exception as ex:
                        print('Failed to save mesh:', ex)

                if save_pcd or save_colored_mesh:
                    sampled = sample_mesh(mesh, pcd_num_points)
                    if sampled is not None:
                        if save_pcd:
                            pcd_filename = pcd_dir / f"{unique_id}_pcd.ply"
                            try:
                                o3d.io.write_point_cloud(str(pcd_filename), sampled)
                            except Exception as ex:
                                print('Failed to save point cloud:', ex)
                        if save_colored_mesh:
                            # Get colormap from config or use default
                            cmap_name = param.get('visualization_colormap', 'jet')
                            max_dist = param.get('visualization_max_distance', None)
                            
                            # Reuse the already-loaded and normalized GT point cloud
                            colored, distances = colorize_point_cloud(
                                sampled, gt, 
                                cmap_name=cmap_name,
                                max_distance=max_dist,
                                return_distances=True
                            )
                            if colored is not None:
                                colored_filename = colored_pcd_dir / f"{unique_id}_colored.ply"
                                try:
                                    o3d.io.write_point_cloud(str(colored_filename), colored)
                                    
                                    # Save distance statistics
                                    if distances is not None:
                                        stats_filename = colored_pcd_dir / f"{unique_id}_distance_stats.txt"
                                        with open(stats_filename, 'w') as f:
                                            f.write(f"Distance Statistics (meters)\n")
                                            f.write(f"============================\n")
                                            f.write(f"Min:    {distances.min():.6f}\n")
                                            f.write(f"Max:    {distances.max():.6f}\n")
                                            f.write(f"Mean:   {distances.mean():.6f}\n")
                                            f.write(f"Median: {np.median(distances):.6f}\n")
                                            f.write(f"Std:    {distances.std():.6f}\n")
                                            f.write(f"\nPercentiles:\n")
                                            f.write(f"  10%:  {np.percentile(distances, 10):.6f}\n")
                                            f.write(f"  25%:  {np.percentile(distances, 25):.6f}\n")
                                            f.write(f"  50%:  {np.percentile(distances, 50):.6f}\n")
                                            f.write(f"  75%:  {np.percentile(distances, 75):.6f}\n")
                                            f.write(f"  90%:  {np.percentile(distances, 90):.6f}\n")
                                            f.write(f"  95%:  {np.percentile(distances, 95):.6f}\n")
                                            f.write(f"  99%:  {np.percentile(distances, 99):.6f}\n")
                                except Exception as ex:
                                    print('Failed to save colored point cloud:', ex)

            # Safe lookups from ground truth CSV
            label = item['fruit_id'][0]
            def safe_lookup(column, default_value=""):
                try:
                    vals = df.loc[df['label'] == label, column].values
                    return vals[0] if len(vals) > 0 else default_value
                except Exception:
                    return default_value

            _lat_vec = latent.detach().to('cpu').squeeze().float()
            # Per-sample MLP-head volume (when vol_head was trained). Independent of decoder mesh.
            _mlp_vol_for_row = None
            if mlp_vol_ml is not None and len(mlp_vol_ml) > 0:
                _mlp_vol_for_row = round(float(mlp_vol_ml[0]), 1)
            cur_data = {
                'kmelon_id': label,
                'frame_id': item['frame_id'][0],
                'vertical_pos': -1,
                'cultivar': safe_lookup('cultivar', ''),
                'weight_g': safe_lookup('weight_g_inctack', ''),
                'gt_volume_ml': safe_lookup('volume_ml', ''),
                'sfm_volume_ml': safe_lookup('volume_metashape', ''),
                'mesh_volume_ml': round(volume * 1e6, 1),
                'mlp_vol_ml': _mlp_vol_for_row,
                'mesh_height_mm': round(mesh_h_mm, 2),
                'mesh_width_mm': round(mesh_w_mm, 2),
                'chamfer_distance': round(chamfer_distance, 6),
                'auc_precision': round(prec, 4),
                'auc_recall': round(rec, 4),
                'auc_f1': round(f1, 4),
                'precision_5mm': round(prec_5mm, 4),
                'recall_5mm': round(rec_5mm, 4),
                'f1_5mm': round(f1_5mm, 4),
                'latent_norm': round(float(torch.linalg.norm(_lat_vec).item()), 6),
                'latent_mean': round(float(_lat_vec.mean().item()), 6),
                'latent_std': round(float(_lat_vec.std(unbiased=False).item()), 6),
                'encoder_time_ms': round(encoder_time, 2),
                'decoder_time_ms': round(decoder_time, 2),
                'mesh_extraction_time_ms': round(mesh_time, 2),
                'total_time_ms': round(inference_time * 1000, 2)
                }
            _rows.append(cur_data)
            # Periodic crash-safety dump every 100 iters (cheap because it's a one-shot DataFrame build)
            if (n_iter + 1) % 100 == 0:
                pd.DataFrame(_rows, columns=columns).to_csv(results_csv_path, mode='w+', index=False)

        # Build final DataFrame once and persist (replaces per-iter pd.concat which was O(n^2))
        save_df = pd.DataFrame(_rows, columns=columns)
        save_df.to_csv(results_csv_path, mode='w+', index=False)

        # Calculate detailed timing statistics (all samples now have accurate timing)
        if len(save_df) > 0:
            avg_encoder = save_df['encoder_time_ms'].mean()
            std_encoder = save_df['encoder_time_ms'].std()
            avg_decoder = save_df['decoder_time_ms'].mean()
            std_decoder = save_df['decoder_time_ms'].std()
            avg_mesh = save_df['mesh_extraction_time_ms'].mean()
            std_mesh = save_df['mesh_extraction_time_ms'].std()
            avg_total = save_df['total_time_ms'].mean()
            std_total = save_df['total_time_ms'].std()
            
            print("\n" + "="*80)
            print("TIMING SUMMARY (all samples - GPU pre-warmed)")
            print("="*80)
            print(f"  Encoder:          {avg_encoder:6.2f} ± {std_encoder:5.2f} ms ({avg_encoder/avg_total*100:5.1f}%)")
            print(f"  Decoder:          {avg_decoder:6.2f} ± {std_decoder:5.2f} ms ({avg_decoder/avg_total*100:5.1f}%)")
            print(f"  Mesh Extraction:  {avg_mesh:6.2f} ± {std_mesh:5.2f} ms ({avg_mesh/avg_total*100:5.1f}%)")
            print(f"  Other (metrics):  {avg_total-avg_encoder-avg_decoder-avg_mesh:6.2f} ms ({(avg_total-avg_encoder-avg_decoder-avg_mesh)/avg_total*100:5.1f}%)")
            print(f"  ────────────────────────────────────────────────────────────")
            print(f"  Total:            {avg_total:6.2f} ± {std_total:5.2f} ms")
            print(f"  Throughput:       {1000/avg_total:6.2f} samples/second")
            print("="*80)
            
            # Show timing consistency
            max_encoder = save_df['encoder_time_ms'].max()
            min_encoder = save_df['encoder_time_ms'].min()
            print(f"\nEncoder timing range: {min_encoder:.2f} - {max_encoder:.2f} ms (variation: {max_encoder-min_encoder:.2f} ms)")
            print(f"All samples include accurate timing (GPU warm-up was performed separately)")

            # Write averaged metrics summary (JSON + TXT) using values already saved in CSV.
            try:
                summary_json_path = sweep_dir / f"shape_completion_metrics_summary{occ_suffix}.json"
                write_metrics_summary_from_results_csv(
                    results_csv_path=results_csv_path,
                    output_json_path=summary_json_path,
                    valid_mesh_filter=True,
                )
                print(f"[summary] Wrote metrics summary to: {summary_json_path}")
            except Exception as ex:
                print(f"[summary] Failed to write metrics summary: {ex}")

            # NOTE: Unified eval (last_date + intermediate split, with occlusion_pct
            # column) runs once after the entire occ sweep loop (in __main__ block),
            # not per-level — see write_combined_eval_sweep().

        print(f"\nResults saved in: {results_csv_path}")

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="shape completion main file, assume a pretrained deepsdf model")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=False,
        default=None,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well. If not provided, reads from config file.",
    )
    arg_parser.add_argument(
        "--cfg",
        "-c",
        dest="cfg",
        required=True,
        help="Config file for the outer network.",
    )
    arg_parser.add_argument(
        "--checkpoint_decoder",
        dest="checkpoint",
        default=None,
        help="The checkpoint weights to use. This should be a number indicated an epoch. If not provided, reads from config file.",
    )
    arg_parser.add_argument(
        "--checkpoint_folder",
        dest="checkpoint_folder",
        default=None,
        help="Optional path overriding checkpoint_dir from config when locating checkpoint weights",
    )
    arg_parser.add_argument(
        "--data_dir",
        dest="data_dir",
        default=None,
        help="Override data directory from config (useful for testing on different datasets like time-series data)",
    )
    arg_parser.add_argument(
        "--test_occlusion_levels",
        default=None,
        help="Comma-separated occlusion percentages (e.g. '0,20,40,60'). Overrides config.",
    )
    arg_parser.add_argument(
        "--occlusion_mode",
        default=None,
        choices=["plane", "erode", "patch"],
        help="Occlusion mode for test-time augmentation. Overrides config. Default: patch.",
    )
    arg_parser.add_argument(
        "--last_frame_only",
        action="store_true",
        help="(Legacy override) Force last-frame-only filter at ALL occlusion "
             "levels including occ=0. By default, occ=0 runs on all frames "
             "(last_date GT + intermediate corrected-ellipsoid GT) and occ>0 runs "
             "on last frames only (occlusion is meaningful only on last frame).",
    )
    arg_parser.add_argument(
        "--output_dir",
        default=None,
        help="Override the output directory. Default: <checkpoint_dir>/../.. "
             "(i.e. the experiment run root). Use this to keep occlusion sweep "
             "results in a dedicated subdir without overwriting existing CSVs.",
    )
    arg_parser.add_argument(
        "--save_latents",
        action="store_true",
        help="Dump full per-sample latent vectors (.npy) into <output_dir>/latents/. "
             "One file per (fruit_id, frame_id, occlusion_level).",
    )
    arg_parser.add_argument(
        "--min_n_points",
        type=int,
        default=0,
        help="Drop test frames whose input PCD has fewer than this many real points. "
             "Default 0 = no filter. Recommended 300 for v3+ runs.",
    )
    arg_parser.add_argument(
        "--n_points_cache",
        default=None,
        help="CSV with columns fruit_id,date,n_points. If provided and "
             "--min_n_points > 0, used to filter dataset.files. Otherwise "
             "n_points is computed by reading each PLY (slower).",
    )
    arg_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override SDF iso-surface threshold for mesh extraction. "
             "Default reads from config (typically 0.005). Use 0.0 for the true "
             "zero-level; negative values shrink predicted volumes further.",
    )
    arg_parser.add_argument(
        "--drop_first_date_per_fruit",
        action="store_true",
        help="Drop the earliest observation date for each fruit BEFORE other filters. "
             "First-date frames are typically infant fruits (very small ellipsoid, noisy).",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    # Load config to get experiment and checkpoint if not provided via command line
    import json
    with open(args.cfg) as f:
        config = json.load(f)
    
    # Use command line args if provided, otherwise fall back to config
    if args.experiment_directory is None:
        if "experiment" not in config:
            raise ValueError("--experiment not provided and 'experiment' not found in config file")
        args.experiment_directory = config["experiment"]
        print(f"[test] Using experiment from config: {args.experiment_directory}")
    
    if args.checkpoint is None:
        if "checkpoint_decoder" in config:
            args.checkpoint = str(config["checkpoint_decoder"])
            print(f"[test] Using checkpoint_decoder from config: {args.checkpoint}")
        else:
            args.checkpoint = "500"  # default fallback
            print(f"[test] Using default checkpoint_decoder: {args.checkpoint}")

    deep_sdf.configure_logging(args)

    # loading deepsdf model
    specs = ws.load_experiment_specifications(args.experiment_directory)
    latent_size = specs["CodeLength"]
    arch = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    path = os.path.join(args.experiment_directory, 'ModelParameters', args.checkpoint) + '.pth'
    model_state = net_utils.load_without_parallel(torch.load(path))
    decoder.load_state_dict(model_state)
    decoder = net_utils.set_require_grad(decoder, False)

    pretrain_path = os.path.join(args.experiment_directory, 'Reconstructions', args.checkpoint, 'Codes', 'complete')

    checkpoint_override = None
    if args.checkpoint_folder is not None:
        checkpoint_override = str(Path(args.checkpoint_folder).resolve())
    
    data_dir_override = None
    if args.data_dir is not None:
        data_dir_override = args.data_dir

    occ_levels_str = args.test_occlusion_levels or str(config.get("test_occlusion_levels", "0"))
    occlusion_levels = [int(x) for x in occ_levels_str.split(",")]
    print(f"[test] Occlusion sweep levels: {occlusion_levels}%")

    occ_mode = args.occlusion_mode or str(config.get("occlusion_mode", "patch"))
    print(f"[test] Occlusion mode: {occ_mode}")

    # Default eval policy:
    #   occ=0  -> run on ALL frames (last_date vs laser GT + intermediate vs corrected ellipsoid)
    #   occ>0  -> run on last frames only (occlusion is only meaningful on the last frame)
    # If user explicitly passes --last_frame_only, every level is forced last-only (legacy).
    for occ_pct in occlusion_levels:
        force_last = bool(args.last_frame_only)
        use_last_only = force_last or (occ_pct > 0)
        if occ_pct > 0:
            print(f"\n{'='*80}")
            print(f"[test] Running with {occ_pct}% occlusion ({occ_mode}) [last_frame_only={use_last_only}]")
            print(f"{'='*80}")
        else:
            scope = "last only" if use_last_only else "ALL FRAMES (last + intermediate)"
            print(f"\n{'='*80}")
            print(f"[test] Running occ=0 [{scope}]")
            print(f"{'='*80}")
        main_function(decoder=decoder,
                      pretrain=pretrain_path,
                      cfg=args.cfg,
                      latent_size=latent_size,
                      checkpoint_dir_override=checkpoint_override,
                      data_dir_override=data_dir_override,
                      occlusion_pct=occ_pct,
                      occlusion_mode=occ_mode,
                      last_frame_only=use_last_only,
                      output_dir_override=args.output_dir,
                      save_latents=args.save_latents,
                      min_n_points=args.min_n_points,
                      n_points_cache=args.n_points_cache,
                      threshold_override=args.threshold,
                      drop_first_date_per_fruit=args.drop_first_date_per_fruit)

    # === Sweep-level unified eval (all occ levels, last_date + intermediate combined) ===
    try:
        from corepp.scripts.eval_unified import write_combined_eval_sweep
        from pathlib import Path
        if args.output_dir is not None:
            run_root = Path(args.output_dir).resolve()
        else:
            run_root = Path(config["checkpoint_dir"]).resolve().parent.parent
        sweep_dir = run_root / "eval" / "shape_completion_results"
        eval_root = run_root / "eval"
        field_csv = config.get("field_size_csv",
            "/home/mainuser/mount/sungjay/tcore/corepp/data/kmelon_centered/field_size_data_v2.csv")
        if sweep_dir.exists() and os.path.exists(field_csv):
            write_combined_eval_sweep(
                sweep_dir=sweep_dir,
                field_csv=field_csv,
                output_dir=eval_root,
                occlusion_levels=occlusion_levels,
            )
            print(f"\n[unified-eval] Wrote sweep-level combined evaluation to {eval_root}")
    except Exception as ex:
        print(f"[unified-eval] Sweep eval skipped: {ex}")