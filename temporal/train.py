#!/usr/bin/env python3
"""Temporal training entry point for CoRe++.

This script mirrors ``train.py`` but processes complete temporal sequences.
"""
import argparse
import json
import math
import os
import random
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import shutil
import subprocess

# Add root directory to python path for imports
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(root_dir / "corepp") not in sys.path:
    sys.path.append(str(root_dir / "corepp"))

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

import deepsdf.deep_sdf as deep_sdf
import deepsdf.deep_sdf.workspace as ws

from corepp.loss import SuperLoss
from corepp.networks.temporal_models import TemporalFusionModule, TwoStreamTemporalFusionModule, CrossAttentionTemporalFusion, EncoderMemoryBuffer, LatentKalmanFusion, LatentKalmanFusionV2, LatentKalmanFusionV3
from corepp.networks.transformer_temporal import CausalTemporalTransformer
from corepp.temporal.core import build_encoder, prepare_dataset, run_encoder_forward, run_encoder_forward_with_aux, apply_plane_occlusion, apply_patch_occlusion
from corepp.dataloaders.pcd_transforms import PCDRandomRotate, PCDRandomJitter, Compose as PCDCompose
from corepp.temporal.utils import TemporalMetadataIndex
from corepp.sdfrenderer.grid import Grid3D
from corepp.utils import sdf2mesh_cuda, tensor_dict_2_float_dict



def compute_pca_trace(pcd: torch.Tensor) -> float:
    """Compute PCA eigenvalue trace of a point cloud as spatial coverage quality metric.

    PCA trace = total variance = sum of eigenvalues = how much 3D space the points cover.
    Higher trace → better spatial coverage → more informative measurement.

    .. deprecated:: Use :func:`compute_smoothness` instead. PCA trace incorrectly
       assigns high quality to noisy observations because noise increases spatial spread.
    """
    norms = pcd.norm(dim=1)
    real = pcd[norms > 1e-8]
    if len(real) < 4:  # need at least 4 points for meaningful PCA
        return 1e-8
    centered = real - real.mean(dim=0)
    cov = centered.T @ centered / len(centered)
    eigvals = torch.linalg.eigvalsh(cov)
    return eigvals.sum().item()


def compute_smoothness(pcd: torch.Tensor, k: int = 20) -> float:
    """Inverse local surface roughness — observation quality metric.

    For each point's k-NN neighborhood, computes the smallest eigenvalue of the
    local covariance matrix (= variance perpendicular to the local surface = noise).
    Returns 1 / mean_roughness, so **higher = smoother = better quality**.

    Unlike PCA trace (which *increases* with noise-induced spatial spread), this
    metric correctly penalises noisy observations: noisy points have larger
    perpendicular variance → higher roughness → lower return value.
    """
    norms = pcd.norm(dim=1)
    real = pcd[norms > 1e-8]
    n = len(real)
    if n < k + 1:
        return 1e-8
    # Pairwise distances  (n, n)
    dists = torch.cdist(real.unsqueeze(0), real.unsqueeze(0)).squeeze(0)
    # k nearest neighbours excluding self  (n, k)
    _, knn_idx = dists.topk(k + 1, dim=1, largest=False)
    knn_idx = knn_idx[:, 1:]
    neighbors = real[knn_idx]                              # (n, k, 3)
    local_means = neighbors.mean(dim=1, keepdim=True)      # (n, 1, 3)
    centered = neighbors - local_means                     # (n, k, 3)
    local_cov = centered.transpose(1, 2) @ centered / k   # (n, 3, 3)
    eigvals = torch.linalg.eigvalsh(local_cov)             # (n, 3) ascending
    roughness = eigvals[:, 0].mean().item()                # mean smallest eigenvalue
    return 1.0 / (roughness + 1e-10)


def compute_voxel_coverage(
    pcd: torch.Tensor,
    running_bbox_min: Optional[torch.Tensor],
    running_bbox_max: Optional[torch.Tensor],
    resolution: int = 15,
    percentile_lo: float = 5.0,
    percentile_hi: float = 95.0,
):
    """Voxel coverage within a causally accumulated percentile bounding box.

    Measures how much of the running-max extent is covered by the current PCD,
    making it robust to outlier points (via percentile clipping) and sensitive
    to narrow/occluded views (low coverage → low quality).

    Returns:
        (coverage, new_running_bbox_min, new_running_bbox_max)
        coverage ∈ [0, 1] — non-empty voxels / total voxels at given resolution
    """
    norms = pcd.norm(dim=1)
    real = pcd[norms > 1e-8]
    if len(real) < 10:
        return 0.0, running_bbox_min, running_bbox_max

    pts_np = real.detach().cpu().numpy()
    p_min = torch.tensor(np.percentile(pts_np, percentile_lo, axis=0), dtype=torch.float32)
    p_max = torch.tensor(np.percentile(pts_np, percentile_hi, axis=0), dtype=torch.float32)

    if running_bbox_min is None:
        new_min, new_max = p_min, p_max
    else:
        new_min = torch.minimum(running_bbox_min, p_min)
        new_max = torch.maximum(running_bbox_max, p_max)

    extent = (new_max - new_min).clamp(min=1e-6)
    idx = ((real.cpu() - new_min) / extent * resolution).long()
    valid = ((idx >= 0) & (idx < resolution)).all(dim=1)
    idx = idx[valid]
    if len(idx) == 0:
        return 0.0, new_min, new_max

    flat = idx[:, 0] * resolution ** 2 + idx[:, 1] * resolution + idx[:, 2]
    coverage = flat.unique().numel() / resolution ** 3
    return coverage, new_min, new_max


def compute_combined_quality(pcd: torch.Tensor,
                             running_max_smooth: float,
                             running_max_pca: float,
                             k: int = 20):
    """Geometric mean of smoothness and PCA-trace quality metrics.

    Combines two complementary signals:
    - Smoothness (inverse roughness): penalises *noisy* observations.
      Noise increases perpendicular variance → higher roughness → lower quality.
    - PCA trace (spatial coverage): penalises *occluded* observations.
      Occlusion reduces the observed surface area → smaller trace → lower quality.

    Each is normalised by its causal within-sequence running maximum so both
    live in (0, 1].  The geometric mean requires *both* to be high for q_t to
    be high, making the model robust to noise AND occlusion simultaneously.

    Returns:
        (q_t, new_running_max_smooth, new_running_max_pca)
    """
    smooth_t = compute_smoothness(pcd, k=k)
    pca_t    = compute_pca_trace(pcd)

    new_max_smooth = max(running_max_smooth, smooth_t)
    new_max_pca    = max(running_max_pca,    pca_t)

    q_smooth = smooth_t / new_max_smooth if new_max_smooth > 1e-8 else 1.0
    q_pca    = pca_t    / new_max_pca    if new_max_pca    > 1e-8 else 1.0

    q_t = (q_smooth * q_pca) ** 0.5   # geometric mean in (0, 1]
    return q_t, new_max_smooth, new_max_pca


def compute_ang_az_entropy(pcd: torch.Tensor, n_az: int = 12) -> float:
    """Azimuth-direction occupancy entropy of a point cloud (scaled to [0, 1]).

    Splits the azimuth around the centroid into ``n_az`` bins, computes the
    Shannon entropy of the bin occupancy histogram, normalised by ``log(n_az)``
    so that 1.0 = perfectly uniform azimuth coverage and 0.0 = all points in a
    single bin.  Higher entropy = wider angular coverage = better observation.

    Returns the raw entropy in [0, 1] — no running-max normalisation needed.
    """
    norms = pcd.norm(dim=1)
    real = pcd[norms > 1e-8]
    if len(real) < 4:
        return 0.0
    centroid = real.mean(dim=0)
    centered = real - centroid
    az = torch.atan2(centered[:, 1], centered[:, 0])
    az_bin = ((az + math.pi) / (2 * math.pi) * n_az).long().clamp(0, n_az - 1)
    counts = torch.bincount(az_bin, minlength=n_az).float()
    p = counts / counts.sum().clamp(min=1.0)
    eps = 1e-12
    entropy = -(p * (p + eps).log()).sum().item()
    return float(entropy / math.log(n_az))


def compute_point_to_plane_residual(pcd: torch.Tensor, k: int = 20) -> float:
    """Mean point-to-plane residual (same units as pcd).

    For each point, fits a local plane to its k-NN neighbourhood via PCA
    (smallest eigenvector = surface normal), then computes the perpendicular
    distance from the point to that plane.  Returns the mean over all points.

    Caller is responsible for normalising (e.g. dividing by Z_m in metres to
    obtain a dimensionless q_p2plane = 1 - clip(mean_residual / Z_m, 0, 1)).
    """
    norms = pcd.norm(dim=1)
    real  = pcd[norms > 1e-8]
    n = len(real)
    if n < k + 1:
        return 0.0
    dists   = torch.cdist(real.unsqueeze(0), real.unsqueeze(0)).squeeze(0)
    _, idx  = dists.topk(k + 1, dim=1, largest=False)
    idx     = idx[:, 1:]
    neighbors  = real[idx]                                   # (n, k, 3)
    local_mean = neighbors.mean(dim=1, keepdim=True)         # (n, 1, 3)
    centered   = neighbors - local_mean                      # (n, k, 3)
    local_cov  = centered.transpose(1, 2) @ centered / k    # (n, 3, 3)
    _, eigvecs = torch.linalg.eigh(local_cov)                # ascending order
    normals    = eigvecs[:, :, 0]                            # (n, 3)
    diff       = real - local_mean.squeeze(1)                # (n, 3)
    residuals  = (diff * normals).sum(dim=1).abs()           # (n,)
    return float(residuals.mean().item())


class TeeIO:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        self.flush()

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

def chamfer_distance_torch(pred_pcd, target_pcd, batch_size=1024):
    """Compute chamfer distance between two point clouds with batched processing."""
    max_points = 5000
    if pred_pcd.shape[0] > max_points:
        indices = torch.randperm(pred_pcd.shape[0])[:max_points]
        pred_pcd = pred_pcd[indices]
    if target_pcd.shape[0] > max_points:
        indices = torch.randperm(target_pcd.shape[0])[:max_points]
        target_pcd = target_pcd[indices]

    min_dists_p2t = []
    for i in range(0, pred_pcd.shape[0], batch_size):
        batch = pred_pcd[i:i+batch_size].unsqueeze(1)
        dist = torch.sum((batch - target_pcd.unsqueeze(0)) ** 2, dim=2)
        min_dists_p2t.append(torch.min(dist, dim=1)[0])
    p2t = torch.cat(min_dists_p2t).mean()

    min_dists_t2p = []
    for i in range(0, target_pcd.shape[0], batch_size):
        batch = target_pcd[i:i+batch_size].unsqueeze(1)
        dist = torch.sum((batch - pred_pcd.unsqueeze(0)) ** 2, dim=2)
        min_dists_t2p.append(torch.min(dist, dim=1)[0])
    t2p = torch.cat(min_dists_t2p).mean()

    return torch.sqrt((p2t + t2p) / 2.0)


def compute_temporal_triplet_loss(hidden_states, target_latent, margin=0.1):
    """
    hidden_states: list of (B, D) tensors ordered by time.
    target_latent: (B, D) tensor, the final goal.
    Enforces that frame t_B (later) is closer to target than frame t_A (earlier).
    """
    loss = torch.tensor(0.0, device=target_latent.device)
    num_pairs = 0
    n = len(hidden_states)
    if n < 2:
        return loss

    # Sample random pairs or all pairs? For small sequences, all pairs is fine.
    # But let's just do adjacent pairs + random pairs to keep it O(N).
    # Actually, N is small (~3 to ~20). O(N^2) is negligible.
    for i in range(n):
        for j in range(i + 1, n):
            # i is earlier (t_A), j is later (t_B)
            # anchor = target
            # positive = h_j (should be closer)
            # negative = h_i (should be further)
            
            dist_pos = torch.norm(hidden_states[j] - target_latent, p=2, dim=1)
            dist_neg = torch.norm(hidden_states[i] - target_latent, p=2, dim=1)
            
            # Loss = ReLU(dist_pos - dist_neg + margin)
            triplet_loss = torch.relu(dist_pos - dist_neg + margin).mean()
            loss += triplet_loss
            num_pairs += 1
            
    if num_pairs > 0:
        loss /= num_pairs
    return loss

def _seed_everything(seed: int = 1337) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    cudnn.enabled = False


def _maybe_launch_tensorboard(log_dir: Path, port: Optional[int]):
    if port is None:
        return None
    if shutil.which("tensorboard") is None:
        print("[temporal-train] TensorBoard executable not found in PATH; skipping auto-launch.")
        return None
    cmd = [
        "tensorboard",
        f"--logdir={log_dir}",
        f"--port={port}",
        "--host=0.0.0.0",
    ]
    print(f"[temporal-train] Launching TensorBoard on port {port} (logdir={log_dir})")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return proc
    except Exception as exc:
        print(f"[temporal-train] Failed to launch TensorBoard: {exc}")
        return None


def _build_run_dirs(param: Dict, cfg_path: str) -> Tuple[Path, Path, Path]:
    cfg_fname = Path(cfg_path).stem
    task_name = str(param.get("task_name", param.get("encoder", "temporal_task")))
    original_checkpoint_dir = Path(param["checkpoint_dir"]).resolve()

    def derive_base_dir(path: Path) -> Path:
        base = path
        while base.name in {"checkpoints", "runs", "logs"}:
            base = base.parent
        if base.name.lower() == "train":
            return base.parent
        if base.parent and base.parent.name.lower() == "train":
            return base.parent.parent
        return base

    base_dir = derive_base_dir(original_checkpoint_dir)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_timestamp}_{task_name}"
    run_root = base_dir / run_name
    checkpoints_dir = run_root / "logs" / "checkpoints"
    runs_dir = run_root / "logs" / "runs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    param["checkpoint_dir"] = str(checkpoints_dir)
    param["log_dir"] = str(runs_dir)
    param["run_root"] = str(run_root)
    param["checkpoint_file"] = f"{cfg_fname}_best_model.pt"
    config_save_path = run_root / f"{cfg_fname}.json"
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(param, f, indent=4)
    print(f"[temporal-train] Config saved to {config_save_path}")
    return run_root, checkpoints_dir, runs_dir


def train_temporal(args, param: Dict, latent_size: int, decoder: nn.Module) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_root, checkpoints_dir, runs_dir = _build_run_dirs(param, args.cfg)

    # Setup Logging to log.txt
    log_txt_path = run_root / "logs" / "log.txt"
    log_fh = open(log_txt_path, "a", buffering=1)
    sys.stdout = TeeIO(sys.stdout, log_fh)
    sys.stderr = TeeIO(sys.stderr, log_fh)
    print(f"[temporal-train] Logging to {log_txt_path}")

    is_two_stream = param.get("two_stream", False)
    pca_vectors_path = param.get("pca_vectors", None)
    
    if pca_vectors_path and os.path.exists(pca_vectors_path):
        print(f"[temporal-train] Loading PCA vectors from {pca_vectors_path}")
        # Expecting directory containing pca_components.npy and pca_mean.npy
        pca_comp = np.load(os.path.join(pca_vectors_path, "pca_components.npy"))
        pca_mean = np.load(os.path.join(pca_vectors_path, "pca_mean.npy"))
        
        # Convert to torch
        pca_comp_t = torch.tensor(pca_comp, device=device)
        pca_mean_t = torch.tensor(pca_mean, device=device)
        
        from corepp.networks.temporal_models import PCATemporalFusionModule
        fusion = PCATemporalFusionModule(
             pca_components=pca_comp_t,
             pca_mean=pca_mean_t,
             dynamic_dim=int(param.get("pca_dynamic_dim", 16)),
             hidden_dim=int(param.get("fusion_hidden_size", latent_size)),
             dropout=float(param.get("fusion_dropout", 0.0)),
             use_delta=True
        ).to(device)
        print(f"[temporal-train] Using PCATemporalFusionModule (Dynamic K={fusion.dynamic_dim})")
        
    elif is_two_stream:
        fusion = TwoStreamTemporalFusionModule(
            shape_dim=int(param.get("shape_dim", 32)),
            size_dim=int(param.get("size_dim", 3)),
            hidden_dim=int(param.get("fusion_hidden_size", latent_size)),
            use_delta=True,
        ).to(device)
        print("[temporal-train] using TwoStreamTemporalFusionModule")
    elif param.get("temporal_arch") == "transformer":
        fusion = CausalTemporalTransformer(
            latent_dim=latent_size,
            hidden_dim=int(param.get("fusion_hidden_size", 128)),
            num_layers=int(param.get("transformer_layers", 2)),
            num_heads=int(param.get("transformer_heads", 4)),
            dropout=float(param.get("fusion_dropout", 0.1)),
            zero_init=bool(param.get("zero_init", True))
        ).to(device)
        print("[temporal-train] using CausalTemporalTransformer")
    elif param.get("temporal_arch") == "cross_attention":
        fusion = CrossAttentionTemporalFusion(
            latent_dim=latent_size,
            num_heads=int(param.get("cross_attn_heads", 1)),
            dropout=float(param.get("fusion_dropout", 0.0)),
            use_time_embedding=bool(param.get("cross_attn_time_embedding", True)),
        ).to(device)
        print("[temporal-train] using CrossAttentionTemporalFusion")
    elif param.get("temporal_arch") == "kalman":
        quality_mode = param.get("quality_mode", "normalized")
        init_R = float(param.get("kalman_init_R", 0.1))
        growth_mode = param.get("growth_mode", "mlp")
        robust_gating = bool(param.get("robust_gating", False))
        mahal_threshold = float(param.get("mahal_threshold", 53.0))
        warmup_frames = int(param.get("warmup_frames", 3))
        gate_sharpness = float(param.get("gate_sharpness", 2.0))
        gate_R_floor = float(param.get("gate_R_floor", 0.01))
        anisotropic_gating = bool(param.get("anisotropic_gating", False))
        aniso_npz_path = param.get("aniso_npz_path", None)
        aniso_K = int(param.get("aniso_K", 2))
        aniso_threshold_kalman = float(param.get("aniso_threshold_kalman", 50.0))
        aniso_threshold_anchor = float(param.get("aniso_threshold_anchor", 50.0))
        aniso_use_anchor = bool(param.get("aniso_use_anchor", True))
        aniso_use_size_gate = bool(param.get("aniso_use_size_gate", False))
        aniso_threshold_size_kalman = float(param.get("aniso_threshold_size_kalman", 4.4))  # used for DOWNward direction
        _v = param.get("aniso_threshold_size_kalman_up", None)
        aniso_threshold_size_kalman_up = float(_v) if _v is not None else None  # if None → symmetric
        aniso_use_dz_floor = bool(param.get("aniso_use_dz_floor", False))
        aniso_dz_floor = float(param.get("aniso_dz_floor", -0.1))
        aniso_dz_floor_gate_value = float(param.get("aniso_dz_floor_gate_value", 0.05))
        fusion = LatentKalmanFusion(latent_dim=latent_size, quality_mode=quality_mode,
                                    init_R=init_R, growth_mode=growth_mode,
                                    robust_gating=robust_gating,
                                    mahal_threshold=mahal_threshold,
                                    warmup_frames=warmup_frames,
                                    gate_sharpness=gate_sharpness,
                                    gate_R_floor=gate_R_floor,
                                    anisotropic_gating=anisotropic_gating,
                                    aniso_npz_path=aniso_npz_path,
                                    aniso_K=aniso_K,
                                    aniso_threshold_kalman=aniso_threshold_kalman,
                                    aniso_threshold_anchor=aniso_threshold_anchor,
                                    aniso_use_anchor=aniso_use_anchor,
                                    aniso_use_size_gate=aniso_use_size_gate,
                                    aniso_threshold_size_kalman=aniso_threshold_size_kalman,
                                    aniso_threshold_size_kalman_up=aniso_threshold_size_kalman_up,
                                    aniso_use_dz_floor=aniso_use_dz_floor,
                                    aniso_dz_floor=aniso_dz_floor,
                                    aniso_dz_floor_gate_value=aniso_dz_floor_gate_value).to(device)
        print(f"[temporal-train] using LatentKalmanFusion (quality_mode={quality_mode}, init_R={init_R}, growth_mode={growth_mode}, "
              f"robust_gating={robust_gating}, mahal_threshold={mahal_threshold}, "
              f"anisotropic_gating={anisotropic_gating} K={aniso_K} thr_kalman={aniso_threshold_kalman} use_anchor={aniso_use_anchor} "
              f"size_gate={aniso_use_size_gate} thr_size_dn={aniso_threshold_size_kalman} thr_size_up={aniso_threshold_size_kalman_up} dz_floor={aniso_use_dz_floor}/{aniso_dz_floor})")
    elif param.get("temporal_arch") == "kalman_v2":
        init_R = float(param.get("kalman_init_R", 0.01))
        fusion = LatentKalmanFusionV2(
            latent_dim=latent_size, init_R=init_R,
        ).to(device)
        print(f"[temporal-train] using LatentKalmanFusionV2 (init_R={init_R})")
    elif param.get("temporal_arch") == "kalman_v3":
        init_R = float(param.get("kalman_init_R", 0.01))
        fusion = LatentKalmanFusionV3(
            latent_dim=latent_size, init_R=init_R,
        ).to(device)
        print(f"[temporal-train] using LatentKalmanFusionV3 (init_R={init_R})")
    else:
        fusion = TemporalFusionModule(
            latent_dim=latent_size,
            hidden_dim=int(param.get("fusion_hidden_size", latent_size)),
            dropout=float(param.get("fusion_dropout", 0.0)),
            use_delta=True,
        ).to(device)

    encoder = build_encoder(param, latent_size, device)
    pretrained_encoder_path = param.get("pretrained_encoder")
    if pretrained_encoder_path and os.path.isfile(pretrained_encoder_path):
        checkpoint = torch.load(pretrained_encoder_path, map_location=device)
        state_dict = (
            checkpoint.get("encoder_state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint
        )
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        print(f"[temporal-train] Loaded encoder from {pretrained_encoder_path}")
        if missing or unexpected:
            print(f"[temporal-train]   missing={len(missing)}, unexpected={len(unexpected)} (e.g. vol_head)")

    tb_process = _maybe_launch_tensorboard(run_root / "logs", getattr(args, "tensorboard_port", None))
    resume_path = getattr(args, "resume", None)
    start_epoch = 1
    total_epochs_config = int(param.get("epoch", 1))
    global_step = 0
    best_loss = float("inf")
    best_metric_val = float("inf")
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")

    writer = SummaryWriter(filename_suffix="__temporal", log_dir=str(runs_dir))
    try:
        dataset = prepare_dataset(param, pretrain=param["pretrain_path"], split="train")
        metadata = TemporalMetadataIndex(
            dataset.files,
            timestamp_csv=param.get("timestamp_csv"),
            min_sequence_length=int(param.get("min_sequence_length", 2)),
            orbbec_suffix=param.get("orbbec_pcd_suffix", "_preprocessed"),
        )

        if not metadata.sequences:
            raise RuntimeError("No temporal sequences found. Check timestamp CSV and split.json")

        # Optional: bin-uniform + size-inverse weighted sequence sampling (training only)
        _bin_uniform_temporal = bool(param.get("bin_uniform_sampling", False))
        if _bin_uniform_temporal:
            from pathlib import Path as _Path
            _vol_csv_rel = param.get("vol_map_csv", "field_size_data_v2.csv")
            _vol_csv = _vol_csv_rel if os.path.isabs(_vol_csv_rel) else str(_Path(param["data_dir"]) / _vol_csv_rel)
            metadata.configure_balanced(
                vol_map_csv=_vol_csv,
                bin_edges=param.get("bin_edges", [0, 100, 200, 300, 400, 500, 1000]),
                target_count_per_bin=int(param.get("bin_target_count", 15)),
                size_inverse_alpha=float(param.get("size_inverse_alpha", 1.0)),
                size_inverse_vol_ref=float(param.get("size_inverse_vol_ref", 300.0)),
            )
            print(f"[temporal-train] bin_uniform_sampling ENABLED for training (validation uses unbalanced).")

        # Caliper weak supervision per-frame volume loss. Two modes:
        #   "mlp"     — small head (latent → log_vol), simpler/faster, learned mapping.
        #   "softvol" — decode fused latent on bbox grid + sigmoid soft volume integration,
        #               principled (uses frozen DeepSDF decoder geometry), no extra params.
        vol_head_temporal = None
        _caliper_weak_temporal = bool(param.get("caliper_weak_supervision", False))
        _weak_vol_loss_mode = str(param.get("weak_vol_loss_mode", "mlp")).lower()
        df = pd.DataFrame()
        if _caliper_weak_temporal:
            if _weak_vol_loss_mode == "mlp":
                vol_head_temporal = nn.Sequential(
                    nn.Linear(latent_size, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 1),
                ).to(device)
                print(f"[temporal-train] vol_head_temporal MLP initialized (latent_dim={latent_size} → log_vol). "
                      f"lambda_volume_weak={param.get('lambda_volume_weak', 1.0)}, "
                      f"lambda_volume_strong={param.get('lambda_volume_strong', 1.0)}")
            elif _weak_vol_loss_mode == "softvol":
                print(f"[temporal-train] weak_vol_loss_mode=softvol (decoder forward + sigmoid soft volume). "
                      f"softvol_temperature={param.get('softvol_temperature', 0.005)}, "
                      f"lambda_volume_weak={param.get('lambda_volume_weak', 1.0)}, "
                      f"lambda_volume_strong={param.get('lambda_volume_strong', 1.0)}")
            else:
                raise ValueError(f"weak_vol_loss_mode must be 'mlp' or 'softvol', got: {_weak_vol_loss_mode}")
            # Load laser GT volumes for last-frame strong supervision
            _gt_csv = param.get("gt_volume_csv") or param.get("val_gt_csv")
            if _gt_csv and os.path.exists(_gt_csv):
                df = pd.read_csv(_gt_csv)
                print(f"[temporal-train] Loaded GT volume CSV for last-frame strong vol loss: {_gt_csv} ({len(df)} rows)")

        if param.get("freeze_encoder", False):
            print("[temporal-train] Encoder is FROZEN (weights fixed). Training fusion module only.")
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.eval()  # freeze BN running stats (requires_grad=False alone does not stop BN updates)
            params_to_train = list(fusion.parameters())
            if vol_head_temporal is not None:
                params_to_train += list(vol_head_temporal.parameters())
            optimizer = torch.optim.Adam(
                params_to_train,
                lr=param["lr"],
                weight_decay=1e-6,
            )
        else:
            lr_encoder = param.get("lr_encoder", param["lr"])
            lr_fusion = param["lr"]
            print(f"[temporal-train] Encoder UNFROZEN. lr_encoder={lr_encoder}, lr_fusion={lr_fusion}")
            params_to_train = list(encoder.parameters()) + list(fusion.parameters())
            if vol_head_temporal is not None:
                params_to_train += list(vol_head_temporal.parameters())
            optimizer = torch.optim.Adam([
                {"params": list(encoder.parameters()), "lr": lr_encoder},
                {"params": list(fusion.parameters()) + (list(vol_head_temporal.parameters()) if vol_head_temporal is not None else []), "lr": lr_fusion},
            ], weight_decay=1e-6)

        # Gradient Accumulation Setup
        # We use 'batch_size' from config as the number of sequences to accumulate
        accumulation_steps = int(param.get("batch_size", 1))
        if accumulation_steps < 1: accumulation_steps = 1
        print(f"[temporal-train] Gradient Accumulation Steps: {accumulation_steps} (Simulating Batch Size)")


        if resume_path:
            print(f"[temporal-train] Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            if "encoder_state_dict" in checkpoint:
                encoder.load_state_dict(checkpoint["encoder_state_dict"])
            if "fusion_state_dict" in checkpoint:
                fusion.load_state_dict(checkpoint["fusion_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            best_loss = checkpoint.get("best_loss", best_loss)
            start_epoch = checkpoint.get("epoch", 0) + 1
            global_step = checkpoint.get("global_step", 0)
            print(f"[temporal-train] Resumed at epoch {start_epoch}, best_loss={best_loss:.6f}, global_step={global_step}")

        if resume_path:
            end_epoch = start_epoch + total_epochs_config - 1
            print(f"[temporal-train] Will train for {end_epoch - start_epoch + 1} additional epoch(s) (until epoch {end_epoch}).")
        else:
            end_epoch = total_epochs_config

        # LR Scheduler: optional warmup (LinearLR) + main decay (StepLR or CosineAnnealingLR).
        # Selected via lr_scheduler config: 'step' (default) or 'cosine'.
        warmup_epochs = int(param.get("warmup_epochs", 0))
        lr_sched_kind = str(param.get("lr_scheduler", "step")).lower()
        lr_step = int(param.get("lr_step", 0))
        lr_gamma = float(param.get("lr_gamma", 0.5))

        schedulers = []
        milestones = []

        if warmup_epochs > 0:
            warmup_start_factor = float(param.get("warmup_start_factor", 0.01))
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            schedulers.append(warmup_sched)
            milestones.append(warmup_epochs)
            print(f"[temporal-train] LR Warmup: {warmup_epochs} epochs, "
                  f"start_factor={warmup_start_factor} "
                  f"(LR {param['lr'] * warmup_start_factor:.6f} → {param['lr']:.6f})")

        if lr_sched_kind == "cosine":
            t_max = int(param.get("lr_T_max", end_epoch - warmup_epochs))
            eta_min = float(param.get("lr_eta_min", 1e-6))
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, t_max),
                eta_min=eta_min,
            )
            schedulers.append(cosine_sched)
            print(f"[temporal-train] LR CosineAnnealingLR: T_max={t_max}, eta_min={eta_min} "
                  f"(LR {param['lr']:.6f} → {eta_min:.6f})")
        elif lr_step > 0:
            step_sched = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=lr_step,
                gamma=lr_gamma,
            )
            schedulers.append(step_sched)
            print(f"[temporal-train] LR StepLR: step_size={lr_step}, gamma={lr_gamma}")

        if len(schedulers) > 1:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=schedulers, milestones=milestones
            )
        elif len(schedulers) == 1:
            scheduler = schedulers[0]
        else:
            scheduler = None

        # If resuming, fast-forward scheduler to current epoch
        if scheduler is not None and resume_path and start_epoch > 1:
            for _ in range(start_epoch - 1):
                scheduler.step()

        delta_unit = float(param.get("delta_time_unit", 1.0))
        reset_on_gap = bool(param.get("reset_on_gap", False))
        raw_gap = param.get("max_gap")
        # max_gap should be in the same units as delta_unit (e.g. days if delta_unit is 86400)
        # We assume max_gap in config is already in those units (e.g. 10 days)
        max_gap = float(raw_gap) if raw_gap is not None else None
        checkpoint_every = int(param.get("checkpoint_frequency", 5))

        # PCD augmentation transforms (config-gated, default on for backward compat)
        use_pcd_aug = bool(param.get("use_pcd_aug", True))
        if use_pcd_aug:
            pcd_aug = PCDCompose([
                PCDRandomRotate(angle_range=180),          # full ±180° Y-axis rotation
                PCDRandomJitter(sigma=0.005, clip=0.02),   # Gaussian noise σ=0.005
            ])
            print("[temporal-train] PCD augmentation: RandomRotate(±180°) + RandomJitter(σ=0.005, clip=0.02)")
        else:
            pcd_aug = None
            print("[temporal-train] PCD augmentation DISABLED (use_pcd_aug=False)")

        for epoch in range(start_epoch, end_epoch + 1):
            epoch_loss = 0.0
            seq_count = 0
            if _bin_uniform_temporal:
                epoch_sequences = list(metadata.iter_sequences_balanced(shuffle=True))
            else:
                epoch_sequences = list(metadata.iter_sequences(shuffle=True))
            progress = tqdm(
                epoch_sequences,
                desc=f"Epoch {epoch}/{end_epoch}",
                leave=False,
                ncols=100,
            )

            optimizer.zero_grad() # Zero gradients at start of epoch
            
            use_cross_attn = isinstance(fusion, CrossAttentionTemporalFusion)
            use_kalman = isinstance(fusion, (LatentKalmanFusion, LatentKalmanFusionV2, LatentKalmanFusionV3))

            for seq_idx, sequence in enumerate(progress):
                # Gradient Accumulation: Don't zero grad here!
                # We zero grad ONLY after stepping.

                hidden = None
                prev_ts = None
                seq_loss = 0.0
                gate_mean = 0.0
                steps = 0
                collected_hiddens = []
                encoder_latents_seq = []
                encoder_times_seq = [] # Collect normalized times
                last_target = None
                start_ts = None # To normalize time relative to start of sequence
                running_max_smooth = 0.0  # for Kalman quality: within-sequence running max of smoothness
                running_max_pca    = 0.0  # for combined quality: within-sequence running max of PCA trace
                running_bbox_min_pct: Optional[torch.Tensor] = None  # for voxel coverage
                running_bbox_max_pct: Optional[torch.Tensor] = None
                running_max_vox = 0.0  # for voxel coverage running-max normalization

                # Cross-attention memory buffer (reset per sequence)
                if use_cross_attn:
                    mem_buffer = EncoderMemoryBuffer(max_len=int(param.get("max_memory_len", 20)))
                    seq_fruit_id = "seq"
                cumulative_time = 0.0
                
                # Frame Dropping Augmentation: randomly drop intermediate frames
                # to simulate irregular/sparse observation schedules.
                drop_prob = float(param.get("drop_frame_prob", 0.0))
                min_keep = int(param.get("drop_frame_min_keep", 5))
                if len(sequence) > min_keep and random.random() < drop_prob:
                    keep_count = random.randint(min_keep, len(sequence) - 1)
                    # Always keep first and last frame; sample the rest
                    middle_indices = sorted(random.sample(range(1, len(sequence) - 1), keep_count - 2))
                    indices = [0] + middle_indices + [len(sequence) - 1]
                    sequence = [sequence[i] for i in indices]

                should_supervise_all = not param.get("loss_final_frame_only", False)
                use_occlusion = param.get("use_occlusion_augmentation", False)
                max_occ = float(param.get("max_occlusion_frac", 0.3))
                occ_min_pts = int(param.get("occlusion_min_points", 64))
                # P(apply occlusion) per frame: default 0.4 → 60% frames stay clean
                occ_apply_prob = float(param.get("occlusion_apply_prob", 0.4))

                for i, record in enumerate(sequence):
                    is_last_frame = (i == len(sequence) - 1)

                    item = dataset[record.dataset_index]

                    # Patch occlusion augmentation — all frames, applied with occ_apply_prob
                    if use_occlusion and "partial_pcd" in item and random.random() < occ_apply_prob:
                        pcd = item["partial_pcd"]
                        had_batch = pcd.dim() == 3
                        if had_batch:
                            pcd = pcd[0]
                        pcd = apply_patch_occlusion(
                            pcd,
                            max_occlusion=max_occ,
                            min_points=occ_min_pts,
                        )
                        if had_batch:
                            pcd = pcd.unsqueeze(0)
                        item["partial_pcd"] = pcd

                    # PCD augmentation: config-gated (use_pcd_aug, default True)
                    if pcd_aug is not None and "partial_pcd" in item:
                        pcd = item["partial_pcd"]
                        had_batch = pcd.dim() == 3
                        if had_batch:
                            pcd = pcd[0]
                        pcd = pcd_aug(pcd)
                        if had_batch:
                            pcd = pcd.unsqueeze(0)
                        item["partial_pcd"] = pcd
                    
                    # If the encoder has an NLL log_scale head, extract it for use as Kalman R_t.
                    use_learned_R = bool(param.get("use_learned_R_from_encoder", False))
                    learned_R_t = None
                    if use_learned_R and hasattr(encoder, "encode_with_aux"):
                        aux_out = run_encoder_forward_with_aux(encoder, item, param, device)
                        encoder_latent = aux_out["latent"]
                        if "log_scale" in aux_out:
                            ls = aux_out["log_scale"]
                            # Clamp consistent with training (default ±3) → R = exp(2·ls) (variance)
                            ls_clamp_min = float(param.get("nll_log_scale_min", -3.0))
                            ls_clamp_max = float(param.get("nll_log_scale_max", 3.0))
                            ls = ls.clamp(ls_clamp_min, ls_clamp_max)
                            learned_R_t = (2.0 * ls).exp()  # (B, D)
                    else:
                        encoder_latent = run_encoder_forward(encoder, item, param, device)

                    if isinstance(fusion, CausalTemporalTransformer):
                        # Collect for batch processing later
                        encoder_latents_seq.append(encoder_latent)

                        # Collect Time
                        curr_ts = record.timestamp

                        # Fallback if timestamp is missing from CSV
                        if curr_ts is None:
                             # Assume 1 unit per frame if no timestamp
                             curr_ts = float(seq_idx) * delta_unit

                        if start_ts is None: start_ts = curr_ts
                        try:
                            # Normalize time: (curr - start) / unit
                            rel_time = (curr_ts - start_ts) / delta_unit
                        except Exception:
                            rel_time = 0.0
                        encoder_times_seq.append(rel_time)

                        # We will compute 'hidden' later, so we can't compute loss YET.
                        # We need to restructure this loop to be 2 passes:
                        # Pass 1: Encode all frames
                        # Pass 2: Fuse (All at once if Transformer, Recursive if RNN)
                        # Pass 3: Loss
                        continue # Skip to next frame to collect more
                    elif use_cross_attn:
                        # Cross-attention: z_t queries past encoder outputs
                        curr_ts = record.timestamp
                        delta = metadata.compute_delta(prev_ts, curr_ts, unit=delta_unit)
                        cumulative_time += delta

                        memory = mem_buffer.get(seq_fruit_id)
                        hidden, gate_vals = fusion(encoder_latent, memory, current_abs_time=cumulative_time)
                        mem_buffer.append(seq_fruit_id, encoder_latent, cumulative_time)
                    else:
                        curr_ts = record.timestamp
                        delta = metadata.compute_delta(prev_ts, curr_ts, unit=delta_unit)
                        if reset_on_gap and max_gap is not None and delta > max_gap:
                            hidden = None
                            prev_ts = curr_ts
                            delta = 0.0
                            cumulative_time = 0.0  # reset absolute time on temporal gap
                            running_max_smooth = 0.0  # reset quality tracker on temporal gap
                            running_max_pca    = 0.0
                            running_bbox_min_pct = None
                            running_bbox_max_pct = None
                            running_max_vox = 0.0
                        cumulative_time += delta
                        delta_tensor = torch.tensor([[delta]], device=device, dtype=encoder_latent.dtype)
                        t_abs_tensor = torch.tensor([[cumulative_time]], device=device, dtype=encoder_latent.dtype)
                        if use_kalman and "partial_pcd" in item:
                            use_quality = param.get("use_quality", True)
                            if use_quality:
                                quality_fn = param.get("quality_fn", "smoothness")
                                quality_mode = param.get("quality_mode", "normalized")
                                if quality_mode == "raw_exp":
                                    # raw_exp: pass raw metric value directly, model learns scaling
                                    if quality_fn == "combined":
                                        smooth_t = compute_smoothness(item["partial_pcd"])
                                        pca_t = compute_pca_trace(item["partial_pcd"])
                                        q_t = (smooth_t * pca_t) ** 0.5
                                    elif quality_fn == "pca_trace":
                                        q_t = compute_pca_trace(item["partial_pcd"])
                                    elif quality_fn == "ang_az_entropy":
                                        q_t = compute_ang_az_entropy(item["partial_pcd"])
                                    else:
                                        q_t = compute_smoothness(item["partial_pcd"])
                                else:
                                    # normalized: running max normalization → q_t ∈ (0, 1]
                                    if quality_fn == "point_to_plane":
                                        pcd_w = item.get("partial_pcd_world", item["partial_pcd"])
                                        mean_res = compute_point_to_plane_residual(pcd_w)
                                        z_m = pcd_w.norm(dim=1).median().item()
                                        q_t = float(np.clip(1.0 - mean_res / (z_m + 1e-8), 0.0, 1.0))
                                    elif quality_fn == "combined":
                                        q_t, running_max_smooth, running_max_pca = compute_combined_quality(
                                            item["partial_pcd"], running_max_smooth, running_max_pca
                                        )
                                    elif quality_fn == "pca_trace":
                                        pca_t = compute_pca_trace(item["partial_pcd"])
                                        running_max_pca = max(running_max_pca, pca_t)
                                        q_t = pca_t / running_max_pca if running_max_pca > 1e-8 else 1.0
                                    elif quality_fn == "smoothness_vox":
                                        smooth_t = compute_smoothness(item["partial_pcd"])
                                        running_max_smooth = max(running_max_smooth, smooth_t)
                                        q_smooth = smooth_t / running_max_smooth if running_max_smooth > 1e-8 else 1.0
                                        vox, running_bbox_min_pct, running_bbox_max_pct = compute_voxel_coverage(
                                            item["partial_pcd"], running_bbox_min_pct, running_bbox_max_pct
                                        )
                                        vox_scale = float(param.get("vox_scale", 0.15))
                                        q_vox = min(vox / vox_scale, 1.0)
                                        q_t = q_smooth * q_vox
                                    elif quality_fn == "ang_az_entropy":
                                        # Already in [0, 1]; no running-max normalisation needed.
                                        q_t = compute_ang_az_entropy(item["partial_pcd"])
                                    else:  # smoothness (default)
                                        smooth_t = compute_smoothness(item["partial_pcd"])
                                        running_max_smooth = max(running_max_smooth, smooth_t)
                                        q_t = smooth_t / running_max_smooth if running_max_smooth > 1e-8 else 1.0
                                quality = torch.tensor([[q_t]], device=device, dtype=encoder_latent.dtype)
                            else:
                                quality = None  # ablation: fixed R, no quality modulation
                            hidden, gate_vals = fusion(encoder_latent, hidden, delta_tensor, quality=quality, t_abs=t_abs_tensor, learned_R=learned_R_t)
                        elif use_kalman:
                            hidden, gate_vals = fusion(encoder_latent, hidden, delta_tensor, t_abs=t_abs_tensor, learned_R=learned_R_t)
                        else:
                            hidden, gate_vals = fusion(encoder_latent, hidden, delta_tensor)

                    # Extract decodable latent: mu only for Kalman, full hidden for all others
                    output_latent = fusion.get_output(hidden) if use_kalman else hidden

                    # Phase 8: Consistency Loss (Regularization toward Frozen Encoder)
                    use_consistency = bool(param.get("use_consistency_loss", True))
                    lambda_consistency = float(param.get("lambda_consistency", 0.0))

                    if use_consistency and lambda_consistency > 0.0 and not is_last_frame:
                         # Detach encoder_latent just in case (though it is from frozen encoder)
                        cons_loss = nn.MSELoss()(output_latent, encoder_latent.detach())

                        if not should_supervise_all:
                             # Normalize consistency loss
                             cons_loss = cons_loss / max(1, len(sequence) - 1)

                        seq_loss += lambda_consistency * cons_loss
                        writer.add_scalar("Train/consistency_loss", cons_loss.item(), global_step)

                    collected_hiddens.append(output_latent)
                    target_latent = item["latent"].to(device).unsqueeze(0)
                    last_target = target_latent # Assuming constant target, keep the last one seen

                    is_two_stream = param.get("two_stream", False)

                    if is_two_stream:
                         # Split Hidden
                         shape_dim = fusion.shape_dim
                         pred_shape = output_latent[:, :shape_dim]
                         pred_size = output_latent[:, shape_dim:]

                         # 1. Shape Loss (Dense) - Constancy/Stability
                         loss_shape = SuperLoss(pred_shape, target_latent)
                         seq_loss += param.get("lambda_shape", 1.0) * loss_shape
                         writer.add_scalar("Train/shape_loss", loss_super.item() if 'loss_super' in locals() else loss_shape.item(), global_step) # Re-use or separate

                         # 2. Size Stream Losses
                         gt_size = item.get("size")
                         if gt_size is not None:
                             gt_size = gt_size.to(device)
                             if gt_size.dim() == 1:
                                 gt_size = gt_size.unsqueeze(0)

                         # A. Monotonicity (Curr >= Prev)
                         if len(collected_hiddens) > 1:
                              pred_prev_size = collected_hiddens[-2][:, shape_dim:]
                              # Penalty for shrinking: Relu(Prev - Curr)
                              mono_loss = torch.relu(pred_prev_size - pred_size).mean()
                              seq_loss += param.get("lambda_mono", 1.0) * mono_loss
                              writer.add_scalar("Train/mono_loss", mono_loss.item(), global_step)

                         # B. Final Frame Size (MSE)
                         if is_last_frame and gt_size is not None:
                              # Compare final predictive size with GT final size
                              try:
                                  size_loss = nn.MSELoss()(pred_size, gt_size)
                                  seq_loss += param.get("lambda_size", 1.0) * size_loss
                                  writer.add_scalar("Train/size_loss", size_loss.item(), global_step)
                              except RuntimeError:
                                  # Shape mismatch fallback
                                  pass

                    # Compute loss only if we supervise all frames OR if it's the last frame (Standard Mode)
                    elif should_supervise_all or is_last_frame:
                        loss_super = SuperLoss(output_latent, target_latent)
                        seq_loss += param.get("lambda_super", 1.0) * loss_super
                        steps += 1
                        writer.add_scalar("Train/frame_loss", loss_super.item(), global_step)

                    # Caliper weak intermediate loss: per-frame log-volume regression on fused latent.
                    # Last frame uses laser GT (item['fruit_id'] → df), intermediate frames use
                    # caliper-derived proxy_volume_ml. Loss is L1 in log-space.
                    # Two modes: 'mlp' (vol_head_temporal MLP) or 'softvol' (decoder + sigmoid soft volume).
                    if _caliper_weak_temporal:
                        _is_weak_t = bool(item['is_weak'].item()) if isinstance(item.get('is_weak'), torch.Tensor) else bool(item.get('is_weak', False))
                        _proxy_t = float(item['proxy_volume_ml'].item()) if isinstance(item.get('proxy_volume_ml'), torch.Tensor) else -1.0
                        _gt_vol_for_frame = -1.0
                        _lam_vol = 0.0
                        if _is_weak_t and _proxy_t > 0:
                            _gt_vol_for_frame = _proxy_t
                            _lam_vol = float(param.get('lambda_volume_weak', 1.0))
                        elif is_last_frame:
                            try:
                                _fid_for_vol = item['fruit_id'][0] if isinstance(item['fruit_id'], (list, tuple)) else item['fruit_id']
                                _vals = df.loc[df['label'] == _fid_for_vol, 'volume_ml'].values
                                if len(_vals) > 0 and float(_vals[0]) > 0:
                                    _gt_vol_for_frame = float(_vals[0])
                                    _lam_vol = float(param.get('lambda_volume_strong', 1.0))
                            except Exception:
                                pass
                        if _gt_vol_for_frame > 0 and _lam_vol > 0:
                            log_gt = torch.log(torch.tensor(max(_gt_vol_for_frame, 1e-3), device=device, dtype=output_latent.dtype))
                            if _weak_vol_loss_mode == "mlp" and vol_head_temporal is not None:
                                log_vol_pred = vol_head_temporal(output_latent).squeeze()
                                vol_loss_frame = F.l1_loss(log_vol_pred, log_gt)
                            elif _weak_vol_loss_mode == "softvol":
                                # Decoder forward at fused latent on bbox grid → soft volume
                                box = tensor_dict_2_float_dict(item['bbox'])
                                grid = Grid3D(param['grid_density'], device, output_latent.dtype, bbox=box)
                                deepsdf_input = torch.cat([output_latent.expand(grid.points.size(0), -1), grid.points], dim=1)
                                pred_sdf = decoder(deepsdf_input)  # (N, 1) in normalized SDF units
                                T_temp = float(param.get('softvol_temperature', 0.005))
                                soft_inside = torch.sigmoid(-pred_sdf / T_temp)  # (N, 1) approx 1.0 inside, 0 outside
                                # voxel volume in m³ from bbox extents (assuming bbox is in meters)
                                gd = float(param['grid_density'])
                                vox_m3 = ((box['xmax'] - box['xmin']) * (box['ymax'] - box['ymin']) * (box['zmax'] - box['zmin'])) / (gd ** 3)
                                soft_vol_ml = soft_inside.sum() * vox_m3 * 1e6  # m³ → ml (1 m³ = 1e6 ml)
                                log_vol_pred = torch.log(soft_vol_ml.clamp(min=1e-3))
                                vol_loss_frame = F.l1_loss(log_vol_pred, log_gt)
                            else:
                                vol_loss_frame = None
                            if vol_loss_frame is not None:
                                seq_loss = seq_loss + _lam_vol * vol_loss_frame
                                writer.add_scalar(f"Train/vol_loss_{'weak' if _is_weak_t else 'strong'}",
                                                 (_lam_vol * vol_loss_frame).item(), global_step)

                    # SDF loss on last_frame only (samples.npz contains last-date geometry,
                    # using it for intermediate frames would punish encoder for predicting smaller fruit).
                    if (is_last_frame
                        and bool(param.get("3D_loss", False))
                        and item.get("target_sdf") is not None
                        and item.get("target_sdf_xyz") is not None):
                        try:
                            t_xyz = item["target_sdf_xyz"]
                            t_sdf = item["target_sdf"]
                            if t_xyz.dim() == 2:
                                t_xyz = t_xyz.unsqueeze(0)
                            if t_sdf.dim() == 2:
                                t_sdf = t_sdf.unsqueeze(0)
                            t_xyz = t_xyz.to(device, dtype=output_latent.dtype)
                            t_sdf = t_sdf.to(device, dtype=output_latent.dtype)
                            N_pts = t_xyz.shape[1]
                            sdf_input = torch.cat([output_latent.unsqueeze(1).expand(-1, N_pts, -1), t_xyz], dim=-1)
                            sdf_input = sdf_input.reshape(-1, sdf_input.shape[-1])
                            pred_sdf_pts = decoder(sdf_input).reshape(-1, N_pts, 1)
                            trunc = float(param.get("sdf_trunc", 0.015))
                            pred_sdf_clamped = pred_sdf_pts.clamp(-trunc, trunc)
                            tgt_sdf_clamped = t_sdf.clamp(-trunc, trunc)
                            loss_sdf = F.l1_loss(pred_sdf_clamped, tgt_sdf_clamped)
                            _lam_sdf = float(param.get("lambda_sdf", 10.0))
                            seq_loss = seq_loss + _lam_sdf * loss_sdf
                            writer.add_scalar("Train/sdf_loss", (_lam_sdf * loss_sdf).item(), global_step)
                        except Exception as _ex_sdf:
                            pass

                    gate_mean += gate_vals.mean().item()
                    prev_ts = curr_ts
                    global_step += 1
                
                # --- Transformer Batch Pass (If applicable) ---
                if isinstance(fusion, CausalTemporalTransformer) and len(encoder_latents_seq) > 0:
                    # Stack: (SeqLen, Batch=1, D) -> (Batch=1, SeqLen, D)
                    seq_tensor = torch.stack(encoder_latents_seq).transpose(0, 1) 
                    
                    # Stack Times: (SeqLen, Batch=1) -> (Batch=1, SeqLen)
                    # We need to create a tensor from the list of floats
                    times_tensor = torch.tensor(encoder_times_seq, device=device, dtype=torch.float32).unsqueeze(0)

                    # Forward Pass
                    fused_seq = fusion(seq_tensor, sequence_times=times_tensor) # (1, SeqLen, D)
                    
                    # Unpack and Compute Losses recursively (imitating the loop above)
                    # We reuse the logic by iterating again. 
                    # Ideally we should refactor the loss logic into a function, but for minimal diff:
                    
                    for i, hidden in enumerate(fused_seq.transpose(0, 1)): # Iterate (1, D)
                        record = sequence[i]
                        item = dataset[record.dataset_index]
                        encoder_latent = encoder_latents_seq[i]
                        is_last_frame = (i == len(sequence) - 1)
                        
                        collected_hiddens.append(hidden)
                        target_latent = item["latent"].to(device).unsqueeze(0)
                        last_target = target_latent
                        
                        # --- Copy-Paste Loss Logic from above (Refactored slightly) ---
                        # Phase 8: Consistency
                        use_consistency = bool(param.get("use_consistency_loss", True))
                        lambda_consistency = float(param.get("lambda_consistency", 0.0))
                    
                        if use_consistency and lambda_consistency > 0.0 and not is_last_frame:
                            cons_loss = nn.MSELoss()(hidden, encoder_latent.detach())
                            if not should_supervise_all:
                                cons_loss = cons_loss / max(1, len(sequence) - 1)
                            seq_loss += lambda_consistency * cons_loss
                            writer.add_scalar("Train/consistency_loss", cons_loss.item(), global_step) # Approximate step
                            
                        # Supervision (Only Scalar Frame Loss supported for Transformer for now)
                        if should_supervise_all or is_last_frame:
                            loss_super = SuperLoss(hidden, target_latent)
                            seq_loss += param.get("lambda_super", 1.0) * loss_super
                            steps += 1
                            writer.add_scalar("Train/frame_loss", loss_super.item(), global_step)
                            
                        # Dummy gate val
                        gate_mean += 0.5 
                        global_step += 1 # Catch up global steps

                    
                if steps == 0:
                    continue
                # Normalize by total sequence length (aligned with train_weak_supervision.py)
                seq_loss = seq_loss / len(sequence)
                
                # Add Temporal Triplet Loss
                use_triplet = bool(param.get("use_triplet_loss", True))
                if use_triplet and param.get("lambda_triplet", 0.0) > 0 and len(collected_hiddens) > 1 and last_target is not None:
                    triplet_loss = compute_temporal_triplet_loss(collected_hiddens, last_target, margin=0.1)
                    seq_loss += param.get("lambda_triplet", 0.1) * triplet_loss
                    writer.add_scalar("Train/triplet_loss", triplet_loss.item(), global_step)

                gate_mean /= len(sequence)

                # Normalize loss by accumulation steps
                loss_normalized = seq_loss / accumulation_steps
                loss_normalized.backward()
                
                # Gradient Clipping (Applied only on step)
                # But here we stick to clipping every micro-step or macro-step?
                # PyTorch usually clips on the accumulated gradients.
                
                if (seq_idx + 1) % accumulation_steps == 0 or (seq_idx + 1) == len(epoch_sequences):
                    torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()


                writer.add_scalar("Train/sequence_loss", seq_loss.item(), global_step)
                writer.add_scalar("Train/gate_mean", gate_mean, global_step)
                epoch_loss += seq_loss.item()
                seq_count += 1
                progress.set_postfix({"seq_loss": f"{seq_loss.item():.4f}"})
            if seq_count == 0:
                continue
            avg_loss = epoch_loss / seq_count
            writer.add_scalar("Train/epoch_loss", avg_loss, epoch)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("Train/learning_rate", current_lr, epoch)
            print(f"[temporal-train] Epoch {epoch}/{param['epoch']} - loss: {avg_loss:.6f} - lr: {current_lr:.6f}")

            # Step LR warmup scheduler
            if scheduler is not None:
                scheduler.step()

            if avg_loss < best_loss:
                best_loss = avg_loss
                # Save just before validation? No, usually based on validation metric if available.
                # But here we keep saving best training loss model as 'best_model.pt' for safety.
                save_path = checkpoints_dir / f"train_loss_{param['checkpoint_file']}"
                ckpt = {
                        "epoch": epoch,
                        "encoder_state_dict": encoder.state_dict(),
                        "fusion_state_dict": fusion.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                        "global_step": global_step,
                    }
                if vol_head_temporal is not None:
                    ckpt["vol_head_temporal_state_dict"] = vol_head_temporal.state_dict()
                torch.save(ckpt, save_path)
                print(f"[temporal-train] ✓ Saved best model (train_loss) to {save_path}")
            elif epoch % checkpoint_every == 0:
                save_path = checkpoints_dir / f"epoch_{epoch:04d}.pt"
                ckpt = {
                        "epoch": epoch,
                        "encoder_state_dict": encoder.state_dict(),
                        "fusion_state_dict": fusion.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                        "global_step": global_step,
                    }
                if vol_head_temporal is not None:
                    ckpt["vol_head_temporal_state_dict"] = vol_head_temporal.state_dict()
                torch.save(ckpt, save_path)

            # --- VALIDATION LOOP ---
            val_freq = int(param.get("validation_frequency", 5))
            if epoch % val_freq == 0:
                print("\n[temporal-train] Starting validation...")
                encoder.eval()
                fusion.eval()

                # Lazy load validation dataset/metadata
                if not hasattr(train_temporal, "val_loader"):
                    print("[temporal-train] Loading validation dataset...")
                    val_dataset = prepare_dataset(param, pretrain=param["pretrain_path"], split="val")
                    train_temporal.val_dataset = val_dataset
                    train_temporal.val_metadata = TemporalMetadataIndex(
                        val_dataset.files,
                        timestamp_csv=param.get("timestamp_csv"),
                        min_sequence_length=int(param.get("min_sequence_length", 2)),
                        orbbec_suffix=param.get("orbbec_pcd_suffix", "_preprocessed"),
                    )
                    # Load GT CSV
                    # Load GT CSV
                    gt_csv_path = param.get("val_gt_csv")
                    if gt_csv_path and os.path.exists(gt_csv_path):
                         try:
                             train_temporal.gt_df = pd.read_csv(gt_csv_path)
                             print(f"[temporal-train] Loaded GT CSV: {gt_csv_path}")
                         except Exception as e:
                             print(f"[temporal-train] Failed to load GT CSV: {e}")
                             train_temporal.gt_df = pd.DataFrame()
                    else:
                         # Backward compatibility or fallback
                         species = str(param.get('species', '')).lower()
                         fallback_path = "./data/3DKmelonDemo/ground_truth.csv" if 'kmelon' in species else "./data/3DPotatoTwinDemo/ground_truth.csv"
                         if os.path.exists(fallback_path):
                              train_temporal.gt_df = pd.read_csv(fallback_path)
                              print(f"[temporal-train] Loaded GT CSV (fallback): {fallback_path}")
                         else:
                              print(f"[temporal-train] No GT CSV found. RMSE Volume metric will be skipped.")
                              train_temporal.gt_df = pd.DataFrame()

                val_meta = train_temporal.val_metadata
                val_ds = train_temporal.val_dataset
                gt_df = train_temporal.gt_df
                
                
                val_sequences = list(val_meta.iter_sequences(shuffle=False))
                
                pred_volumes = []
                gt_volumes = []
                chamfer_distances = []
                all_pred_points = []
                all_gt_points = []
                val_seq_counts = 0

                val_loss_sum = 0.0
                val_loss_count = 0
                val_cons_sum = 0.0
                val_cons_count = 0
                val_triplet_sum = 0.0
                val_triplet_count = 0

                # Grid for meshing
                # Note: Validation meshing is slow, so we only do it if validation_frequency is reasonable
                val_use_cross_attn = isinstance(fusion, CrossAttentionTemporalFusion)
                val_use_kalman = isinstance(fusion, (LatentKalmanFusion, LatentKalmanFusionV2, LatentKalmanFusionV3))

                with torch.no_grad():
                    for seq in tqdm(val_sequences, desc="Validation", leave=False):
                        hidden = None
                        prev_ts = None

                        collected_hiddens = []
                        last_target = None
                        val_running_max_smooth = 0.0  # for Kalman quality within this val sequence
                        val_running_max_pca    = 0.0
                        val_running_bbox_min_pct: Optional[torch.Tensor] = None
                        val_running_bbox_max_pct: Optional[torch.Tensor] = None
                        val_running_max_vox = 0.0

                        # Cross-attention memory buffer for validation
                        if val_use_cross_attn:
                            val_mem_buffer = EncoderMemoryBuffer(max_len=int(param.get("max_memory_len", 20)))
                            val_seq_key = "val_seq"
                        val_cumulative_time = 0.0

                        # Process whole sequence
                        for idx, record in enumerate(seq):
                            item = val_ds[record.dataset_index]
                            val_use_learned_R = bool(param.get("use_learned_R_from_encoder", False))
                            val_learned_R_t = None
                            if val_use_learned_R and hasattr(encoder, "encode_with_aux"):
                                _aux = run_encoder_forward_with_aux(encoder, item, param, device)
                                latent = _aux["latent"]
                                if "log_scale" in _aux:
                                    _ls = _aux["log_scale"].clamp(
                                        float(param.get("nll_log_scale_min", -3.0)),
                                        float(param.get("nll_log_scale_max", 3.0)),
                                    )
                                    val_learned_R_t = (2.0 * _ls).exp()
                            else:
                                latent = run_encoder_forward(encoder, item, param, device)
                            curr_ts = record.timestamp
                            delta = val_meta.compute_delta(prev_ts, curr_ts, unit=delta_unit)
                            if reset_on_gap and max_gap is not None and delta > max_gap:
                                hidden = None
                                prev_ts = curr_ts
                                delta = 0.0
                                val_cumulative_time = 0.0  # reset absolute time on gap
                                collected_hiddens = [] # Reset collection on gap
                                val_running_max_smooth = 0.0  # reset quality tracker on gap
                                val_running_max_pca    = 0.0
                                val_running_bbox_min_pct = None
                                val_running_bbox_max_pct = None
                                val_running_max_vox = 0.0
                                if val_use_cross_attn:
                                    val_mem_buffer.reset(val_seq_key)
                            val_cumulative_time += delta
                            delta_tensor = torch.tensor([[delta]], device=device, dtype=latent.dtype)
                            val_t_abs_tensor = torch.tensor([[val_cumulative_time]], device=device, dtype=latent.dtype)
                            if isinstance(fusion, CausalTemporalTransformer):
                                # For transformer, we need the full history of inputs up to now
                                # validation 'hidden' variable will store the list of encoder latents
                                if hidden is None:
                                    hidden = [] # Use 'hidden' variable to store history list
                                hidden.append(latent)

                                # Forward pass on history (Batch=1, SeqLen=t, D)
                                # hidden is list of (1, D) tensors
                                # stack -> (t, 1, D)
                                # permute -> (1, t, D)
                                seq_tensor = torch.stack(hidden).permute(1, 0, 2)
                                out_seq = fusion(seq_tensor) # (1, t, D)

                                # Current prediction is the last one
                                current_pred = out_seq[0, -1, :] # (D)

                                # Returns to update loop variable
                                hidden_state_out = hidden
                                gate_val = torch.tensor([0.0], device=device) # Dummy

                                # Use prediction for metrics
                                logging_latent = current_pred.unsqueeze(0)

                            elif val_use_cross_attn:
                                # Cross-attention: z_t queries past encoder outputs
                                memory = val_mem_buffer.get(val_seq_key)
                                hidden, gate_val = fusion(latent, memory, current_abs_time=val_cumulative_time)
                                val_mem_buffer.append(val_seq_key, latent, val_cumulative_time)
                                hidden_state_out = hidden
                                logging_latent = hidden

                            elif val_use_kalman:
                                # Kalman: quality-weighted state update
                                val_use_quality = param.get("use_quality", True)
                                if "partial_pcd" in item and val_use_quality:
                                    val_quality_fn = param.get("quality_fn", "smoothness")
                                    val_quality_mode = param.get("quality_mode", "normalized")
                                    if val_quality_mode == "raw_exp":
                                        if val_quality_fn == "combined":
                                            smooth_t = compute_smoothness(item["partial_pcd"])
                                            pca_t = compute_pca_trace(item["partial_pcd"])
                                            q_t = (smooth_t * pca_t) ** 0.5
                                        elif val_quality_fn == "pca_trace":
                                            q_t = compute_pca_trace(item["partial_pcd"])
                                        elif val_quality_fn == "ang_az_entropy":
                                            q_t = compute_ang_az_entropy(item["partial_pcd"])
                                        else:
                                            q_t = compute_smoothness(item["partial_pcd"])
                                    else:
                                        if val_quality_fn == "point_to_plane":
                                            pcd_w = item.get("partial_pcd_world", item["partial_pcd"])
                                            mean_res = compute_point_to_plane_residual(pcd_w)
                                            z_m = pcd_w.norm(dim=1).median().item()
                                            q_t = float(np.clip(1.0 - mean_res / (z_m + 1e-8), 0.0, 1.0))
                                        elif val_quality_fn == "combined":
                                            q_t, val_running_max_smooth, val_running_max_pca = compute_combined_quality(
                                                item["partial_pcd"], val_running_max_smooth, val_running_max_pca
                                            )
                                        elif val_quality_fn == "pca_trace":
                                            pca_t = compute_pca_trace(item["partial_pcd"])
                                            val_running_max_pca = max(val_running_max_pca, pca_t)
                                            q_t = pca_t / val_running_max_pca if val_running_max_pca > 1e-8 else 1.0
                                        elif val_quality_fn == "smoothness_vox":
                                            smooth_t = compute_smoothness(item["partial_pcd"])
                                            val_running_max_smooth = max(val_running_max_smooth, smooth_t)
                                            q_smooth = smooth_t / val_running_max_smooth if val_running_max_smooth > 1e-8 else 1.0
                                            vox, val_running_bbox_min_pct, val_running_bbox_max_pct = compute_voxel_coverage(
                                                item["partial_pcd"], val_running_bbox_min_pct, val_running_bbox_max_pct
                                            )
                                            vox_scale = float(param.get("vox_scale", 0.15))
                                            q_vox = min(vox / vox_scale, 1.0)
                                            q_t = q_smooth * q_vox
                                        elif val_quality_fn == "ang_az_entropy":
                                            q_t = compute_ang_az_entropy(item["partial_pcd"])
                                        else:  # smoothness (default)
                                            smooth_t = compute_smoothness(item["partial_pcd"])
                                            val_running_max_smooth = max(val_running_max_smooth, smooth_t)
                                            q_t = smooth_t / val_running_max_smooth if val_running_max_smooth > 1e-8 else 1.0
                                    quality = torch.tensor([[q_t]], device=device, dtype=latent.dtype)
                                else:
                                    quality = None  # ablation or no partial_pcd: fixed R
                                hidden, gate_val = fusion(latent, hidden, delta_tensor, quality=quality, t_abs=val_t_abs_tensor, learned_R=val_learned_R_t)
                                hidden_state_out = hidden
                                logging_latent = fusion.get_output(hidden)

                            else:
                                # RNN Logic
                                hidden, gate_val = fusion(latent, hidden, delta_tensor)
                                hidden_state_out = hidden
                                logging_latent = hidden
                            
                            # Update loop variable
                            hidden = hidden_state_out
                            
                            collected_hiddens.append(logging_latent)
                            if 'latent' in item:
                                last_target = item["latent"].to(device).unsqueeze(0)
                                
                            prev_ts = curr_ts
                            
                            # Compute Val Loss (Supervision)
                            if 'latent' in item:
                                target_latent = item["latent"].to(device).unsqueeze(0)
                                loss_val = SuperLoss(logging_latent, target_latent)
                                val_loss_sum += loss_val.item()
                                val_loss_count += 1
                                
                            # Compute Val Consistency Loss (Diagnostic)
                            # ALWAYS compute this to track if model drifts from encoder
                            cons_val = nn.MSELoss()(logging_latent, latent.detach())
                            val_cons_sum += cons_val.item()
                            val_cons_count += 1
                            
                            # Mesh ONLY the last frame for metrics (metrics_final_frame_only logic)
                            if idx == len(seq) - 1:
                                try:
                                    box = tensor_dict_2_float_dict(item['bbox'])
                                    grid = Grid3D(param['grid_density'], device, torch.float32, bbox=box)
                                    deepsdf_input = torch.cat([logging_latent.expand(grid.points.size(0), -1), grid.points], dim=1)
                                    pred_sdf = decoder(deepsdf_input)
                                    
                                    # Get GT
                                    label = item['fruit_id']
                                    if isinstance(label, list): label = label[0]
                                    if isinstance(label, torch.Tensor): label = str(label.item())
                                    
                                    # Use configured threshold
                                    t_val = float(param.get('threshold', 0.0))
                                    mesh = sdf2mesh_cuda(pred_sdf, grid.points, t=t_val)
                                    vol = mesh.get_volume() * 1e6 # cm3
                                    pred_volumes.append(vol)
                                    
                                    gt_row = gt_df.loc[gt_df['label'] == label]
                                    if not gt_row.empty:
                                        gt_vol = gt_row.iloc[0]['volume_ml']
                                        if pd.isna(gt_vol) and 'volume_metashape' in gt_row.columns:
                                            gt_vol = gt_row.iloc[0]['volume_metashape']
                                        if not pd.isna(gt_vol):
                                            gt_volumes.append(gt_vol)

                                    # Chamfer / F1: decode GT latent -> GT mesh -> compare
                                    if 'latent' in item and item['latent'].abs().sum() > 0:
                                        gt_latent_val = item['latent'].to(device).unsqueeze(0)
                                        gt_deepsdf_input = torch.cat([gt_latent_val.expand(grid.points.size(0), -1), grid.points], dim=1)
                                        gt_sdf = decoder(gt_deepsdf_input)
                                        gt_mesh = sdf2mesh_cuda(gt_sdf, grid.points, t=t_val)
                                        n_samples = param['grid_density'] ** 3
                                        try:
                                            gt_pts = np.asarray(gt_mesh.sample(n_samples))
                                        except Exception:
                                            gt_pts = np.asarray(gt_mesh.vertices)
                                        try:
                                            pred_pts = np.asarray(mesh.sample(n_samples))
                                        except Exception:
                                            pred_pts = np.asarray(mesh.vertices)

                                        if len(gt_pts) > 0 and len(pred_pts) > 0:
                                            pred_pcd_t = torch.tensor(pred_pts, device=device, dtype=torch.float32)
                                            gt_pcd_t = torch.tensor(gt_pts, device=device, dtype=torch.float32)
                                            cd = chamfer_distance_torch(pred_pcd_t, gt_pcd_t)
                                            chamfer_distances.append(cd.item() * 1000)  # mm
                                            all_pred_points.append(pred_pcd_t.cpu())
                                            all_gt_points.append(gt_pcd_t.cpu())

                                except Exception as exc:
                                    print(f"[Val-Error] Mesh/Metric failed for {label}: {exc}")
                        
                        # Compute Sequence Triplet Loss
                        if len(collected_hiddens) > 1 and last_target is not None:
                            triplet_val = compute_temporal_triplet_loss(collected_hiddens, last_target, margin=0.1)
                            val_triplet_sum += triplet_val.item()
                            val_triplet_count += 1
                                    
                        val_seq_counts += 1 # Increment sequence count AFTER processing sequence
                
                # Compute Metrics & Print in requested format
                if gt_volumes and pred_volumes and len(gt_volumes) == len(pred_volumes):
                    gt_arr = np.array(gt_volumes)
                    pred_arr = np.array(pred_volumes)
                    mse = np.mean((gt_arr - pred_arr) ** 2)
                    rmse = np.sqrt(mse)
                    mape_mask = gt_arr > 0
                    mape = float(np.mean(np.abs(pred_arr[mape_mask] - gt_arr[mape_mask]) / gt_arr[mape_mask]) * 100.0) if mape_mask.any() else float('inf')

                    mean_pred = np.mean(pred_arr)
                    mean_gt = np.mean(gt_arr)

                    print(f"RMSE volume:  {rmse:.1f}")
                    print(f"  MAPE volume: {mape:.2f}%")
                    print(f"  Mean predicted: {mean_pred:.1f}, Mean GT: {mean_gt:.1f}")
                    print(f"  Pred range: [{pred_arr.min():.1f}, {pred_arr.max():.1f}]")
                    print(f"  GT range: [{gt_arr.min():.1f}, {gt_arr.max():.1f}]")

                    writer.add_scalar("Val/rmse_volume", rmse, epoch)
                    writer.add_scalar("Val/mape_volume", mape, epoch)
                else:
                    print(f"RMSE Volume: N/A (GT/Pred mismatch or empty)")
                    print(f"  Debug: len(gt)={len(gt_volumes)}, len(pred)={len(pred_volumes)}")
                    rmse = float('inf'); mape = float('inf')
                
                # Chamfer distance
                if chamfer_distances:
                    cd_mean = float(np.mean(chamfer_distances))
                    print(f"[validate] Chamfer distance: {cd_mean:.2f} mm (avg over {len(chamfer_distances)} samples)")
                    writer.add_scalar("Val/chamfer_dist_mm", cd_mean, epoch)
                else:
                    print(f"[validate] Chamfer distance: N/A")

                # F1 / Precision / Recall @ 5mm (macro-average)
                if all_pred_points and all_gt_points:
                    threshold_m = 5.0 / 1000.0
                    sample_precs, sample_recs, sample_f1s = [], [], []
                    for pred_pcd_c, gt_pcd_c in zip(all_pred_points, all_gt_points):
                        pred_pcd_c = pred_pcd_c.to(device)
                        gt_pcd_c = gt_pcd_c.to(device)
                        if len(pred_pcd_c) == 0 or len(gt_pcd_c) == 0:
                            sample_precs.append(0.0)
                            sample_recs.append(0.0)
                            sample_f1s.append(0.0)
                            continue
                        d_p2g = torch.min(torch.cdist(pred_pcd_c.unsqueeze(0), gt_pcd_c.unsqueeze(0)).squeeze(0), dim=1)[0]
                        p = (d_p2g < threshold_m).sum().item() / len(d_p2g) * 100.0
                        d_g2p = torch.min(torch.cdist(gt_pcd_c.unsqueeze(0), pred_pcd_c.unsqueeze(0)).squeeze(0), dim=1)[0]
                        r = (d_g2p < threshold_m).sum().item() / len(d_g2p) * 100.0
                        f_sample = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                        sample_precs.append(p)
                        sample_recs.append(r)
                        sample_f1s.append(f_sample)
                    prec = sum(sample_precs) / len(sample_precs)
                    rec = sum(sample_recs) / len(sample_recs)
                    f1 = sum(sample_f1s) / len(sample_f1s)
                    print(f"[validate] F1 @ 5mm: {f1:.4f} (macro-avg)")
                    print(f"[validate] Precision @ 5mm: {prec:.4f}")
                    print(f"[validate] Recall @ 5mm: {rec:.4f}")
                    writer.add_scalar("Val/f1_5mm", f1, epoch)
                    writer.add_scalar("Val/precision_5mm", prec, epoch)
                    writer.add_scalar("Val/recall_5mm", rec, epoch)
                else:
                    print(f"[validate] F1/Precision/Recall: N/A")

                if val_loss_count > 0:
                    val_loss_mean = val_loss_sum / val_loss_count
                    print(f"Val loss (Supervision, avg): {val_loss_mean:.6f}")
                    writer.add_scalar("Val/val_loss_super", val_loss_mean, epoch)
                else:
                    val_loss_mean = float('inf')
                    print(f"Val loss: N/A")

                if val_cons_count > 0 and bool(param.get("use_consistency_loss", True)):
                    val_cons_mean = val_cons_sum / val_cons_count
                    print(f"Val Consistency (Encoder Dist, avg): {val_cons_mean:.6f}")
                    writer.add_scalar("Val/val_loss_consistency", val_cons_mean, epoch)
                    
                if val_triplet_count > 0 and bool(param.get("use_triplet_loss", True)):
                    val_triplet_mean = val_triplet_sum / val_triplet_count
                    print(f"Val Triplet (Temporal Order, avg): {val_triplet_mean:.6f}")
                    writer.add_scalar("Val/val_loss_triplet", val_triplet_mean, epoch)

                # Save best model independently for each tracked validation metric:
                # val_loss (super loss avg), val_rmse_volume, val_mape_volume.
                # Mirrors static train.py — every metric gets its own checkpoint.
                # The "primary" metric (config: model_selection_metric) also writes the
                # main checkpoint_file path used by downstream tools.
                _val_loss_now = val_loss_mean if val_loss_count > 0 else float('inf')
                _val_rmse_now = rmse if (gt_volumes and pred_volumes and len(gt_volumes) == len(pred_volumes)) else float('inf')
                _val_mape_now = mape if (gt_volumes and pred_volumes and len(gt_volumes) == len(pred_volumes)) else float('inf')

                if not hasattr(train_temporal, "_best_loss_val"):
                    train_temporal._best_loss_val = float('inf')
                    train_temporal._best_rmse_val = float('inf')
                    train_temporal._best_mape_val = float('inf')

                def _build_state(epoch_, best_val, name):
                    sd = {
                        "epoch": epoch_,
                        "encoder_state_dict": encoder.state_dict(),
                        "fusion_state_dict": fusion.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric": best_val,
                        "metric_name": name,
                        "global_step": global_step,
                    }
                    if vol_head_temporal is not None:
                        sd["vol_head_temporal_state_dict"] = vol_head_temporal.state_dict()
                    return sd

                # Save each metric to its own file (no symlink/main update unless primary).
                tracked = [
                    ("val_loss",        _val_loss_now, "_best_loss_val", "val_loss_best_model.pt"),
                    ("val_rmse_volume", _val_rmse_now, "_best_rmse_val", "val_rmse_volume_best_model.pt"),
                    ("val_mape_volume", _val_mape_now, "_best_mape_val", "val_mape_volume_best_model.pt"),
                ]
                sel = str(param.get("model_selection_metric", "val_rmse")).lower()
                primary_keys = {
                    "val_loss":        "val_loss",
                    "val_rmse":        "val_rmse_volume",
                    "val_rmse_volume": "val_rmse_volume",
                    "val_mape":        "val_mape_volume",
                    "val_mape_volume": "val_mape_volume",
                }
                primary_name = primary_keys.get(sel, "val_rmse_volume")

                for name, value, attr, fname in tracked:
                    if value == float('inf'): continue
                    cur_best = getattr(train_temporal, attr)
                    if value < cur_best:
                        setattr(train_temporal, attr, value)
                        save_path = checkpoints_dir / fname
                        sd = _build_state(epoch, value, name)
                        torch.save(sd, save_path)
                        if name == primary_name:
                            best_metric_val = value
                            main_best_path = checkpoints_dir / param["checkpoint_file"]
                            torch.save(sd, main_best_path)
                            print(f"[temporal-train] ✓ Saved best {name}: {value:.4f}  [PRIMARY → {param['checkpoint_file']}]")
                        else:
                            print(f"[temporal-train] ✓ Saved best {name}: {value:.4f}")

                if not param.get("freeze_encoder", False):
                    encoder.train()
                fusion.train()

    finally:
        writer.close()
        if tb_process:
            tb_process.terminate()
            try:
                tb_process.wait(timeout=5)
            except Exception:
                tb_process.kill()
    print(f"[temporal-train] Training complete. Best loss: {best_loss:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temporal CoRe++ encoder")
    parser.add_argument("--cfg", dest="cfg", required=True)
    parser.add_argument(
        "--checkpoint_decoder",
        dest="checkpoint_decoder",
        default=None,
        help="Decoder checkpoint epoch. Overrides config if provided",
    )
    parser.add_argument(
        "--experiment",
        dest="experiment_directory",
        default=None,
        help="Path to the DeepSDF experiment directory",
    )
    parser.add_argument(
        "--tensorboard_port",
        type=int,
        default=None,
        help="If provided, automatically launch TensorBoard on this port pointing to the current run logs.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint (.pt) to resume training from (encoder/fusion/optimizer).",
    )
    deep_sdf.add_common_args(parser)
    args = parser.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        param = json.load(f)

    if args.experiment_directory is None:
        args.experiment_directory = param.get("experiment")
        if args.experiment_directory is not None:
            print(f"[temporal-train] Using experiment from config: {args.experiment_directory}")
    if args.experiment_directory is None:
        raise ValueError("Experiment directory must be provided via --experiment or config")

    checkpoint_decoder = args.checkpoint_decoder or param.get("checkpoint_decoder") or "301"

    specs = ws.load_experiment_specifications(args.experiment_directory)
    latent_size = specs["CodeLength"]

    # Load Decoder (for validation)
    print(f"[temporal-train] Loading decoder from {args.experiment_directory} (ckpt {checkpoint_decoder})...")
    arch = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])
    
    # Inject ExplicitSize parameters if enabled, ensuring Decoder structure matches checkpoint
    explicit_size_conf = specs.get("ExplicitSize", {})
    if explicit_size_conf.get("Enabled", False):
        tgt_type = explicit_size_conf.get("Target", {}).get("Type", "volume")
        explicit_dim = 3 if tgt_type == "bbox_extents" else 1
        embed_dim = explicit_size_conf.get("EmbeddingDim", 0)
        
        specs["NetworkSpecs"]["explicit_size_dim"] = explicit_dim
        specs["NetworkSpecs"]["size_embed_dim"] = embed_dim
        print(f"[temporal-train] ExplicitSize Enabled: dim={explicit_dim}, embed={embed_dim}")

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()
    decoder_ckpt_path = os.path.join(
        args.experiment_directory,
        "ModelParameters",
        str(checkpoint_decoder),
    ) + ".pth"
    # Load weights logic similar to test.py
    decoder_checkpoint = torch.load(decoder_ckpt_path, map_location="cuda")
    if isinstance(decoder_checkpoint, dict):
        if "model_state_dict" in decoder_checkpoint:
            decoder_state = decoder_checkpoint["model_state_dict"]
        elif "decoder_state_dict" in decoder_checkpoint:
            decoder_state = decoder_checkpoint["decoder_state_dict"]
        else:
            decoder_state = decoder_checkpoint
    else:
        decoder_state = decoder_checkpoint
    # Handle module prefix
    if any(key.startswith("module.") for key in decoder_state.keys()):
        decoder_state = {key.replace("module.", "", 1): value for key, value in decoder_state.items()}
    decoder.load_state_dict(decoder_state)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False
    print("[temporal-train] Decoder loaded and frozen.")

    param["pretrain_path"] = os.path.join(
        args.experiment_directory,
        "Reconstructions",
        str(checkpoint_decoder),
        "Codes",
        "complete",
    )

    _seed_everything()
    train_temporal(args, param, latent_size, decoder)


if __name__ == "__main__":
    main()
