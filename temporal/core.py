"""Shared helpers for the temporal CoRe++ pipeline."""
from __future__ import annotations

import hashlib
import random
from typing import Dict
import sys
from pathlib import Path


def stable_sample_seed(sample_id: str, base: int = 42) -> int:
    """Dataset-order-independent occlusion seed derived from a stable sample id.

    Used so that the same sample at the same occlusion ratio receives an
    identical occlusion pattern across static / temporal / repeated runs,
    regardless of dataset iteration order or split changes.
    """
    h = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()
    return ((int(h[:8], 16) ^ base) & 0x7fffffff)

# Add root directory to python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _repo_bootstrap  # noqa: F401  (binds the repo root as `corepp`)

import torch
import torchvision.transforms as transforms

from corepp.dataloaders.cameralaser_w_masks import MaskedCameraLaserData
from corepp.dataloaders.transforms import Pad, Rotate, RandomHorizontalFlip, RandomVerticalFlip, RandomScale
from corepp.networks.models import (

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
    PointCloudEncoderSplit,
    PointNetPlusPlusEncoder,
    FoldNetEncoder,
)

POINT_CLOUD_ENCODERS = {"point_cloud", "point_cloud_maxmean", "point_cloud_maxmean_masked", "point_cloud_large", "point_cloud_maxmean_large", "point_cloud_maxmean_large_masked", "foldnet", "point_cloud_split", "pointnet2"}


def build_encoder(param: Dict, latent_size: int, device: torch.device) -> torch.nn.Module:
    encoder_name = param["encoder"]
    size = param.get("input_size", 304)
    _vol = bool(param.get("volume_head", False))
    _nll = bool(param.get("nll_super_loss", False)) or bool(param.get("use_learned_R_from_encoder", False))
    if encoder_name == "big":
        encoder = EncoderBig(in_channels=4, out_channels=latent_size, size=size)
    elif encoder_name == "small_pool":
        encoder = EncoderPooled(in_channels=4, out_channels=latent_size, size=size)
    elif encoder_name == "erfnet":
        encoder = ERFNetEncoder(in_channels=4, out_channels=latent_size, size=size)
    elif encoder_name == "pool":
        encoder = EncoderBigPooled(in_channels=4, out_channels=latent_size, size=size)
    elif encoder_name == "double":
        encoder = DoubleEncoder(out_channels=latent_size, size=size)
    elif encoder_name == "point_cloud":
        encoder = PointCloudEncoder(in_channels=3, out_channels=latent_size)
    elif encoder_name == "point_cloud_maxmean":
        encoder = PointCloudEncoderMaxMean(in_channels=3, out_channels=latent_size)
    elif encoder_name == "point_cloud_maxmean_masked":
        encoder = PointCloudEncoderMaxMeanMasked(
            in_channels=3, out_channels=latent_size,
            volume_head=_vol, nll_head=_nll,
        )
    elif encoder_name == "point_cloud_split":
        # Note: latent_size here should be total size (e.g. 35)
        encoder = PointCloudEncoderSplit(in_channels=3, out_channels=latent_size)
    elif encoder_name == "point_cloud_large":
        encoder = PointCloudEncoderLarge(in_channels=3, out_channels=latent_size)
    elif encoder_name == "point_cloud_maxmean_large":
        encoder = PointCloudEncoderMaxMeanLarge(in_channels=3, out_channels=latent_size)
    elif encoder_name == "point_cloud_maxmean_large_masked":
        encoder = PointCloudEncoderMaxMeanLargeMasked(
            in_channels=3, out_channels=latent_size,
            volume_head=_vol, nll_head=_nll,
        )
    elif encoder_name == "pointnet2":
        encoder = PointNetPlusPlusEncoder(in_channels=3, out_channels=latent_size)
    elif encoder_name == "foldnet":
        encoder = FoldNetEncoder(in_channels=3, out_channels=latent_size)
    else:
        encoder = Encoder(in_channels=4, out_channels=latent_size, size=size)
    return encoder.to(device)


def prepare_dataset(param: Dict, pretrain: str, split: str = "train") -> MaskedCameraLaserData:
    scale_tf = RandomScale(scale_range=(0.5, 1.3))
    other_geo_tfs = transforms.RandomChoice([Rotate(angle=45), RandomHorizontalFlip(), RandomVerticalFlip()])
    geo_tfs = transforms.Compose([scale_tf, other_geo_tfs])
    color_tfs = [geo_tfs, transforms.ColorJitter(brightness=0.5, hue=(-0.1, 0.1), saturation=0.5), Pad(size=param["input_size"])]
    color_tf = transforms.Compose(color_tfs)
    default_tfs = [geo_tfs, Pad(size=param["input_size"])]
    default_tf = transforms.Compose(default_tfs)

    dataset = MaskedCameraLaserData(
        data_source=param["data_dir"],
        tf=default_tf,
        color_tf=color_tf,
        pretrain=pretrain,
        pad_size=param["input_size"],
        detection_input=param["detection_input"],
        normalize_depth=param["normalize_depth"],
        depth_min=param["depth_min"],
        depth_max=param["depth_max"],
        supervised_3d=param.get("supervised_3d", True),
        sdf_loss=bool(param.get("3D_loss", False)),
        grid_density=param["grid_density"],
        split=split,
        overfit=False,
        species=param["species"],
        use_phenobox_pcd=bool(param.get("use_phenobox_pcd", False)),
        phenobox_subdir=param.get("phenobox_subdir", "phenobox/pointclouds"),
        use_orbbec_pcd=bool(param.get("use_orbbec_pcd", False)),
        orbbec_pcd_suffix=param.get("orbbec_pcd_suffix", "_preprocessed"),
        split_json_path=param.get("split_json", None),
        require_dataset_json=param.get("require_dataset_json", True),
        temporal_loading=True,
        cache_pcds_in_ram=bool(param.get("cache_pcds_in_ram", False)),
        vol_map_csv=param.get("vol_map_csv", None),
        gt_vol_similar_thr=float(param.get("gt_vol_similar_thr", 0.05)),
        # Caliper weak supervision filters intermediate frames lacking caliper data,
        # which is correct for training. For val/test we want the full split (all 101
        # last-date fruits) — disable filtering on those splits by default.
        # Override: set eval_caliper_filter=true in config to enable filter at test/val too
        # (drops intermediate frames lacking caliper measurements; still keeps last_date frames).
        caliper_weak_supervision=bool(param.get("caliper_weak_supervision", False)) and (
            split == "train" or bool(param.get("eval_caliper_filter", False))
        ),
        gt_volume_csv=param.get("gt_volume_csv", None),
        caliper_correct_proxy=bool(param.get("caliper_correct_proxy", False)),
        use_pcd_render_as_rgbd=bool(param.get("use_pcd_render_as_rgbd", False)),
        render_size=int(param.get("render_size", 304)),
        render_splat_radius=int(param.get("render_splat_radius", 3)),
        cache_rendered_rgbd=bool(param.get("cache_rendered_rgbd", False)),
        render_fixed_scale=bool(param.get("render_fixed_scale", False)),
        render_fixed_xy_half_extent=float(param.get("render_fixed_xy_half_extent", 0.07)),
        render_fixed_z_half_extent=float(param.get("render_fixed_z_half_extent", 0.07)),
        render_target_points=param.get("render_target_points", None),
        render_target_jitter_mm=float(param.get("render_target_jitter_mm", 1.5)),
        use_raw_camera_rgbd=bool(param.get("use_raw_camera_rgbd", False)),
        raw_rgbd_depth_min_mm=float(param.get("raw_rgbd_depth_min_mm", 500.0)),
        raw_rgbd_depth_max_mm=float(param.get("raw_rgbd_depth_max_mm", 1100.0)),
        raw_rgbd_bbox_margin=float(param.get("raw_rgbd_bbox_margin", 0.10)),
    )
    return dataset


def extract_encoder_input(item: Dict, param: Dict, device: torch.device) -> torch.Tensor:
    encoder_type = param["encoder"]
    if encoder_type not in POINT_CLOUD_ENCODERS:
        # Move RGB / depth to GPU first, then concat there. The CPU-side cat on
        # 1.5 MB tensors otherwise takes ~85 ms under load (PyTorch CPU thread
        # contention with other workloads); the GPU-side path is ~1 ms total.
        rgb = item["rgb"].unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
        depth = item["depth"].unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
        return torch.cat((rgb, depth), dim=1)

    partial = item["partial_pcd"]
    if partial.dim() == 2:
        partial = partial.unsqueeze(0)
    return partial.permute(0, 2, 1).contiguous().to(device=device, dtype=torch.float32)


def run_encoder_forward(encoder: torch.nn.Module, item: Dict, param: Dict, device: torch.device) -> torch.Tensor:
    encoder_type = param["encoder"]
    if encoder_type in POINT_CLOUD_ENCODERS:
        encoder_input = extract_encoder_input(item, param, device)
        if encoder_type in ("point_cloud_maxmean_masked", "point_cloud_maxmean_large_masked") and "partial_pcd_real_mask" in item:
            mask = item["partial_pcd_real_mask"]
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            mask = mask.to(device=device)
            return encoder(encoder_input, mask=mask)
        return encoder(encoder_input)
    # Image-based encoders (vanilla CoRe++ baselines): consume RGB+D via the
    # rgb/depth keys that the dataloader's rendering hook populates.
    encoder_input = extract_encoder_input(item, param, device)
    return encoder(encoder_input)


def run_encoder_forward_with_aux(encoder: torch.nn.Module, item: Dict, param: Dict, device: torch.device):
    """Extended forward that returns latent + optional auxiliary outputs.

    Returns dict with key 'latent' always present; optionally 'log_scale' (heteroscedastic NLL),
    'log_vol' (volume head). Falls back gracefully if encoder lacks ``encode_with_aux``.
    """
    encoder_type = param["encoder"]
    if encoder_type not in POINT_CLOUD_ENCODERS:
        # Image-based encoders: no aux outputs (volume_head off for vanilla
        # CoRe++ baselines). Return just the latent.
        encoder_input = extract_encoder_input(item, param, device)
        return {"latent": encoder(encoder_input)}
    encoder_input = extract_encoder_input(item, param, device)
    mask = None
    if encoder_type in ("point_cloud_maxmean_masked", "point_cloud_maxmean_large_masked") and "partial_pcd_real_mask" in item:
        m = item["partial_pcd_real_mask"]
        if m.dim() == 1:
            m = m.unsqueeze(0)
        mask = m.to(device=device)
    if hasattr(encoder, "encode_with_aux"):
        if mask is not None:
            return encoder.encode_with_aux(encoder_input, mask=mask)
        return encoder.encode_with_aux(encoder_input)
    # Fallback: deterministic latent only
    if mask is not None:
        return {"latent": encoder(encoder_input, mask=mask)}
    return {"latent": encoder(encoder_input)}


def run_encoder_features(encoder: torch.nn.Module, item: Dict, param: Dict, device: torch.device) -> torch.Tensor:
    """Return post-pooling features (1024d) instead of final latent."""
    encoder_input = extract_encoder_input(item, param, device)
    return encoder.encode_features(encoder_input)


# ---------------------------------------------------------------------------
# Occlusion augmentation
# ---------------------------------------------------------------------------

def apply_plane_occlusion(
    pcd: torch.Tensor,
    max_occlusion: float = 0.5,
    min_points: int = 64,
) -> torch.Tensor:
    """Simulate occlusion by removing points on one side of a random plane.

    Unlike uniform random dropout, this removes a spatially contiguous region
    similar to real-world occlusion by leaves, other fruits, or the ground.

    Args:
        pcd: point cloud, shape ``(N, 3)`` (channel-last, zero-padded).
        max_occlusion: maximum fraction of *real* points to remove.
            The actual fraction is sampled from ``U(0.0, max_occlusion)``.
        min_points: minimum number of real points to keep.

    Returns:
        Augmented point cloud, same shape ``(N, 3)`` with removed points
        replaced by zeros (matching the existing padding convention).
    """
    N = pcd.shape[0]

    # Identify real (non-padding) points
    norms = pcd.norm(dim=1)
    real_mask = norms > 1e-8
    n_real = int(real_mask.sum().item())

    if n_real <= min_points:
        return pcd  # too few points to occlude

    real_indices = torch.where(real_mask)[0]
    real_points = pcd[real_indices]  # (n_real, 3)

    # Random occlusion plane: sample direction on unit sphere
    normal = torch.randn(3, device=pcd.device, dtype=pcd.dtype)
    normal = normal / (normal.norm() + 1e-8)

    # Project real points onto the normal direction (relative to centroid)
    centroid = real_points.mean(dim=0)
    projections = (real_points - centroid) @ normal  # (n_real,)

    # Sample how much to remove
    removal_frac = random.uniform(0.0, max_occlusion)
    n_keep = max(int(n_real * (1.0 - removal_frac)), min_points)

    # Keep points with the largest projections (remove the "occluded" side)
    _, keep_order = projections.topk(n_keep)
    kept = real_points[keep_order]  # (n_keep, 3)

    # Rebuild padded tensor
    result = torch.zeros_like(pcd)
    result[:n_keep] = kept
    return result


def apply_random_occlusion(
    pcd: torch.Tensor,
    max_occlusion: float = 0.5,
    min_points: int = 64,
) -> torch.Tensor:
    """Simulate sparse observation by randomly dropping points (NOT contiguous).

    Samples a removal fraction from U(0, max_occlusion), then drops that fraction
    uniformly at random across real points. Mimics a low-density observation,
    which is what actually happens when fruit is small/far in the greenhouse camera.
    """
    N = pcd.shape[0]
    norms = pcd.norm(dim=1)
    real_mask = norms > 1e-8
    n_real = int(real_mask.sum().item())
    if n_real <= min_points:
        return pcd
    real_indices = torch.where(real_mask)[0]
    real_points = pcd[real_indices]
    removal_frac = random.uniform(0.0, max_occlusion)
    n_keep = max(int(n_real * (1.0 - removal_frac)), min_points)
    # Uniform random permutation, keep first n_keep
    perm = torch.randperm(n_real, device=pcd.device)[:n_keep]
    kept = real_points[perm]
    result = torch.zeros_like(pcd)
    result[:n_keep] = kept
    return result


def apply_patch_occlusion(
    pcd: torch.Tensor,
    max_occlusion: float = 0.5,
    min_points: int = 64,
) -> torch.Tensor:
    """Simulate occlusion by keeping only a local patch around a random anchor point.

    Picks a random real point as anchor, then keeps its nearest neighbors.
    This simulates a partial view where only a contiguous local region is visible
    (e.g., single camera viewpoint on part of the fruit).

    Args:
        pcd: point cloud, shape ``(N, 3)`` (channel-last, zero-padded).
        max_occlusion: maximum fraction of *real* points to remove.
            The actual fraction is sampled from ``U(0.0, max_occlusion)``.
        min_points: minimum number of real points to keep.

    Returns:
        Augmented point cloud, same shape ``(N, 3)`` with removed points
        replaced by zeros (matching the existing padding convention).
    """
    N = pcd.shape[0]

    # Identify real (non-padding) points
    norms = pcd.norm(dim=1)
    real_mask = norms > 1e-8
    n_real = int(real_mask.sum().item())

    if n_real <= min_points:
        return pcd  # too few points to occlude

    real_indices = torch.where(real_mask)[0]
    real_points = pcd[real_indices]  # (n_real, 3)

    # Sample how many to keep
    removal_frac = random.uniform(0.0, max_occlusion)
    n_keep = max(int(n_real * (1.0 - removal_frac)), min_points)

    # Pick a random anchor point, keep its n_keep nearest neighbors
    anchor_idx = random.randint(0, n_real - 1)
    anchor = real_points[anchor_idx]
    dists = (real_points - anchor).norm(dim=1)
    _, keep_order = dists.topk(n_keep, largest=False)
    kept = real_points[keep_order]  # (n_keep, 3)

    # Rebuild padded tensor
    result = torch.zeros_like(pcd)
    result[:n_keep] = kept
    return result


def _oversample_interp(
    kept: torch.Tensor, N: int, seed: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Interpolation-based oversampling to fill kept points back to N."""
    if kept.shape[0] >= N:
        return kept
    n_new = N - kept.shape[0]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 1000)

    anchor_idx = torch.randint(0, kept.shape[0], (n_new,), generator=gen, device=device)
    anchors = kept[anchor_idx]

    dists = torch.cdist(anchors, kept)
    dists.scatter_(1, anchor_idx.unsqueeze(1), float('inf'))
    nn_idx = dists.argmin(dim=1)
    neighbors = kept[nn_idx]

    alpha = torch.rand(n_new, 1, device=device, dtype=dtype, generator=gen)
    new_points = anchors + alpha * (neighbors - anchors)
    return torch.cat([kept, new_points], dim=0)


def apply_test_occlusion(
    pcd: torch.Tensor,
    occlusion_frac: float,
    min_points: int = 64,
    seed: int = 42,
    mode: str = "plane",
) -> torch.Tensor:
    """Deterministic occlusion for test-time robustness evaluation.

    Args:
        pcd: point cloud, shape ``(N, 3)``.
        occlusion_frac: exact fraction of real points to remove (e.g. 0.4).
        min_points: minimum number of real points to keep.
        seed: random seed for reproducibility.
        mode: ``"plane"`` removes one side via a random plane.
              ``"erode"`` removes points from the boundary inward
              (outermost points w.r.t. centroid are removed first).
              ``"patch"`` picks a random surface point and keeps
              only its nearest neighbors (simulates a local partial view).

    Returns:
        Occluded point cloud, shape ``(N, 3)``.
    """
    if occlusion_frac <= 0:
        return pcd

    N = pcd.shape[0]
    norms = pcd.norm(dim=1)
    real_mask = norms > 1e-8
    n_real = int(real_mask.sum().item())

    if n_real <= min_points:
        return pcd

    real_indices = torch.where(real_mask)[0]
    real_points = pcd[real_indices]
    n_keep = max(int(n_real * (1.0 - occlusion_frac)), min_points)

    if mode == "plane":
        gen = torch.Generator(device=pcd.device)
        gen.manual_seed(seed)
        normal = torch.randn(3, device=pcd.device, dtype=pcd.dtype, generator=gen)
        normal = normal / (normal.norm() + 1e-8)

        centroid = real_points.mean(dim=0)
        projections = (real_points - centroid) @ normal
        _, keep_order = projections.topk(n_keep)
        kept = real_points[keep_order]

    elif mode == "erode":
        centroid = real_points.mean(dim=0)
        dists = (real_points - centroid).norm(dim=1)
        _, keep_order = dists.topk(n_keep, largest=False)
        kept = real_points[keep_order]

    elif mode == "patch":
        # Pick a random anchor point on the surface, keep its n_keep nearest neighbors
        gen = torch.Generator(device=pcd.device)
        gen.manual_seed(seed)
        anchor_idx = torch.randint(0, n_real, (1,), generator=gen, device=pcd.device).item()
        anchor = real_points[anchor_idx]
        dists = (real_points - anchor).norm(dim=1)
        _, keep_order = dists.topk(n_keep, largest=False)
        kept = real_points[keep_order]

    else:
        raise ValueError(f"Unknown occlusion mode: {mode!r}")

    kept = _oversample_interp(kept, N, seed, pcd.device, pcd.dtype)

    result = torch.zeros_like(pcd)
    result[:N] = kept
    return result
