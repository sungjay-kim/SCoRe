#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _repo_bootstrap  # noqa: F401  (binds the repo root as `corepp`)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import json
from tqdm import tqdm

import os
import pandas as pd
import sys
import atexit

from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import deepsdf.deep_sdf as deep_sdf
import deepsdf.deep_sdf.workspace as ws

from sdfrenderer.grid import Grid3D

from dataloaders.cameralaser_w_masks import MaskedCameraLaserData, collate_fn_variable_pcd
from torch.utils.data import Dataset
from dataloaders import dataset_layout as layout


class ScaleExpandedDataset(Dataset):
    """Wraps a dataset to include all scale-augmented variants deterministically.

    For each original sample, produces (1 + len(scale_pcts)) items:
      - index 0: original sample (scale=100)
      - index 1..K: PCD scaled by scale_pcts[k]/100, latent swapped

    Total length = len(base_dataset) * (1 + len(scale_pcts)).
    """

    def __init__(self, base_dataset, scaled_latents, scale_pcts):
        self.base = base_dataset
        self.scaled_latents = scaled_latents
        self.scale_pcts = [100] + list(scale_pcts)  # 100 = original
        self.n_scales = len(self.scale_pcts)

    def __len__(self):
        return len(self.base) * self.n_scales

    def __getattr__(self, name):
        """Proxy attribute access to base dataset (e.g. sdf_trunc)."""
        if name in ('base', 'scaled_latents', 'scale_pcts', 'n_scales'):
            raise AttributeError(name)
        return getattr(self.base, name)

    def __getitem__(self, idx):
        base_idx = idx // self.n_scales
        scale_idx = idx % self.n_scales
        s_pct = self.scale_pcts[scale_idx]

        item = self.base[base_idx]

        # Ensure latent is always on CPU (some .pth files may have been saved on CUDA)
        if 'latent' in item and isinstance(item['latent'], torch.Tensor):
            item['latent'] = item['latent'].cpu()

        item['scale_pct'] = s_pct  # 100 = original, <100 = scaled

        if s_pct != 100:
            s_factor = s_pct / 100.0
            # Scale PCD
            if isinstance(item['partial_pcd'], list):
                item['partial_pcd'] = [p * s_factor for p in item['partial_pcd']]
            else:
                item['partial_pcd'] = item['partial_pcd'] * s_factor
            # Swap latent target
            fid = item['fruit_id']
            key = f"{fid}_s{s_pct}"
            if key in self.scaled_latents:
                item['latent'] = self.scaled_latents[key]

        return item


class BinBalancedDataset(Dataset):
    """Bin-uniform sampling via stratified scale augmentation.

    For each __getitem__(i), picks bin = i % num_bins (round-robin) then
    weighted-samples a (fruit_idx, scale_pct) pair from that bin's pool.
    Resulting volume = vol_orig * (s/100)^3 falls in the chosen bin.

    Pool composition per bin:
      - Natural samples (s=100, only when vol_orig itself is in this bin),
        weighted by size_inverse: w = (vol_ref / vol_orig)^alpha
      - Scaled samples (s<100, when vol_orig*(s/100)^3 lands in this bin),
        with uniform weight 1.0 (all augmented samples treated equally).

    Notes:
      - size_inverse weight is applied ONLY to natural data per user request.
      - When sampling from an empty bin, falls back to any non-empty bin.
    """

    def __init__(
        self,
        base_dataset,
        scaled_latents,
        scale_pcts,
        vol_last_per_file,
        bin_edges,
        target_count_per_bin,
        size_inverse_sampling=False,
        size_inverse_alpha=0.5,
        size_inverse_vol_ref=300.0,
        cap_aug_to_max_natural=False,
        cap_aug_seed=42,
        density_scale_aug=False,
        density_scale_exponent=2.0,
        density_scale_min_pts=64,
    ):
        from collections import defaultdict
        import random as _random
        self.base = base_dataset
        self.scaled_latents = scaled_latents
        self.scale_pcts = scale_pcts
        self.bin_edges = list(bin_edges)
        self.num_bins = len(self.bin_edges) - 1
        self.target = int(target_count_per_bin)
        self._random = _random.Random()  # local RNG (not seed-locked) for stochastic resampling
        self.density_scale_aug = bool(density_scale_aug)
        self.density_scale_exponent = float(density_scale_exponent)
        self.density_scale_min_pts = int(density_scale_min_pts)

        natural_by_bin = defaultdict(list)
        aug_by_bin = defaultdict(list)
        files = getattr(base_dataset, 'files', None)
        n_items = len(base_dataset)

        for idx in range(n_items):
            vol = vol_last_per_file[idx] if (vol_last_per_file is not None and idx < len(vol_last_per_file)) else 0
            if vol is None or vol <= 0:
                continue

            # Natural entry (s=100)
            bn = self._bin_of(vol)
            if bn is not None:
                w_nat = 1.0
                if size_inverse_sampling:
                    w_nat = (size_inverse_vol_ref / vol) ** size_inverse_alpha
                natural_by_bin[bn].append((idx, 100, w_nat))

            # Scaled entries
            if files is not None and idx < len(files):
                fid = self._fruit_id_from_path(files[idx])
                for s in scale_pcts:
                    key = f"{fid}_s{s}"
                    if key in scaled_latents:
                        vol_after = vol * ((s / 100.0) ** 3)
                        bn_a = self._bin_of(vol_after)
                        if bn_a is not None:
                            aug_by_bin[bn_a].append((idx, int(s), 1.0))

        # Optional: cap aug per bin so each bin has at most max_natural entries.
        # aug[bn] = max(0, max_natural_across_bins - n_natural[bn]) per bin (random subsample).
        # Idea: equalize bin sizes to the largest natural-only bin, capping aug expansion.
        # Reduces aug-domination in large bins where natural is plentiful, while still
        # padding small bins with whatever aug is available.
        self._cap_aug_to_max_natural = bool(cap_aug_to_max_natural)
        if self._cap_aug_to_max_natural:
            import random as _r2
            _capper = _r2.Random(cap_aug_seed)
            max_nat = max((len(natural_by_bin[bn]) for bn in range(self.num_bins)), default=0)
            for bn in range(self.num_bins):
                n_nat = len(natural_by_bin[bn])
                aug_quota = max(0, max_nat - n_nat)
                if len(aug_by_bin[bn]) > aug_quota:
                    aug_by_bin[bn] = _capper.sample(aug_by_bin[bn], aug_quota)
            print(f"[BinBalanced] cap_aug_to_max_natural=True  max_natural={max_nat}  (aug quotas computed)")

        self.combined_pool = {}
        for bn in range(self.num_bins):
            self.combined_pool[bn] = natural_by_bin[bn] + aug_by_bin[bn]

        # Stats logging
        print(f"[BinBalanced] num_bins={self.num_bins}  target_per_bin={self.target}  total_per_epoch={self.target*self.num_bins}")
        if self.density_scale_aug:
            print(f"[BinBalanced] density_scale_aug=True  exponent={self.density_scale_exponent}  min_pts={self.density_scale_min_pts}  (scaled samples will have real_pts ~ n_real * s^{self.density_scale_exponent})")
        for bn in range(self.num_bins):
            n_nat = len(natural_by_bin[bn])
            n_aug = len(aug_by_bin[bn])
            edge_lo = self.bin_edges[bn]
            edge_hi = self.bin_edges[bn + 1]
            tag = ""
            if not self.combined_pool[bn]:
                tag = "  ⚠️ EMPTY"
            print(f"  bin {bn} [{edge_lo:>4}-{edge_hi:>4}]: {n_nat:>4} natural + {n_aug:>4} scaled = {n_nat + n_aug:>4} entries{tag}")

    def _bin_of(self, vol):
        for i in range(self.num_bins):
            if self.bin_edges[i] <= vol < self.bin_edges[i + 1]:
                return i
        return None

    @staticmethod
    def _fruit_id_from_path(path):
        return layout.fruit_id_from_path(path)

    def __getattr__(self, name):
        """Proxy to base dataset (e.g., dataset.Ks)."""
        if name in ('base', 'scaled_latents', 'scale_pcts', 'bin_edges', 'num_bins',
                    'target', '_random', 'combined_pool'):
            raise AttributeError(name)
        return getattr(self.__dict__['base'], name)

    def __len__(self):
        return self.num_bins * self.target

    def __getitem__(self, i):
        bn = i % self.num_bins
        pool = self.combined_pool[bn]
        if not pool:
            for alt in range(self.num_bins):
                if self.combined_pool[alt]:
                    pool = self.combined_pool[alt]
                    bn = alt
                    break
        weights = [e[2] for e in pool]
        fruit_idx, s_pct, _ = self._random.choices(pool, weights=weights, k=1)[0]

        item = self.base[fruit_idx]
        if 'latent' in item and isinstance(item['latent'], torch.Tensor):
            item['latent'] = item['latent'].cpu()
        item['scale_pct'] = int(s_pct)

        if s_pct != 100:
            s_factor = s_pct / 100.0
            if isinstance(item.get('partial_pcd'), list):
                item['partial_pcd'] = [p * s_factor for p in item['partial_pcd']]
            elif item.get('partial_pcd') is not None:
                item['partial_pcd'] = item['partial_pcd'] * s_factor
            fid = item.get('fruit_id', '')
            if isinstance(fid, str):
                key = f"{fid}_s{s_pct}"
                if key in self.scaled_latents:
                    item['latent'] = self.scaled_latents[key]
            # SDF loss for scale-augmented sample: linearly rescale the cached SDF
            # target (xyz and sdf both scale by s_factor — distance scales linearly).
            # If target_sdf_xyz unavailable (samples.npz missing), fall back to None.
            if item.get('target_sdf_xyz') is not None and item.get('target_sdf') is not None:
                item['target_sdf_xyz'] = item['target_sdf_xyz'] * s_factor
                item['target_sdf']     = item['target_sdf']     * s_factor
                # target_sdf_weights stays unchanged (still all 1)
            else:
                item['target_sdf'] = None
                item['target_sdf_weights'] = None
                item['target_sdf_xyz'] = None
            # Density coupling: a smaller fruit projects fewer points onto the depth sensor.
            # Surface area scales as s^2 (default exponent), so subsample real pts by that
            # factor and replicate-pad the rest, mirroring how the dataloader pads short PCDs.
            # Updates partial_pcd_real_mask so the masked encoder pools over the new n_real.
            if (self.density_scale_aug
                    and isinstance(item.get('partial_pcd'), torch.Tensor)
                    and 'partial_pcd_real_mask' in item):
                pcd = item['partial_pcd']  # (N, 3)
                mask = item['partial_pcd_real_mask']  # (N,) bool
                n_real = int(mask.sum().item())
                if n_real > self.density_scale_min_pts:
                    keep_frac = s_factor ** self.density_scale_exponent
                    n_keep = max(int(round(n_real * keep_frac)), self.density_scale_min_pts)
                    n_keep = min(n_keep, n_real)
                    if n_keep < n_real:
                        N = pcd.shape[0]
                        real_idx = torch.where(mask)[0]
                        perm = torch.randperm(n_real)
                        keep_idx = real_idx[perm[:n_keep]]
                        kept_pts = pcd[keep_idx]
                        new_pcd = pcd.clone()
                        new_pcd[:n_keep] = kept_pts
                        if N > n_keep:
                            fill_idx = torch.randint(0, n_keep, (N - n_keep,))
                            new_pcd[n_keep:] = kept_pts[fill_idx]
                        new_mask = torch.zeros_like(mask)
                        new_mask[:n_keep] = True
                        item['partial_pcd'] = new_pcd
                        item['partial_pcd_real_mask'] = new_mask
                        item['n_real'] = n_keep
        # Detach all tensors (covers BinBalanced's modifications: scaled partial_pcd,
        # swapped latent). Mirrors the same guard in MaskedCameraLaserData.__getitem__.
        for _k, _v in list(item.items()):
            if isinstance(_v, torch.Tensor):
                if _v.requires_grad or _v.grad_fn is not None:
                    item[_k] = _v.detach().requires_grad_(False)
        return item
from dataloaders.transforms import Pad, Rotate, RandomHorizontalFlip, RandomVerticalFlip, RandomScale
from dataloaders.pcd_transforms import PCDRandomRotate, PCDRandomJitter, PCDRandomScale, Compose as PCDCompose

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

from loss import KLDivLoss, DistMatchLoss, SuperLoss, SDFLoss, RegLatentLoss, AttRepLoss
from utils import sdf2mesh_cuda, save_model, tensor_dict_2_float_dict
from temporal.core import apply_plane_occlusion, apply_patch_occlusion, apply_random_occlusion

DEBUG = True

torch.autograd.set_detect_anomaly(True)

from sklearn.metrics import mean_squared_error
from pathlib import Path
from datetime import datetime

def chamfer_distance_torch(pred_pcd, target_pcd, batch_size=1024):
    """
    Compute chamfer distance between two point clouds with batched processing to save memory.
    pred_pcd: (N, 3) tensor
    target_pcd: (M, 3) tensor
    batch_size: chunk size for batched computation
    Returns: scalar tensor
    """
    # Subsample if point clouds are too large
    max_points = 5000
    if pred_pcd.shape[0] > max_points:
        indices = torch.randperm(pred_pcd.shape[0])[:max_points]
        pred_pcd = pred_pcd[indices]
    if target_pcd.shape[0] > max_points:
        indices = torch.randperm(target_pcd.shape[0])[:max_points]
        target_pcd = target_pcd[indices]
    
    # pred to target (chunked)
    min_dists_pred_to_target = []
    for i in range(0, pred_pcd.shape[0], batch_size):
        pred_batch = pred_pcd[i:i+batch_size]
        # Compute distances from this batch to all target points
        pred_expanded = pred_batch.unsqueeze(1)  # (batch, 1, 3)
        target_expanded = target_pcd.unsqueeze(0)  # (1, M, 3)
        dist_matrix = torch.sum((pred_expanded - target_expanded) ** 2, dim=2)  # (batch, M)
        min_dists = torch.min(dist_matrix, dim=1)[0]  # (batch,)
        min_dists_pred_to_target.append(min_dists)
    pred_to_target = torch.cat(min_dists_pred_to_target).mean()
    
    # target to pred (chunked)
    min_dists_target_to_pred = []
    for i in range(0, target_pcd.shape[0], batch_size):
        target_batch = target_pcd[i:i+batch_size]
        # Compute distances from this batch to all pred points
        target_expanded = target_batch.unsqueeze(1)  # (batch, 1, 3)
        pred_expanded = pred_pcd.unsqueeze(0)  # (1, N, 3)
        dist_matrix = torch.sum((target_expanded - pred_expanded) ** 2, dim=2)  # (batch, N)
        min_dists = torch.min(dist_matrix, dim=1)[0]  # (batch,)
        min_dists_target_to_pred.append(min_dists)
    target_to_pred = torch.cat(min_dists_target_to_pred).mean()
    
    return (pred_to_target + target_to_pred) / 2.0

def check_direxcist(dir):
    if dir is not None:
        if not os.path.exists(dir):
            os.makedirs(dir)  # make new folder


def main_function(
    decoder,
    pretrain,
    cfg,
    latent_size,
    trunc_val,
    overfit,
    update_decoder,
    resume: bool = False,
    resume_checkpoint: str = None,
    resume_run_root: str = None,
):

    if DEBUG:
        # Seed precedence: config["seed"] > env SEED > default 143.
        # Allows re-rolling stuck local minima without editing source.
        import os as _os
        _seed_cfg = None
        try:
            with open(cfg) as _f:
                _seed_cfg = json.load(_f).get("seed")
        except Exception:
            pass
        _seed = int(_seed_cfg) if _seed_cfg is not None else int(_os.environ.get("SEED", 143))
        print(f"[train] seed = {_seed}")
        torch.manual_seed(_seed)
        random.seed(_seed)
        np.random.seed(_seed)

    cfg_fname = cfg.split('/')[-1].replace('.json', '')  # getting filename

    with open(cfg) as json_file:
        param = json.load(json_file)

    # Allow CLI overrides for resume (without editing JSON on disk)
    if resume:
        param["resume"] = True
    if resume_checkpoint:
        param["resume_checkpoint"] = str(resume_checkpoint)
        param["resume"] = True
    if resume_run_root:
        param["run_root"] = str(resume_run_root)
        param["resume"] = True

    # Use task_name if provided, otherwise fall back to encoder name
    task_name = str(param.get("task_name", param.get("encoder", "task")))

    # Configure run directories
    original_checkpoint_dir = Path(param["checkpoint_dir"]).resolve()

    def derive_base_dir(path: Path) -> Path:
        base = path
        while base.name in {"checkpoints", "runs", "logs"}:
            base = base.parent
        if base.name.lower() == "train":
            return base.parent
        if base.parent.name.lower() == "train":
            return base.parent.parent
        if base.name.lower() in {"pretrain", "finetune", "temporal"}:
            return base
        return base.parent

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if bool(param.get("resume", False)):
        # Prefer explicit run_root from config/CLI; otherwise infer from checkpoint_dir if possible.
        if param.get("run_root"):
            run_root = Path(param["run_root"]).resolve()
        elif original_checkpoint_dir.name == "checkpoints" and original_checkpoint_dir.parent.name == "logs":
            run_root = original_checkpoint_dir.parent.parent
        else:
            # Fallback: resume without a known run_root -> create a new run directory.
            base_dir = derive_base_dir(original_checkpoint_dir)
            run_name = f"{run_timestamp}_{task_name}"
            run_root = base_dir / run_name
        checkpoints_dir = run_root / "logs" / "checkpoints"
        runs_dir = run_root / "logs" / "runs"
    else:
        # Default behavior: timestamped run directory to avoid overwriting previous experiments
        base_dir = derive_base_dir(original_checkpoint_dir)
        run_name = f"{run_timestamp}_{task_name}"
        run_root = base_dir / run_name
        checkpoints_dir = run_root / "logs" / "checkpoints"
        runs_dir = run_root / "logs" / "runs"

    for directory in [checkpoints_dir, runs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    param["checkpoint_dir"] = str(checkpoints_dir)
    param["log_dir"] = str(runs_dir)
    param["run_root"] = str(run_root)

    # Also save stdout/stderr to a log file under the same run directory.
    log_txt_path = run_root / "logs" / "log.txt"
    log_fh = open(log_txt_path, "a", buffering=1)

    class _TeeIO:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for s in self._streams:
                try:
                    s.write(data)
                except Exception:
                    # Best-effort: don't crash training due to logging
                    pass
            self.flush()

        def flush(self):
            for s in self._streams:
                try:
                    s.flush()
                except Exception:
                    pass

    _old_stdout, _old_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeIO(_old_stdout, log_fh)
    sys.stderr = _TeeIO(_old_stderr, log_fh)

    def _cleanup_log_streams():
        try:
            sys.stdout = _old_stdout
            sys.stderr = _old_stderr
        finally:
            try:
                log_fh.close()
            except Exception:
                pass

    atexit.register(_cleanup_log_streams)

    print(f"[train] Output directory: {run_root}")
    print(f"[train] Task: {task_name}")
    print(f"[train] Console log: {log_txt_path}")

    model_prefix = cfg_fname
    
    # Update checkpoint_file parameter to match actual checkpoint filename that will be created
    param["checkpoint_file"] = f"{model_prefix}_best_model.pt"

    # Save config file to run directory for reproducibility
    config_save_path = run_root / f"{cfg_fname}.json"
    with open(config_save_path, 'w') as f:
        json.dump(param, f, indent=4)
    print(f"[train] Config saved to: {config_save_path}")
    print(f"[train] Checkpoint file will be: {param['checkpoint_file']}")

    check_direxcist(param["checkpoint_dir"])
    check_direxcist(param["log_dir"])
    device = 'cuda'
    shuffle = True
    # Model selection metric (lower is better):
    # - "val_volume_rmse" / "val_rmse" (default, current behavior) -- primary
    # - "val_loss" (validation loss computed from enabled loss components)
    # - "val_mape" / "val_volume_mape" (mean absolute percentage error on volume)
    model_selection_metric = str(param.get("model_selection_metric", "val_volume_rmse")).strip().lower()
    best_score = np.inf
    # Always track all three metrics independently and save separate best_<metric>_model.pt
    best_rmse_score = np.inf
    best_loss_score = np.inf
    best_mape_score = np.inf

    # creating variables for 3d grid for diff SDF renderer
    grid_density = param['grid_density']
    precision = torch.float32

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

    # Load pretrained encoder weights if specified (for fine-tuning)
    pretrained_encoder_path = param.get('pretrained_encoder', None)
    if pretrained_encoder_path:
        pretrained_encoder_path = Path(pretrained_encoder_path)

        if pretrained_encoder_path.exists():
            print(f"[train] Loading pretrained encoder from: {pretrained_encoder_path}")
            checkpoint = torch.load(pretrained_encoder_path, map_location=device)

            # Handle different checkpoint formats
            if 'encoder_state_dict' in checkpoint:
                pretrained_state = checkpoint['encoder_state_dict']
                print(f"[train] Loaded encoder weights from epoch {checkpoint.get('epoch', 'unknown')}")
            elif 'model_state_dict' in checkpoint:
                pretrained_state = checkpoint['model_state_dict']
            else:
                # Assume checkpoint is the state dict itself
                pretrained_state = checkpoint

            # Check if we need partial loading (e.g. MaxMean encoder with different FC layers)
            encoder_state = encoder.state_dict()
            shape_mismatch = {
                k for k in pretrained_state
                if k in encoder_state and pretrained_state[k].shape != encoder_state[k].shape
            }
            if shape_mismatch:
                print(f"[train] Partial load: skipping {len(shape_mismatch)} shape-mismatched keys: {shape_mismatch}")
                pretrained_state = {k: v for k, v in pretrained_state.items()
                                    if k in encoder_state and k not in shape_mismatch}
                encoder.load_state_dict(pretrained_state, strict=False)
            else:
                # Non-strict load: volume_head (or other new heads) absent from old checkpoint
                missing, unexpected = encoder.load_state_dict(pretrained_state, strict=False)
                if missing:
                    print(f"[train] New params not in checkpoint (random init): {len(missing)} keys (e.g. {list(missing)[:3]})")
                if unexpected:
                    print(f"[train] Unexpected keys in checkpoint: {len(unexpected)}")

            print(f"[train] ✓ Successfully loaded pretrained encoder for fine-tuning")
        else:
            print(f"[train] ERROR: Pretrained encoder file not found: {pretrained_encoder_path}")
            print(f"[train] Starting with randomly initialized encoder")

    #############################
    # TRAINING LOOP STARTS HERE #
    #############################

    writer = SummaryWriter(filename_suffix='__'+cfg_fname, log_dir=param["log_dir"])
    decoder.to(device)

    # transformations
    # IMPORTANT: Apply geometric transforms FIRST, then Pad LAST to ensure consistent batch sizes
    # Scale augmentation is always applied, then randomly choose one of the other geometric transforms
    scale_tf = RandomScale(scale_range=(0.5, 1.5))  # Scale augmentation: 50% to 130% of original size
    other_geo_tfs = transforms.RandomChoice([
        Rotate(angle=30), 
        RandomHorizontalFlip(), 
        RandomVerticalFlip()
    ])
    geo_tfs = transforms.Compose([scale_tf, other_geo_tfs])
    
    # Color transforms: geo + color jitter + pad (pad must be LAST)
    color_tfs = [geo_tfs, transforms.ColorJitter(brightness=0.5, hue=(-0.1, 0.1), saturation=0.5), Pad(size=param["input_size"])]
    color_tf = transforms.Compose(color_tfs)
    # Default transforms: geo + pad (pad must be LAST)
    default_tfs = [geo_tfs, Pad(size=param["input_size"])]
    default_tf = transforms.Compose(default_tfs)

    # Point Cloud Transforms (config-gated, default on for backward compat)
    if bool(param.get("use_pcd_aug", True)):
        _rot_angle = float(param.get("pcd_rotate_angle", 180))
        _rot_axes = str(param.get("pcd_rotate_axes", "y"))
        _jitter_sigma = float(param.get("pcd_jitter_sigma", 0.005))
        _jitter_clip = float(param.get("pcd_jitter_clip", 0.02))
        pcd_tf = PCDCompose([
            PCDRandomRotate(angle_range=_rot_angle, axes=_rot_axes),
            PCDRandomJitter(sigma=_jitter_sigma, clip=_jitter_clip)
        ])
        print(f"[train] PCD augmentation: RandomRotate(±{_rot_angle:.0f}°, axes='{_rot_axes}') + RandomJitter(σ={_jitter_sigma}, clip={_jitter_clip})")
    else:
        pcd_tf = PCDCompose([])
        print(f"[train] PCD augmentation DISABLED (use_pcd_aug=False)")

    # Pre-padding occlusion (applied to REAL points BEFORE replicate-padding)
    # Enabled when `occlusion_pre_padding: true` in config. Prevents the dropped
    # points from being re-added via replication to pad_size.
    _pc_encoders = ['point_cloud', 'point_cloud_maxmean', 'point_cloud_large',
                    'point_cloud_maxmean_large', 'point_cloud_maxmean_large_masked',
                    'pointnet2', 'foldnet']
    _use_prepad_occ = (bool(param.get("occlusion_pre_padding", False))
                       and bool(param.get("use_occlusion_augmentation", False))
                       and param.get("occlusion_augmentation_mode", "patch") == "random"
                       and param['encoder'] in _pc_encoders)
    pre_pad_pcd_tf = None
    if _use_prepad_occ:
        import random as _rand
        _pp_max_occ = float(param.get("max_occlusion_frac", 0.5))
        _pp_min_occ = float(param.get("min_occlusion_frac", 0.0))
        _pp_min_pts = int(param.get("occlusion_min_points", 64))
        _pp_min_n_real = int(param.get("occlusion_min_n_real", 0))
        _pp_prob = float(param.get("occlusion_aug_prob", 1.0))
        if _pp_min_occ > _pp_max_occ:
            raise ValueError(f"min_occlusion_frac ({_pp_min_occ}) > max_occlusion_frac ({_pp_max_occ})")
        def pre_pad_pcd_tf(pts):  # noqa: F811
            n = pts.shape[0]
            if _pp_min_n_real > 0 and n < _pp_min_n_real:
                return pts
            if _pp_prob < 1.0 and _rand.random() > _pp_prob:
                return pts
            if n <= _pp_min_pts:
                return pts
            removal_frac = _rand.uniform(_pp_min_occ, _pp_max_occ)
            n_keep = max(int(n * (1.0 - removal_frac)), _pp_min_pts)
            if n_keep >= n:
                return pts
            idx = np.random.permutation(n)[:n_keep]
            return pts[idx]
        print(f"[train] Pre-padding occlusion ENABLED: min_frac={_pp_min_occ}, max_frac={_pp_max_occ}, "
              f"min_pts={_pp_min_pts}, min_n_real={_pp_min_n_real}, prob={_pp_prob}")

    # Enable sdf_loss in dataloader if 3D_loss is enabled (chamfer_loss removed)
    use_target_pcd = param.get("3D_loss", False)
    
    # Determine if PCD normalization should be enabled:
    # - Default: False for PointCloudEncoder/FoldNet (preserving scale is crucial for volume estimation)
    # - Default: True for Minkowski (voxelization prefers normalized coords or explicit scaling)
    # - Config can override via 'normalize_pcd'
    default_norm_pcd = param['encoder'] not in ['point_cloud', 'point_cloud_maxmean', 'point_cloud_maxmean_masked', 'point_cloud_large', 'point_cloud_maxmean_large', 'point_cloud_maxmean_large_masked', 'pointnet2', 'foldnet']
    normalize_pcd_flag = bool(param.get('normalize_pcd', default_norm_pcd))
    print(f"[train] PCD Normalization: {normalize_pcd_flag} (Encoder: {param['encoder']})")

    cl_dataset = MaskedCameraLaserData(data_source=param["data_dir"],
                                        tf=default_tf,
                                        color_tf = color_tf,
                                        pcd_tf=pcd_tf, 
                                        pretrain=pretrain,
                                        pad_size=param["input_size"],
                                        detection_input=param["detection_input"],
                                        normalize_depth=param["normalize_depth"],
                                        depth_min=param["depth_min"],
                                        depth_max=param["depth_max"],
                                        supervised_3d=param["supervised_3d"],
                                        sdf_loss=use_target_pcd,
                                        grid_density=param["grid_density"],
                                        split='train',
                                        overfit=overfit,
                                        species=param["species"],
                                        use_phenobox_pcd=bool(param.get("use_phenobox_pcd", False)),
                                        phenobox_subdir=param.get("phenobox_subdir", "phenobox/pointclouds"),
                                        use_orbbec_pcd=bool(param.get("use_orbbec_pcd", False)),
                                        orbbec_pcd_suffix=param.get("orbbec_pcd_suffix", "_preprocessed"),
                                        split_json_path=param.get("split_json", None),
                                        require_dataset_json=param.get("require_dataset_json", True),
                                        normalize_pcd=normalize_pcd_flag,
                                        partial_pcd_count=param.get("partial_pcd_count", None),
                                        pre_pad_pcd_tf=pre_pad_pcd_tf,
                                        temporal_loading=bool(param.get("temporal_loading", False)),
                                        vol_map_csv=param.get("vol_map_csv", None),
                                        gt_vol_similar_thr=float(param.get("gt_vol_similar_thr", 0.05)),
                                        caliper_weak_supervision=bool(param.get("caliper_weak_supervision", False)),
                                        caliper_correct_proxy=bool(param.get("caliper_correct_proxy", False)),
                                        gt_volume_csv=param.get("gt_volume_csv", None),
                                        use_pcd_render_as_rgbd=bool(param.get("use_pcd_render_as_rgbd", False)),
                                        render_size=int(param.get("render_size", 304)),
                                        render_splat_radius=int(param.get("render_splat_radius", 3)),
                                        cache_rendered_rgbd=bool(param.get("cache_rendered_rgbd", False)),
                                        cache_pcds_in_ram=bool(param.get("cache_pcds_in_ram", False)),
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

    # KL / DistMatch target stats: precompute mean/var (per-dim) from supervision latents
    if param.get("kl_divergence", False) or param.get("dist_match", False):
        _kl_lats = list(getattr(cl_dataset, "latents_dict", {}).values())
        if not _kl_lats:
            raise RuntimeError("[KL/DistMatch] cl_dataset.latents_dict is empty — cannot compute target stats")
        _kl_stack = torch.stack([
            (l.detach().squeeze().cpu() if isinstance(l, torch.Tensor) else torch.tensor(l).squeeze().cpu())
            for l in _kl_lats
        ]).detach()
        cl_dataset.target_mean = _kl_stack.mean(dim=0).detach()
        cl_dataset.target_var = _kl_stack.var(dim=0, unbiased=False).detach()
        print(f"[target_stats] n={len(_kl_stack)} dim={_kl_stack.shape[1]} "
              f"mean.norm={cl_dataset.target_mean.norm():.4f} "
              f"var.mean={cl_dataset.target_var.mean():.6f}")

    # Load pre-computed scaled latents for scale augmentation
    scaled_latents = {}
    scaled_latents_path = param.get("scaled_latents_path", None)
    if scaled_latents_path and os.path.exists(scaled_latents_path):
        _sl_data = np.load(scaled_latents_path)
        scaled_latents = {k: torch.from_numpy(_sl_data[k]).float() for k in _sl_data.files}
        print(f"[train] Loaded {len(scaled_latents)} scaled latents from {scaled_latents_path}")
    scale_aug_prob = float(param.get("scale_aug_prob", 0.0))
    scale_aug_pcts = param.get("scale_factors", [70, 75, 80, 85, 90, 95])
    scale_augment_all = bool(param.get("scale_augment_all", False))

    # Wrap dataset with deterministic scale expansion if requested
    bin_uniform_sampling = bool(param.get("bin_uniform_sampling", False))
    train_dataset = cl_dataset
    if bin_uniform_sampling and scaled_latents:
        # Bin-uniform sampling needs vol_last_per_file from the underlying dataset
        _vol_last = getattr(cl_dataset, 'vol_last_per_file', None)
        if _vol_last is None or all(v <= 0 for v in _vol_last):
            print(f"[train] WARNING: bin_uniform_sampling requested but vol_last_per_file unavailable — falling back to scale_aug_prob mode")
        else:
            bin_edges = param.get("bin_edges", [0, 100, 200, 300, 400, 500, 1000])
            bin_target = int(param.get("bin_target_count", 200))
            train_dataset = BinBalancedDataset(
                base_dataset=cl_dataset,
                scaled_latents=scaled_latents,
                scale_pcts=scale_aug_pcts,
                vol_last_per_file=_vol_last,
                bin_edges=bin_edges,
                target_count_per_bin=bin_target,
                size_inverse_sampling=bool(param.get("size_inverse_sampling", False)),
                size_inverse_alpha=float(param.get("size_inverse_alpha", 0.5)),
                size_inverse_vol_ref=float(param.get("size_inverse_vol_ref", 300.0)),
                cap_aug_to_max_natural=bool(param.get("cap_aug_to_max_natural", False)),
                cap_aug_seed=int(param.get("cap_aug_seed", 42)),
                density_scale_aug=bool(param.get("density_scale_aug", False)),
                density_scale_exponent=float(param.get("density_scale_exponent", 2.0)),
                density_scale_min_pts=int(param.get("density_scale_min_pts", 64)),
            )
            print(f"[train] BinUniformSampling enabled: total={len(train_dataset)} samples/epoch (size_inverse applied to NATURAL only)")
    elif scale_augment_all and scaled_latents:
        train_dataset = ScaleExpandedDataset(cl_dataset, scaled_latents, scale_aug_pcts)
        print(f"[train] Scale augment ALL: {len(cl_dataset)} x {1+len(scale_aug_pcts)} scales = {len(train_dataset)} samples/epoch")
    elif scaled_latents and scale_aug_prob > 0:
        print(f"[train] Scale augmentation (random): prob={scale_aug_prob}, scales={scale_aug_pcts}")

    # Use custom collate when 3D_loss is enabled OR when using point cloud encoders with phenobox data (variable-length point clouds)
    use_phenobox = bool(param.get("use_phenobox_pcd", False))
    is_point_cloud_encoder = param['encoder'] in ['point_cloud', 'point_cloud_maxmean', 'point_cloud_maxmean_masked', 'point_cloud_large', 'point_cloud_maxmean_large', 'point_cloud_maxmean_large_masked', 'pointnet2', 'foldnet', 'minkowski']
    collate_fn = collate_fn_variable_pcd if (use_target_pcd or (use_phenobox and is_point_cloud_encoder)) else None

    # P-sample oversampling: assign higher sampling weight to P samples
    p_oversample = float(param.get("p_oversample_factor", 1.0))
    sampler = None
    # When bin_uniform_sampling is on, the dataset handles weighting internally.
    # Force shuffle=True with no external sampler so DataLoader iterates round-robin.
    _bin_uniform_active = isinstance(train_dataset, BinBalancedDataset)
    if _bin_uniform_active:
        print(f"[train] BinBalancedDataset active → skipping WeightedRandomSampler / has_gt oversampling")

    # Caliper weak supervision: balance strong (last_date, laser GT) and
    # weak (intermediate, caliper proxy) samples per batch.
    # target_strong_ratio: float in (0,1) sets explicit ratio via WeightedRandomSampler.
    # target_strong_ratio <= 0 (or null) disables the sampler — falls back to natural
    # shuffle, sampling proportional to file count (intermediate weak ~9x more frequent).
    caliper_weak_active = bool(param.get("caliper_weak_supervision", False)) and bool(getattr(cl_dataset, 'is_weak_per_file', None))
    target_strong_ratio = float(param.get("target_strong_ratio", 0.0) or 0.0)
    if caliper_weak_active and not _bin_uniform_active and 0.0 < target_strong_ratio < 1.0:
        from torch.utils.data import WeightedRandomSampler
        is_weak_pf = cl_dataset.is_weak_per_file
        n_weak = sum(1 for x in is_weak_pf if x)
        n_strong = len(is_weak_pf) - n_weak
        if n_strong > 0 and n_weak > 0:
            tr = max(0.01, min(0.99, target_strong_ratio))
            w_strong = tr / n_strong
            w_weak = (1.0 - tr) / n_weak
        else:
            w_strong = w_weak = 1.0
        weights = [w_weak if iw else w_strong for iw in is_weak_pf]
        sampler = WeightedRandomSampler(weights, num_samples=(n_strong + n_weak), replacement=True)
        shuffle = False
        print(f"[train] CaliperWeakSupervision sampler: target_strong_ratio={tr}  "
              f"n_strong={n_strong}  n_weak={n_weak}  w_strong={w_strong:.5f}  w_weak={w_weak:.5f}  "
              f"epoch_samples={n_strong + n_weak}")
    elif caliper_weak_active:
        is_weak_pf = cl_dataset.is_weak_per_file
        n_weak = sum(1 for x in is_weak_pf if x)
        n_strong = len(is_weak_pf) - n_weak
        print(f"[train] CaliperWeakSupervision active, sampler DISABLED (target_strong_ratio<=0). "
              f"Natural ratio: n_strong={n_strong}, n_weak={n_weak}, weak_ratio={n_weak/(n_strong+n_weak):.2%}")

    # has_gt oversampling: when temporal_loading is on AND vol_map is built, oversample
    # the frames flagged has_gt so every batch has SuperLoss signal.
    has_gt_target_ratio = float(param.get("has_gt_target_ratio", 0.0))
    # resolve the underlying dataset (unwrap ScaleExpandedDataset if needed)
    _base_dataset = train_dataset
    if hasattr(_base_dataset, 'base'):
        _base_dataset = _base_dataset.base
    has_gt_per_file = getattr(_base_dataset, 'has_gt_per_file', None)
    # Size-inverse sampling: up-weight small fruits to improve small-fruit coverage.
    # weight *= (vol_ref / vol_last) ** alpha   for files where vol_last > 0.
    size_inverse_sampling = bool(param.get("size_inverse_sampling", False))
    size_inverse_alpha = float(param.get("size_inverse_alpha", 0.5))
    size_inverse_vol_ref = float(param.get("size_inverse_vol_ref", 300.0))
    vol_last_per_file = getattr(_base_dataset, 'vol_last_per_file', None)
    if (not _bin_uniform_active and not caliper_weak_active and has_gt_target_ratio > 0.0 and has_gt_per_file is not None
            and hasattr(_base_dataset, 'files')):
        from torch.utils.data import WeightedRandomSampler
        n_pos = sum(1 for x in has_gt_per_file if x)
        n_neg = len(has_gt_per_file) - n_pos
        if n_pos > 0 and n_neg > 0:
            # Solve target_ratio = (n_pos*w)/(n_pos*w + n_neg*1)
            # → w = target_ratio * n_neg / ((1 - target_ratio) * n_pos)
            tr = max(0.01, min(0.99, has_gt_target_ratio))
            w_pos = tr * n_neg / ((1.0 - tr) * n_pos)
        else:
            w_pos = 1.0
        weights = []
        for idx, f in enumerate(_base_dataset.files):
            is_p = any(p.startswith('P') and len(p) > 1 and p[1:].split('-')[0].isdigit() for p in f.split('/'))
            w_p = p_oversample if is_p else 1.0
            w_gt = w_pos if has_gt_per_file[idx] else 1.0
            w_size = 1.0
            if size_inverse_sampling and vol_last_per_file is not None:
                vl = vol_last_per_file[idx]
                if vl > 0:
                    w_size = (size_inverse_vol_ref / vl) ** size_inverse_alpha
            weights.append(w_p * w_gt * w_size)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
        extra = f"  size_inverse={size_inverse_sampling} (alpha={size_inverse_alpha}, vol_ref={size_inverse_vol_ref})" if size_inverse_sampling else ""
        print(f"[train] has_gt oversampling: target_ratio={has_gt_target_ratio}  "
              f"n_pos={n_pos}  n_neg={n_neg}  weight_pos={w_pos:.3f}  "
              f"(combined with p_oversample={p_oversample}){extra}")
        if size_inverse_sampling and vol_last_per_file is not None:
            import numpy as _np
            arr = _np.array([w for w in weights])
            vls = _np.array([v for v in vol_last_per_file])
            print(f"  weight stats: min={arr.min():.3f}  max={arr.max():.3f}  median={_np.median(arr):.3f}")
            print(f"  vol_last stats: min={vls[vls>0].min():.1f}  max={vls[vls>0].max():.1f}  median={_np.median(vls[vls>0]):.1f}")
    elif (not _bin_uniform_active and not caliper_weak_active and (p_oversample > 1.0 or size_inverse_sampling) and hasattr(train_dataset, 'files')):
        from torch.utils.data import WeightedRandomSampler
        weights = []
        _si_applied = 0
        for idx, f in enumerate(train_dataset.files):
            # Detect P samples from path: .../PXXX/orbbec/... or .../PXXX/phenobox/...
            parts = f.split('/')
            is_p = any(p.startswith('P') and len(p) > 1 and p[1:].split('-')[0].isdigit() for p in parts)
            w_p = p_oversample if is_p else 1.0
            w_size = 1.0
            if size_inverse_sampling and vol_last_per_file is not None and idx < len(vol_last_per_file):
                vl = vol_last_per_file[idx]
                if vl > 0:
                    w_size = (size_inverse_vol_ref / vl) ** size_inverse_alpha
                    _si_applied += 1
            weights.append(w_p * w_size)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False  # sampler and shuffle are mutually exclusive
        extra = f"  size_inverse={size_inverse_sampling} (alpha={size_inverse_alpha}, vol_ref={size_inverse_vol_ref}, applied={_si_applied}/{len(weights)})" if size_inverse_sampling else ""
        print(f"[train] sampler: p_oversample={p_oversample}, P-files={sum(1 for i,f in enumerate(train_dataset.files) if any(p.startswith('P') and len(p) > 1 and p[1:].split('-')[0].isdigit() for p in f.split('/')))}, total={len(weights)}{extra}")
        if size_inverse_sampling and vol_last_per_file is not None:
            import numpy as _np
            arr = _np.array([w for w in weights])
            vls = _np.array([v for v in vol_last_per_file])
            pos = vls[vls > 0]
            print(f"  weight stats: min={arr.min():.3f}  max={arr.max():.3f}  median={_np.median(arr):.3f}")
            if len(pos) > 0:
                print(f"  vol_last stats: min={pos.min():.1f}  max={pos.max():.1f}  median={_np.median(pos):.1f}")

    _train_nw = int(param.get("train_num_workers", 0))
    dataset = DataLoader(
        train_dataset, batch_size=param["batch_size"], shuffle=shuffle, sampler=sampler,
        drop_last=True, collate_fn=collate_fn,
        num_workers=_train_nw,
        persistent_workers=(_train_nw > 0),
        pin_memory=(_train_nw > 0),
    )
    if _train_nw > 0:
        print(f"[train] DataLoader: num_workers={_train_nw}, persistent_workers=True, pin_memory=True")

    # Setup optimizer
    # Decoder is frozen by default (pretrained DeepSDF)
    # Encoder is what we train (maps observations to latent codes)
    if update_decoder and 'decoder_lr' in param:
        # Special case: Fine-tune both encoder AND decoder with different learning rates
        optim = torch.optim.Adam([
            {'params': encoder.parameters(), 'lr': param["lr"]},
            {'params': decoder.parameters(), 'lr': param["decoder_lr"]}
        ], weight_decay=1e-6)
        print(f"[train] Training BOTH encoder (lr={param['lr']}) and decoder (lr={param['decoder_lr']})")
    elif 'vol_head_lr' in param:
        # Two-rate finetune: keep pretrained encoder backbone slow (preserve features),
        # train vol_head fast (random-init head needs to catch up).
        backbone = [p for n, p in encoder.named_parameters() if not n.startswith('vol_head.')]
        volhead  = [p for n, p in encoder.named_parameters() if n.startswith('vol_head.')]
        optim = torch.optim.Adam([
            {'params': backbone, 'lr': param["lr"]},
            {'params': volhead,  'lr': param["vol_head_lr"]},
        ], weight_decay=1e-4)
        print(f"[train] Training encoder backbone (lr={param['lr']}) + vol_head (lr={param['vol_head_lr']}), decoder frozen")
    else:
        # Normal case: Train encoder only, decoder stays frozen
        # Increased weight decay to 1e-4 for better regularization on small datasets
        optim = torch.optim.Adam(encoder.parameters(), lr=param["lr"], weight_decay=1e-4)
        print(f"[train] Training encoder (lr={param['lr']}), decoder frozen (pretrained DeepSDF)")

    # Optional resume: restore model/optimizer state and continue from next epoch
    start_epoch = 0
    resume_ckpt_path = param.get("resume_checkpoint")
    if resume_ckpt_path:
        resume_ckpt_path = str(resume_ckpt_path)
        if os.path.isfile(resume_ckpt_path):
            print(f"[train] Resuming from checkpoint: {resume_ckpt_path}")
            ckpt = torch.load(resume_ckpt_path, map_location=device)
            if "encoder_state_dict" in ckpt:
                encoder.load_state_dict(ckpt["encoder_state_dict"])
            if "decoder_state_dict" in ckpt:
                decoder.load_state_dict(ckpt["decoder_state_dict"])
            if "optimizer_state_dict" in ckpt:
                try:
                    optim.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as ex:
                    print(f"[train] Warning: failed to load optimizer state (continuing with fresh optimizer). Reason: {ex}")
            if "epoch" in ckpt:
                try:
                    start_epoch = int(ckpt["epoch"]) + 1
                except Exception:
                    start_epoch = 0
            print(f"[train] ✓ Resume start_epoch={start_epoch}")
        else:
            print(f"[train] Warning: resume_checkpoint not found: {resume_ckpt_path} (starting from epoch 0)")

    # When resuming, reset scheduler from epoch 0 so LR restarts fresh
    sched_last_epoch = start_epoch - 1 if not resume_ckpt_path else -1
    lr_sched_type = str(param.get("lr_scheduler", "exponential")).lower()
    if lr_sched_type == "cosine_warm_restarts":
        T_0 = int(param.get("lr_T0", 50))
        T_mult = int(param.get("lr_T_mult", 2))
        eta_min = float(param.get("lr_eta_min", 1e-7))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=sched_last_epoch
        )
        print(f"[train] LR scheduler: CosineAnnealingWarmRestarts(T_0={T_0}, T_mult={T_mult}, eta_min={eta_min})")
    elif lr_sched_type == "cosine":
        T_max = int(param.get("lr_T_max", param["epoch"]))
        eta_min = float(param.get("lr_eta_min", 1e-6))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=T_max, eta_min=eta_min, last_epoch=sched_last_epoch
        )
        print(f"[train] LR scheduler: CosineAnnealingLR(T_max={T_max}, eta_min={eta_min})")
    else:
        gamma = float(param.get("lr_gamma", 0.97))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma, last_epoch=sched_last_epoch)
        print(f"[train] LR scheduler: ExponentialLR(gamma={gamma})")
    if resume_ckpt_path:
        print(f"[train] LR scheduler reset (fresh start from lr={param['lr']})")

    print('\ncfg: ', json.dumps(param, indent=4), '\n')
    print(encoder)
    print(decoder)

    # import ipdb; ipdb.set_trace()
    n_iter = 0  # used for tensorboard
    try:
        species = str(param.get('species', '')).lower()
        if 'kmelon' in species:
            gt_path = "./data/3DKmelonDemo/ground_truth.csv"
            gt_aug_path = "./data/3DKmelonDemo/ground_truth_aug.csv"
        else:
            gt_path = "./data/3DPotatoTwinDemo/ground_truth.csv"
            gt_aug_path = None # No aug for potato yet?

        df = pd.read_csv(gt_path)
        
        if gt_aug_path and os.path.exists(gt_aug_path):
            print(f"[train] Loading augmented ground truth from {gt_aug_path}")
            df_aug = pd.read_csv(gt_aug_path)
            # Rename volume_metashape to volume_ml to match main GT (if needed)
            if 'volume_metashape' in df_aug.columns and 'volume_ml' not in df_aug.columns:
                df_aug.rename(columns={'volume_metashape': 'volume_ml'}, inplace=True)
            
            # Concatenate
            df = pd.concat([df, df_aug], ignore_index=True)
            print(f"[train] Merged GT size: {len(df)}")
            
    except Exception as ex:
        print('Failed to load ground truth CSV based on species; using default. Reason:', ex)
        df = pd.read_csv("./data/3DKmelonDemo/ground_truth.csv")

    for e in range(start_epoch, param["epoch"]):
        # Track average training loss per epoch (logged to TensorBoard)
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for idx, item in enumerate(iter(dataset)):

            # import ipdb;ipdb.set_trace()
            n_iter += 1  # for tensorboard
            logging_string = 'epoch: {}/{} -- iteration {}/{}'.format(e+1, param["epoch"], idx, len(dataset))

            optim.zero_grad()
            loss = 0
            _s_pct_batch = 100  # tracks batch-level scale aug factor for downstream losses (volume_head etc.)
            # Per-sample scale list for downstream losses when bin_uniform_sampling provides mixed scales.
            _scale_pct_per_sample = item.get('scale_pct', None)
            if _scale_pct_per_sample is not None:
                if hasattr(_scale_pct_per_sample, 'tolist'):
                    _scale_pct_per_sample = _scale_pct_per_sample.tolist()
                elif not isinstance(_scale_pct_per_sample, list):
                    _scale_pct_per_sample = [int(_scale_pct_per_sample)]

            # Density-coupling sanity print: first 3 iters of epoch 1 only.
            if e == start_epoch and idx < 3 and bool(param.get("density_scale_aug", False)) \
                    and 'partial_pcd_real_mask' in item and 'scale_pct' in item:
                _rm = item['partial_pcd_real_mask']
                _sp = _scale_pct_per_sample if _scale_pct_per_sample is not None else [100]*_rm.shape[0]
                _n_real_b = _rm.sum(dim=-1).tolist() if _rm.dim() == 2 else [int(_rm.sum().item())]
                print(f"[density-sanity] e=1 it={idx}  per-sample (scale_pct, n_real): {list(zip(_sp, _n_real_b))}")

            # Scale augmentation (applied BEFORE occlusion so occlusion acts on the scaled PCD)
            # Skip when scale_augment_all OR bin_uniform is used (scaling already done in dataset wrapper)
            if (not scale_augment_all and not _bin_uniform_active
                    and scaled_latents and scale_aug_prob > 0
                    and random.random() < scale_aug_prob
                    and param['encoder'] in ['point_cloud', 'point_cloud_maxmean', 'point_cloud_maxmean_masked', 'point_cloud_large', 'point_cloud_maxmean_large', 'point_cloud_maxmean_large_masked', 'pointnet2', 'foldnet']):
                s_pct = random.choice(scale_aug_pcts)
                s_factor = s_pct / 100.0
                _s_pct_batch = s_pct

                # Scale PCD
                if isinstance(item['partial_pcd'], list):
                    item['partial_pcd'] = [p * s_factor for p in item['partial_pcd']]
                else:
                    item['partial_pcd'] = item['partial_pcd'] * s_factor

                # Replace target latent per sample
                new_latents = item['latent'].clone()
                for b in range(new_latents.shape[0]):
                    fid = item['fruit_id'][b]
                    key = f"{fid}_s{s_pct}"
                    if key in scaled_latents:
                        new_latents[b] = scaled_latents[key]
                item['latent'] = new_latents

                # Density-scale augmentation: physically couple coord scaling with
                # reduced real-point count (smaller fruit => smaller projected area =>
                # fewer depth pts). s^2 for surface (depth camera), s^3 for volume.
                if (bool(param.get("density_scale_aug", False))
                        and not isinstance(item['partial_pcd'], list)
                        and 'partial_pcd_real_mask' in item):
                    ds_exp = float(param.get("density_scale_exponent", 2.0))
                    ds_min_pts = int(param.get("density_scale_min_pts", 64))
                    keep_frac = s_factor ** ds_exp
                    B, N, _ = item['partial_pcd'].shape
                    real_mask_batch = item['partial_pcd_real_mask']
                    new_pcd = item['partial_pcd'].clone()
                    new_mask = real_mask_batch.clone()
                    for b in range(B):
                        m = real_mask_batch[b]
                        n_real = int(m.sum().item())
                        if n_real <= ds_min_pts:
                            continue
                        n_keep = max(int(n_real * keep_frac), ds_min_pts)
                        n_keep = min(n_keep, n_real)
                        if n_keep >= n_real:
                            continue
                        real_idx = torch.where(m)[0]
                        perm = torch.randperm(n_real, device=real_idx.device)
                        keep_idx = real_idx[perm[:n_keep]]
                        kept_pts = item['partial_pcd'][b, keep_idx]
                        # Repack: real first, then replicate to pad_size
                        fill_idx = torch.randint(0, n_keep, (N - n_keep,), device=kept_pts.device)
                        new_pcd[b, :n_keep] = kept_pts
                        new_pcd[b, n_keep:] = kept_pts[fill_idx]
                        nm = torch.zeros_like(m)
                        nm[:n_keep] = True
                        new_mask[b] = nm
                    item['partial_pcd'] = new_pcd
                    item['partial_pcd_real_mask'] = new_mask

            # Save un-augmented copy for consistency loss (used by Variant B).
            # Need this BEFORE any aug modifies item['partial_pcd'].
            _use_consistency = bool(param.get("use_consistency_loss", False))
            _consistency_pcd_clean = None
            if _use_consistency and isinstance(item['partial_pcd'], torch.Tensor):
                _consistency_pcd_clean = item['partial_pcd'].clone()

            # Apply occlusion augmentation (point cloud encoders only)
            # Skipped when pre-padding occlusion is active (applied in dataloader on real points).
            # frac ~ U(0.0, max_occlusion_frac) per sample, matching temporal model augmentation
            if ((not _use_prepad_occ)
                    and bool(param.get("use_occlusion_augmentation", False))
                    and param['encoder'] in ['point_cloud', 'point_cloud_maxmean', 'point_cloud_maxmean_masked', 'point_cloud_large', 'point_cloud_maxmean_large', 'point_cloud_maxmean_large_masked', 'pointnet2', 'foldnet']):
                max_occ = float(param.get("max_occlusion_frac", 0.3))
                occ_min_pts = int(param.get("occlusion_min_points", 64))
                occ_mode = param.get("occlusion_augmentation_mode", "patch")
                # Variant A: dynamic max_frac cap so post-occlusion point count
                # stays at or above this floor for all samples (regardless of
                # original PCD size). Set to 0 / unset → behaves as before.
                _occ_min_n_post = int(param.get("occlusion_min_n_post", 0))
                # Variant A: in 'mixed' mode each sample randomly gets either
                # patch or random occlusion (split by `random_aug_prob`).
                _random_prob = float(param.get("random_aug_prob", 0.5))
                # Per-sample probability of applying ANY occlusion aug
                # (defaults to 1.0 = always when above min_n_real).
                _occ_apply_prob = float(param.get("occlusion_apply_prob", 1.0))

                def _pick_fn(m):
                    if m == "patch":   return apply_patch_occlusion
                    if m == "random":  return apply_random_occlusion
                    return apply_plane_occlusion
                _fn_static = _pick_fn(occ_mode) if occ_mode != "mixed" else None

                # Only apply occ to PCDs with at least this many real points
                _occ_min_n_real = int(param.get("occlusion_min_n_real", 0))

                import random as _r
                def _apply_occ_selective(p):
                    n_real = int((p.norm(dim=-1) > 1e-8).sum().item())
                    if _occ_min_n_real > 0 and n_real < _occ_min_n_real:
                        return p
                    if _occ_apply_prob < 1.0 and _r.random() > _occ_apply_prob:
                        return p
                    # Dynamic max_frac cap: ensure n_post >= occ_min_n_post.
                    eff_max = max_occ
                    if _occ_min_n_post > 0 and n_real > _occ_min_n_post:
                        eff_max = min(max_occ, 1.0 - (_occ_min_n_post / n_real))
                        if eff_max <= 0:
                            return p
                    if occ_mode == "mixed":
                        fn = apply_random_occlusion if _r.random() < _random_prob else apply_patch_occlusion
                    else:
                        fn = _fn_static
                    return fn(p, eff_max, occ_min_pts)
                if isinstance(item['partial_pcd'], list):
                    item['partial_pcd'] = [_apply_occ_selective(p) for p in item['partial_pcd']]
                else:
                    # (B, N, 3) — apply independently per sample
                    item['partial_pcd'] = torch.stack([
                        _apply_occ_selective(item['partial_pcd'][b])
                        for b in range(item['partial_pcd'].shape[0])
                    ])

            # unpacking inputs / encoding
            # Initialise variables that the point-cloud branch sets so that the
            # image-encoder branch can re-use them via shared downstream code.
            log_vol_pred_batch = None
            log_scale_pred_batch = None
            _use_nll_super = bool(param.get('nll_super_loss', False))
            _has_aux = hasattr(encoder, 'encode_with_aux')
            _vol_on = bool(param.get('volume_head', False))
            _is_masked_enc = param['encoder'] in ('point_cloud_maxmean_masked', 'point_cloud_maxmean_large_masked')
            _weak_vol_mode = str(param.get('weak_vol_loss_mode', 'mlp')).lower()
            _vol_head_detach = (_weak_vol_mode == 'softvol')
            if param['encoder'] not in ['point_cloud', 'point_cloud_maxmean', 'point_cloud_maxmean_masked', 'point_cloud_large', 'point_cloud_maxmean_large', 'point_cloud_maxmean_large_masked', 'pointnet2', 'foldnet', 'minkowski']:
                encoder_input = torch.cat((item['rgb'], item['depth']), 1).to(device)
                latent_batch_unnormd = encoder(encoder_input)
            elif param['encoder'] == 'minkowski':
                # Handle both stacked tensors and lists of tensors (from custom collate)
                if isinstance(item['partial_pcd_world'], list):
                    partial_points = [p.to(device) for p in item['partial_pcd_world']]
                    bbox_centers = [c.to(device) for c in item['bbox_center_tensor']]
                    bbox_scales = [s.to(device) for s in item['bbox_scale_tensor']]
                else:
                    partial_points = item['partial_pcd_world'].to(device)
                    bbox_centers = item['bbox_center_tensor'].to(device)
                    bbox_scales = item['bbox_scale_tensor'].to(device)
                latent_batch_unnormd = encoder(partial_points, bbox_centers, bbox_scales)
            else:
                if isinstance(item['partial_pcd'], list):
                    # Handle list of tensors with variable sizes - resample to fixed size
                    target_points = param["input_size"]  # Use input_size as target number of points
                    resampled_pcds = []
                    for p in item['partial_pcd']:
                        num_points = p.shape[0]
                        if num_points > target_points:
                            # Downsample
                            indices = torch.randperm(num_points)[:target_points]
                            resampled = p[indices]
                        elif num_points < target_points:
                            # Upsample with replacement
                            indices = torch.randint(0, num_points, (target_points,))
                            resampled = p[indices]
                        else:
                            resampled = p
                        resampled_pcds.append(resampled.permute(1, 0))  # Shape: [3, N]
                    encoder_input = torch.stack(resampled_pcds, dim=0).to(device)  # Shape: [B, 3, N]
                else:
                    encoder_input = item['partial_pcd'].permute(0, 2, 1).to(device)
                # Pass real-point mask to encoders that accept it (mask-aware variants)
                log_vol_pred_batch = None
                log_scale_pred_batch = None  # heteroscedastic NLL per-dim log-scale (Laplace b or Gaussian σ)
                _use_nll_super = bool(param.get('nll_super_loss', False))
                _has_aux = hasattr(encoder, 'encode_with_aux')
                _vol_on = bool(param.get('volume_head', False))
                _is_masked_enc = param['encoder'] in ('point_cloud_maxmean_masked', 'point_cloud_maxmean_large_masked')
                # When weak_vol_loss_mode == "softvol", encoder is supervised by softvol (decoder→sigmoid integration);
                # vol_head trains as parallel inference predictor with gradients detached from backbone.
                _weak_vol_mode = str(param.get('weak_vol_loss_mode', 'mlp')).lower()
                _vol_head_detach = (_weak_vol_mode == 'softvol')
                if _use_nll_super and _has_aux:
                    if _is_masked_enc and 'partial_pcd_real_mask' in item:
                        real_mask = item['partial_pcd_real_mask'].to(device)
                        aux = encoder.encode_with_aux(encoder_input, mask=real_mask)
                    else:
                        aux = encoder.encode_with_aux(encoder_input)
                    latent_batch_unnormd = aux['latent']
                    log_vol_pred_batch = aux.get('log_vol')
                    log_scale_pred_batch = aux.get('log_scale')
                elif _is_masked_enc and 'partial_pcd_real_mask' in item:
                    real_mask = item['partial_pcd_real_mask'].to(device)
                    if _vol_on and hasattr(encoder, 'encode_with_vol'):
                        latent_batch_unnormd, log_vol_pred_batch = encoder.encode_with_vol(
                            encoder_input, mask=real_mask, detach_for_vol=_vol_head_detach)
                    else:
                        latent_batch_unnormd = encoder(encoder_input, mask=real_mask)
                elif _vol_on and hasattr(encoder, 'encode_with_vol'):
                    latent_batch_unnormd, log_vol_pred_batch = encoder.encode_with_vol(
                        encoder_input, detach_for_vol=_vol_head_detach)
                else:
                    latent_batch_unnormd = encoder(encoder_input)

            norms_batch = torch.linalg.norm(latent_batch_unnormd, dim=1)

            latent_batch = latent_batch_unnormd #/ norms_batch.unsqueeze(dim=1)

            # ============================================================
            # Variant B: full-vs-occluded latent CONSISTENCY loss
            # Re-encode the SAME un-augmented input under heavy patch occlusion
            # and pull the heavily-occluded latent toward the (detached) clean
            # latent. Trains the encoder to produce identity-stable latents
            # even when input is degraded into the fallback regime.
            # ============================================================
            consistency_loss_val = None
            if (_use_consistency
                    and _consistency_pcd_clean is not None
                    and param['encoder'] in ['point_cloud', 'point_cloud_maxmean',
                                             'point_cloud_maxmean_masked',
                                             'point_cloud_large',
                                             'point_cloud_maxmean_large',
                                             'point_cloud_maxmean_large_masked']):
                _cons_max_occ = float(param.get("consistency_max_occ_frac", 0.7))
                _cons_min_post = int(param.get("consistency_min_n_post", 150))
                _cons_min_pts = int(param.get("consistency_min_points", 64))
                import random as _r2
                def _heavy_occ(p):
                    n_real = int((p.norm(dim=-1) > 1e-8).sum().item())
                    if n_real <= _cons_min_pts:
                        return p
                    eff_max = _cons_max_occ
                    if _cons_min_post > 0 and n_real > _cons_min_post:
                        eff_max = min(_cons_max_occ, 1.0 - (_cons_min_post / n_real))
                        if eff_max <= 0:
                            return p
                    # Sample fraction from [0.5*eff_max, eff_max] so the heavy
                    # branch is always meaningfully more occluded than the aug
                    # branch (which spans [0, eff_max]).
                    frac = _r2.uniform(0.5 * eff_max, eff_max)
                    return apply_patch_occlusion(p, frac, _cons_min_pts) \
                        if _r2.random() < 0.5 else \
                        apply_random_occlusion(p, frac, _cons_min_pts)

                _heavy_pcd = torch.stack([
                    _heavy_occ(_consistency_pcd_clean[b])
                    for b in range(_consistency_pcd_clean.shape[0])
                ]).to(device)
                _heavy_input = _heavy_pcd.permute(0, 2, 1)
                if _is_masked_enc:
                    _heavy_mask = (_heavy_pcd.norm(dim=-1) > 1e-8).to(device)
                    _z_heavy = encoder(_heavy_input, mask=_heavy_mask)
                else:
                    _z_heavy = encoder(_heavy_input)
                # Teacher (clean): detached. Student (heavy): receives gradient.
                _lambda_cons = float(param.get("lambda_consistency", 1.0))
                consistency_loss_val = ((_z_heavy - latent_batch.detach()) ** 2).sum(dim=-1).mean()
                loss = loss + _lambda_cons * consistency_loss_val
                writer.add_scalar('Loss/Train/Consistency',
                                  _lambda_cons * consistency_loss_val.item(), n_iter)
                logging_string += f' -- loss cons: {_lambda_cons * consistency_loss_val.item():.6f}'

            # Log predicted latent statistics (helps diagnose latent collapse)
            pred_latent_mean = float(latent_batch.mean().item())
            pred_latent_std = float(latent_batch.std().item())
            writer.add_scalar('Debug/Train/PredLatentMean', pred_latent_mean, n_iter)
            writer.add_scalar('Debug/Train/PredLatentStd', pred_latent_std, n_iter)
            logging_string += f' -- pred_lat_mean: {pred_latent_mean:.6f} -- pred_lat_std: {pred_latent_std:.6f}'

            # Debug: check latent shape after encoding (especially for Minkowski)
            if n_iter == 1 and param['encoder'] == 'minkowski':
                batch_size_debug = len(item['partial_pcd']) if isinstance(item['partial_pcd'], list) else item['partial_pcd'].shape[0]
                print(f"[minkowski-debug] Batch size: {batch_size_debug}, Latent shape: {latent_batch.shape}")

            # Debug: inspect bbox spans to ensure they are non-zero
            if n_iter <= 5:
                box = tensor_dict_2_float_dict(item['bbox'])
                dx = box['xmax'] - box['xmin']
                dy = box['ymax'] - box['ymin']
                dz = box['zmax'] - box['zmin']
                print(f"[bbox-span] iter {n_iter}: ({dx:.4f}, {dy:.4f}, {dz:.4f})")

            if param.get("contrastive", False):
                fruit_ids = [list(dataset.dataset.Ks.keys()).index(fid) for fid in item['fruit_id']]
                fruit_ids = torch.Tensor(fruit_ids).to(device)
                
                # Debug: check shapes for contrastive loss
                if n_iter == 1:
                    print(f"[contrastive-debug] latent_batch shape: {latent_batch.shape}, fruit_ids shape: {fruit_ids.shape}")
                    print(f"[contrastive-debug] fruit_ids: {fruit_ids}")

                att_loss = AttRepLoss(latent_batch, fruit_ids, device)
                loss += param['lambda_attraction']*att_loss

                # logging
                writer.add_scalar('Loss/Train/Att', param['lambda_attraction']*att_loss, n_iter)
                logging_string += ' -- loss att: {}'.format(param['lambda_attraction']*att_loss.item())

            if param.get("kl_divergence", False):

                loss_kl, determinant = KLDivLoss(latent_batch, cl_dataset, device)
                loss += param['lambda_kl']*loss_kl

                # logging
                writer.add_scalar('Loss/Train/KLDiv', param['lambda_kl']*loss_kl, n_iter)
                logging_string += ' -- loss kl: {}'.format(param['lambda_kl']*loss_kl.item())
                logging_string += ' -- det: {}'.format(determinant.item())

                writer.add_scalar('Debug/Train/BatchCovDet', determinant, n_iter)

            if param.get("dist_match", False):
                loss_dist, std_ratio = DistMatchLoss(latent_batch, cl_dataset, device)
                lambda_dist = float(param.get('lambda_dist', 10.0))
                loss += lambda_dist * loss_dist
                writer.add_scalar('Loss/Train/DistMatch', lambda_dist * loss_dist, n_iter)
                writer.add_scalar('Debug/Train/StdRatio', std_ratio, n_iter)
                logging_string += ' -- loss dist: {:.6f}'.format(lambda_dist * loss_dist.item())
                logging_string += ' -- std_ratio: {:.4f}'.format(std_ratio.item())

            if param['supervised_3d']:
                lambda_super = param.get('lambda_super', 1.0)
                # Choose SuperLoss type: 'mse' (default) or 'l1'. L1 pulls toward median,
                # reducing mean-regression in multi-target fits (vs MSE which pulls toward mean).
                _super_type = str(param.get('super_loss_type', 'mse')).lower()
                # Optional heteroscedastic NLL: requires nll_super_loss=true AND log_scale_pred_batch.
                # Falls back to plain L1/MSE when log_scale unavailable.
                _ls_clamp_min = float(param.get('nll_log_scale_min', -3.0))
                _ls_clamp_max = float(param.get('nll_log_scale_max', 3.0))
                # cosine_norm loss: separates direction (cosine distance) from magnitude
                # (|‖pred‖ - ‖tgt‖|) to avoid L1/MSE collapse to median.
                _cn_lambda_norm = float(param.get('lambda_norm_in_super', 1.0))
                def _super_loss(pred, tgt, log_scale=None):
                    pred = pred.cuda(); tgt = tgt.cuda()
                    if _super_type == 'cosine_norm':
                        cos_sim = F.cosine_similarity(pred, tgt, dim=-1)
                        cos_dist = (1.0 - cos_sim).mean()
                        pred_norm = pred.norm(dim=-1)
                        tgt_norm = tgt.norm(dim=-1)
                        norm_dist = (pred_norm - tgt_norm).abs().mean()
                        return cos_dist + _cn_lambda_norm * norm_dist
                    if log_scale is None:
                        return F.l1_loss(pred, tgt) if _super_type == 'l1' else F.mse_loss(pred, tgt)
                    # Heteroscedastic NLL (per-dim log_scale aligned with latent_dim)
                    ls = log_scale.cuda().clamp(_ls_clamp_min, _ls_clamp_max)
                    diff = (pred - tgt)
                    if _super_type == 'l1':
                        # Laplace NLL: |y-μ|/b + log(b),   ls = log(b)
                        nll = diff.abs() * torch.exp(-ls) + ls
                    else:
                        # Gaussian NLL: 0.5·(y-μ)²/σ² + log(σ),   ls = log(σ)
                        nll = 0.5 * diff.pow(2) * torch.exp(-2.0 * ls) + ls
                    return nll.mean()
                # Caliper weak supervision: split batch into strong (last_date, laser GT)
                # and weak (intermediate, scaled-latent target) sub-batches and apply
                # different lambdas. Takes precedence over has_gt_mask path.
                _caliper_weak_active_loss = (bool(param.get("caliper_weak_supervision", False))
                                              and 'is_weak' in item)
                if _caliper_weak_active_loss:
                    is_weak_mask = item['is_weak'].bool().to(device)
                    strong_mask = ~is_weak_mask
                    lambda_super_weak = float(param.get('lambda_super_weak', lambda_super))
                    target_lat_dev = item['latent'].to(device)
                    ls_dev = log_scale_pred_batch
                    super_loss_total = torch.tensor(0.0, device=device)
                    n_s = int(strong_mask.sum().item()); n_w = int(is_weak_mask.sum().item())
                    if n_s > 0:
                        ls_s = ls_dev[strong_mask] if ls_dev is not None else None
                        l_s = _super_loss(latent_batch[strong_mask], target_lat_dev[strong_mask], log_scale=ls_s)
                        super_loss_total = super_loss_total + lambda_super * l_s
                        writer.add_scalar('Loss/Train/SuperLoss_strong', lambda_super * l_s, n_iter)
                    if n_w > 0:
                        ls_w = ls_dev[is_weak_mask] if ls_dev is not None else None
                        l_w = _super_loss(latent_batch[is_weak_mask], target_lat_dev[is_weak_mask], log_scale=ls_w)
                        super_loss_total = super_loss_total + lambda_super_weak * l_w
                        writer.add_scalar('Loss/Train/SuperLoss_weak', lambda_super_weak * l_w, n_iter)
                    loss += super_loss_total
                    writer.add_scalar('Loss/Train/SuperLoss', super_loss_total, n_iter)
                    logging_string += f' -- loss super: {super_loss_total.item():.4f} (strong={n_s}, weak={n_w})'
                # When temporal_loading is enabled with vol_map, only apply SuperLoss
                # on frames where vol is close to last-date vol (has_gt_latent=True).
                # Elsewhere the per-fruit target latent does not describe the input.
                use_has_gt_mask = (not _caliper_weak_active_loss
                                    and bool(param.get("temporal_loading", False))
                                    and param.get("vol_map_csv", None) is not None
                                    and 'has_gt_latent' in item)
                if use_has_gt_mask:
                    mask_cpu = item['has_gt_latent'].bool()
                    n_gt = int(mask_cpu.sum().item())
                    if n_gt > 0:
                        target_latent_masked = item['latent'][mask_cpu].to(device)
                        mask_device = mask_cpu.to(device)
                        ls_masked = log_scale_pred_batch[mask_device] if log_scale_pred_batch is not None else None
                        loss_super = _super_loss(latent_batch[mask_device], target_latent_masked, log_scale=ls_masked)
                        loss += lambda_super * loss_super
                        writer.add_scalar('Loss/Train/SuperLoss', lambda_super * loss_super, n_iter)
                        writer.add_scalar('Debug/Train/SuperLoss_n_gt', n_gt, n_iter)
                        logging_string += ' -- loss super: {} (n_gt={}/{})'.format((lambda_super * loss_super).item(), n_gt, int(mask_cpu.numel()))
                    else:
                        logging_string += ' -- loss super: (no gt in batch)'
                elif not _caliper_weak_active_loss:
                    loss_super = _super_loss(latent_batch, item['latent'], log_scale=log_scale_pred_batch)
                    loss += lambda_super * loss_super
                    if n_iter == 1:
                        print(f"[latent-debug] epoch 1 iter 1:")
                        print(f"  encoder latent mean: {latent_batch.mean().item():.6f}, std: {latent_batch.std().item():.6f}")
                        print(f"  target latent mean: {item['latent'].mean().item():.6f}, std: {item['latent'].std().item():.6f}")
                        print(f"  super_loss_type: {_super_type}  value: {loss_super.item():.6f}")
                    writer.add_scalar('Loss/Train/SuperLoss', lambda_super * loss_super, n_iter)
                    logging_string += ' -- loss super: {}'.format((lambda_super * loss_super).item())

            # === Compute volume targets (gt_v_tensor) + masks ===
            # Always compute when caliper_weak is active OR volume_head is enabled OR softvol is on.
            # This block is consumed by both the vol_head MLP loss and the softvol loss below.
            _softvol_gt_v_tensor = None
            _softvol_strong_mask = None
            _softvol_weak_mask = None
            _softvol_valid_mask = None
            _vol_caliper_active = False
            _softvol_lam_strong = float(param.get('lambda_volume', 0.5))
            _softvol_lam_weak = float(param.get('lambda_volume_weak', _softvol_lam_strong))
            _need_gt_vols = (bool(param.get('volume_head', False)) and log_vol_pred_batch is not None) \
                            or (_weak_vol_mode == 'softvol' and bool(param.get("caliper_weak_supervision", False)))
            if _need_gt_vols:
                vol_col = 'volume_ml' if 'volume_ml' in df.columns else 'volume_metashape'
                gt_vols = []
                for fid in item['fruit_id']:
                    vals = df.loc[df['label'] == fid, vol_col].values
                    gt_vols.append(float(vals[0]) if len(vals) > 0 else 0.0)
                # Adjust for scale augmentation: V_scaled = V_gt * (s/100)^3
                if _scale_pct_per_sample is not None and len(_scale_pct_per_sample) == len(gt_vols):
                    gt_vols = [v * ((s / 100.0) ** 3) for v, s in zip(gt_vols, _scale_pct_per_sample)]
                elif _s_pct_batch != 100:
                    _sf = _s_pct_batch / 100.0
                    gt_vols = [v * (_sf ** 3) for v in gt_vols]
                # Caliper weak supervision: override gt_vols with proxy_volume_ml for weak samples.
                _vol_caliper_active = (bool(param.get("caliper_weak_supervision", False))
                                        and 'is_weak' in item and 'proxy_volume_ml' in item)
                if _vol_caliper_active:
                    is_weak_v = item['is_weak'].bool().tolist()
                    proxy_v = item['proxy_volume_ml'].tolist()
                    if _scale_pct_per_sample is not None and len(_scale_pct_per_sample) == len(proxy_v):
                        proxy_v = [pv * ((s / 100.0) ** 3) for pv, s in zip(proxy_v, _scale_pct_per_sample)]
                    elif _s_pct_batch != 100:
                        _sf = _s_pct_batch / 100.0
                        proxy_v = [pv * (_sf ** 3) for pv in proxy_v]
                    gt_vols = [pv if iw and pv > 0 else gv for iw, gv, pv in zip(is_weak_v, gt_vols, proxy_v)]
                _dtype_for_gt = log_vol_pred_batch.dtype if log_vol_pred_batch is not None else latent_batch.dtype
                gt_v_tensor = torch.tensor(gt_vols, device=device, dtype=_dtype_for_gt)
                valid_mask = gt_v_tensor > 0
                _softvol_gt_v_tensor = gt_v_tensor
                _softvol_valid_mask = valid_mask
                if _vol_caliper_active:
                    is_weak_dev = item['is_weak'].bool().to(device)
                    strong_v_mask = valid_mask & (~is_weak_dev)
                    weak_v_mask = valid_mask & is_weak_dev
                    _softvol_strong_mask = strong_v_mask
                    _softvol_weak_mask = weak_v_mask

            # === vol_head MLP loss (only when volume_head is enabled) ===
            if bool(param.get('volume_head', False)) and log_vol_pred_batch is not None and _softvol_gt_v_tensor is not None:
                lam_vol = _softvol_lam_strong
                lam_vol_weak = _softvol_lam_weak
                gt_v_tensor = _softvol_gt_v_tensor
                valid_mask = _softvol_valid_mask
                if _vol_caliper_active:
                    strong_v_mask = _softvol_strong_mask
                    weak_v_mask = _softvol_weak_mask
                    vol_loss_total = torch.tensor(0.0, device=device)
                    if strong_v_mask.any():
                        log_gt_s = torch.log(gt_v_tensor[strong_v_mask].clamp(min=1e-3))
                        l_vol_s = F.l1_loss(log_vol_pred_batch[strong_v_mask], log_gt_s)
                        vol_loss_total = vol_loss_total + lam_vol * l_vol_s
                        writer.add_scalar('Loss/Train/VolumeLoss_strong', lam_vol * l_vol_s, n_iter)
                    if weak_v_mask.any():
                        log_gt_w = torch.log(gt_v_tensor[weak_v_mask].clamp(min=1e-3))
                        l_vol_w = F.l1_loss(log_vol_pred_batch[weak_v_mask], log_gt_w)
                        vol_loss_total = vol_loss_total + lam_vol_weak * l_vol_w
                        writer.add_scalar('Loss/Train/VolumeLoss_weak', lam_vol_weak * l_vol_w, n_iter)
                    if vol_loss_total.item() > 0:
                        loss += vol_loss_total
                        writer.add_scalar('Loss/Train/VolumeLoss', vol_loss_total, n_iter)
                        logging_string += f' -- loss vol: {vol_loss_total.item():.4f} (s={int(strong_v_mask.sum())}, w={int(weak_v_mask.sum())})'
                elif valid_mask.any():
                    log_gt = torch.log(gt_v_tensor[valid_mask].clamp(min=1e-3))
                    loss_vol = F.l1_loss(log_vol_pred_batch[valid_mask], log_gt)
                    loss += lam_vol * loss_vol
                    writer.add_scalar('Loss/Train/VolumeLoss', lam_vol * loss_vol, n_iter)
                    logging_string += ' -- loss vol: {}'.format((lam_vol * loss_vol).item())

            # Size-aware temporal contrastive loss (optional).
            if (bool(param.get("size_aware_contrastive", False))
                    and 'vol_proxy' in item and 'days_since_ref' in item):
                try:
                    from loss import SizeAwareContrastiveLoss
                except Exception:
                    SizeAwareContrastiveLoss = None
                if SizeAwareContrastiveLoss is not None and latent_batch.shape[0] >= 2:
                    # Build per-sample integer fruit index within batch (stable ordering).
                    _fids = list(item['fruit_id'])
                    _uniq = {f: i for i, f in enumerate(dict.fromkeys(_fids))}
                    fruit_idx = torch.tensor([_uniq[f] for f in _fids], device=device, dtype=torch.long)
                    vol_proxy = item['vol_proxy'].to(device).float()
                    dates_days = item['days_since_ref'].to(device).float()
                    loss_tc = SizeAwareContrastiveLoss(
                        latent_batch, fruit_idx, dates_days, vol_proxy,
                        vol_thr=float(param.get("contrastive_vol_thr", 0.05)),
                        days_thr=float(param.get("contrastive_days_thr", 7)),
                        margin=float(param.get("contrastive_margin", 1.0)),
                    )
                    lam_tc = float(param.get("lambda_contrastive", 0.1))
                    loss += lam_tc * loss_tc
                    writer.add_scalar('Loss/Train/SizeAwareContrastive', lam_tc * loss_tc, n_iter)
                    logging_string += ' -- loss tc: {}'.format((lam_tc * loss_tc).item())

            if param.get("reg_latent", False):
                loss_reg = RegLatentLoss(latent_batch, param["lambda_reg_latent"], e)
                loss += loss_reg

                # logging
                writer.add_scalar('Loss/Train/RegLoss',loss_reg, n_iter)
                logging_string += ' -- loss reg: {}'.format(loss_reg.item())

            # creating a Grid3D for each latent in the batch
            box = tensor_dict_2_float_dict(item['bbox'])

            grid_batch = []
            for _ in range(latent_batch.shape[0]):
                grid_batch.append(Grid3D(grid_density, device, precision, bbox=box))

            deepsdf_input = torch.zeros((latent_batch.shape[0], grid_density**3, latent_size+3))
            for batch_idx, (latent, grid) in enumerate(zip(latent_batch, grid_batch)):
                deepsdf_input[batch_idx] = torch.cat([latent.expand(grid.points.size(0), -1), grid.points], dim=1)

            deepsdf_input = deepsdf_input.to(device, latent.dtype)

            # decoding
            pred_sdf = decoder(deepsdf_input)

            # ---- SOFTVOL volume loss (decoder-based, supervises encoder) ----
            # When weak_vol_loss_mode == "softvol", integrate sigmoid(-SDF/T) over the bbox grid to
            # obtain a differentiable soft volume estimate per sample. L1(log soft_vol, log target).
            # Independent of vol_head MLP — works whether volume_head is True or False.
            if (_weak_vol_mode == 'softvol'
                and _softvol_gt_v_tensor is not None):
                T_temp = float(param.get('softvol_temperature', 0.005))
                # pred_sdf shape (B, N, 1). Soft inside ≈ 1 inside, ≈ 0 outside.
                soft_inside = torch.sigmoid(-pred_sdf / T_temp).squeeze(-1)  # (B, N)
                # Voxel volume in m³ (bbox extents are in meters in normalized DeepSDF frame).
                xext = box['xmax'] - box['xmin']
                yext = box['ymax'] - box['ymin']
                zext = box['zmax'] - box['zmin']
                vox_m3 = (xext * yext * zext) / (grid_density ** 3)
                soft_vol_ml = soft_inside.sum(dim=1) * vox_m3 * 1e6  # (B,) m³ → ml
                log_softvol = torch.log(soft_vol_ml.clamp(min=1e-3))
                gt_log = torch.log(_softvol_gt_v_tensor.clamp(min=1e-3))

                if _softvol_strong_mask is not None and _softvol_weak_mask is not None:
                    sv_loss_total = torch.tensor(0.0, device=device)
                    n_s = int(_softvol_strong_mask.sum())
                    n_w = int(_softvol_weak_mask.sum())
                    if n_s > 0:
                        l_s = F.l1_loss(log_softvol[_softvol_strong_mask], gt_log[_softvol_strong_mask])
                        sv_loss_total = sv_loss_total + _softvol_lam_strong * l_s
                        writer.add_scalar('Loss/Train/SoftvolLoss_strong', _softvol_lam_strong * l_s, n_iter)
                    if n_w > 0:
                        l_w = F.l1_loss(log_softvol[_softvol_weak_mask], gt_log[_softvol_weak_mask])
                        sv_loss_total = sv_loss_total + _softvol_lam_weak * l_w
                        writer.add_scalar('Loss/Train/SoftvolLoss_weak', _softvol_lam_weak * l_w, n_iter)
                    if sv_loss_total.item() > 0:
                        loss = loss + sv_loss_total
                        writer.add_scalar('Loss/Train/SoftvolLoss', sv_loss_total, n_iter)
                        logging_string += f' -- loss softvol: {sv_loss_total.item():.4f} (s={n_s}, w={n_w})'
                elif _softvol_valid_mask is not None and _softvol_valid_mask.any():
                    l = F.l1_loss(log_softvol[_softvol_valid_mask], gt_log[_softvol_valid_mask])
                    loss = loss + _softvol_lam_strong * l
                    writer.add_scalar('Loss/Train/SoftvolLoss', _softvol_lam_strong * l, n_iter)
                    logging_string += ' -- loss softvol: {}'.format((_softvol_lam_strong * l).item())

            if param.get("3D_loss", False) and 'target_sdf' in item and 'target_sdf_xyz' in item:
                # SDF supervision via DeepSDF-style sample points (xyz, sdf) from
                # laser/samples.npz. Decoder is forwarded at the SAMPLE points
                # (not the validation grid) because the grid would require
                # interpolating sample SDF values — lossy and unnecessary.
                # `target_sdf_*` may contain None for scaled-aug samples
                # (BinBalancedDataset disables SDF supervision when partial_pcd is
                # scaled and the cached SDF is for the original-scale fruit).
                ts_list = item['target_sdf'] if isinstance(item['target_sdf'], list) else [item['target_sdf']]
                xy_list = item['target_sdf_xyz'] if isinstance(item['target_sdf_xyz'], list) else [item['target_sdf_xyz']]
                tw_list = item['target_sdf_weights'] if isinstance(item['target_sdf_weights'], list) else [item['target_sdf_weights']]
                # Pair with batch latent index, filter out None entries
                valid = [(li, xy, ts, tw) for li, (xy, ts, tw) in enumerate(zip(xy_list, ts_list, tw_list))
                         if xy is not None and ts is not None and tw is not None]
                if valid:
                    z_valid = latent_batch[[li for li, _, _, _ in valid]]                          # (B', D)
                    xy_b = torch.stack([xy for _, xy, _, _ in valid]).to(device)                   # (B', N, 3)
                    ts_b = torch.stack([ts for _, _, ts, _ in valid]).to(device)                   # (B', N, 1)
                    Bp, N, _ = xy_b.shape
                    z_exp = z_valid.unsqueeze(1).expand(-1, N, -1)                                 # (B', N, D)
                    dec_in = torch.cat([z_exp, xy_b], dim=-1).reshape(Bp * N, -1)                  # (B'*N, D+3)
                    pred_at_samples = decoder(dec_in).reshape(Bp, N, 1)                            # (B', N, 1)
                    # Standard DeepSDF loss: clamped L1 (use config sdf_trunc).
                    sdf_trunc = float(getattr(cl_dataset, 'sdf_trunc', 0.015))
                    pred_c = pred_at_samples.clamp(-sdf_trunc, sdf_trunc)
                    target_c = ts_b.clamp(-sdf_trunc, sdf_trunc)
                    loss_sdf = F.l1_loss(pred_c, target_c)
                    loss += param['lambda_sdf'] * loss_sdf
                    writer.add_scalar('Loss/Train/SDFLoss', param['lambda_sdf'] * loss_sdf, n_iter)
                    logging_string += f' -- loss sdf: {(param["lambda_sdf"] * loss_sdf).item():.4f} (n_valid={Bp}/{len(ts_list)})'

            # Only do backward pass if we have actual losses
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                loss.backward()
                optim.step()
                
                # tensorboard logging
                writer.add_scalar('LRate', scheduler.get_last_lr()[0], n_iter)
                writer.add_scalar('Loss/Train/Total', loss, n_iter)
                logging_string += ' -- loss: {}'.format(loss.item())
                logging_string += ' -- lr: {}'.format(scheduler.get_last_lr()[0])
                print(logging_string)

                # epoch-level aggregation
                epoch_loss_sum += float(loss.item())
                epoch_loss_count += 1
            elif loss != 0:
                # loss is a tensor but has no grad (shouldn't happen with proper config)
                print(f"Warning: Loss has no gradient at iteration {n_iter}. Check your loss configuration.")
            else:
                print(f"Warning: No losses computed at iteration {n_iter}. Check your config - all loss flags are False!")

        scheduler.step()

        # Log average train loss per epoch (use epoch index as x-axis)
        if epoch_loss_count > 0:
            avg_epoch_loss = epoch_loss_sum / epoch_loss_count
            writer.add_scalar('Loss/Train/AvgPerEpoch', avg_epoch_loss, e + 1)
            print(f"[train] epoch {e+1}/{param['epoch']} avg train loss: {avg_epoch_loss:.6f} ({epoch_loss_count} iters)")
        else:
            print(f"[train] epoch {e+1}/{param['epoch']} avg train loss: skipped (no valid loss iterations)")

        # validation step
        if (e+1) % param["validation_frequency"] == 0:
            # Validation should run in eval mode (BatchNorm/Dropout behave differently).
            # This is especially important because validation uses batch_size=1 by default.
            encoder_was_training = encoder.training
            encoder.eval()
            try:
                with torch.no_grad():
                    val_tfs = [Pad(size=param["input_size"])]
                    val_tf = transforms.Compose(val_tfs)

                val_cl_dataset = MaskedCameraLaserData(data_source=param["data_dir"],
                                                        tf=val_tf,
                                                        color_tf = None, 
                                                        pretrain=pretrain,
                                                        pad_size=param["input_size"],
                                                        detection_input=param["detection_input"],
                                                        normalize_depth=param["normalize_depth"],
                                                        depth_min=param["depth_min"],
                                                        depth_max=param["depth_max"],
                                                        supervised_3d=param.get("supervised_3d", False),
                                                        sdf_loss=param.get("3D_loss", False),
                                                        grid_density=param.get("grid_density", 0),
                                                        split='val',
                                                        overfit=overfit,
                                                        species=param["species"],
                                                        use_phenobox_pcd=bool(param.get("use_phenobox_pcd", False)),
                                                        phenobox_subdir=param.get("phenobox_subdir", "phenobox/pointclouds"),
                                                        use_orbbec_pcd=bool(param.get("use_orbbec_pcd", False)),
                                                        orbbec_pcd_suffix=param.get("orbbec_pcd_suffix", "_preprocessed"),
                                                        split_json_path=param.get("split_json", None),
                                                        require_dataset_json=param.get("require_dataset_json", True),
                                                        normalize_pcd=normalize_pcd_flag,
                                                        # Val defaults to last-date only so val_rmse reflects
                                                        # final-shape accuracy (avoids penalizing correct early-stage
                                                        # predictions when GT vol is per-fruit constant).
                                                        temporal_loading=bool(param.get("val_temporal_loading", False)),
                                                        vol_map_csv=param.get("vol_map_csv", None),
                                                        gt_vol_similar_thr=float(param.get("gt_vol_similar_thr", 0.05)),
                                                        use_pcd_render_as_rgbd=bool(param.get("use_pcd_render_as_rgbd", False)),
                                                        render_size=int(param.get("render_size", 304)),
                                                        render_splat_radius=int(param.get("render_splat_radius", 3)),
                                                        )

                # Wrap validation with ScaleExpandedDataset if scale_augment_all is active
                if scale_augment_all and scaled_latents:
                    val_cl_dataset = ScaleExpandedDataset(val_cl_dataset, scaled_latents, scale_aug_pcts)
                val_dataset = DataLoader(val_cl_dataset, batch_size=1, shuffle=False)

                gt_volumes = []
                pred_volumes = []
                val_loss_sum = 0.0
                val_loss_count = 0
                print('\nvalidation...')
                for val_idx, item in enumerate(tqdm(iter(val_dataset))):
                    try:
                        if param['encoder'] not in ['point_cloud', 'point_cloud_maxmean', 'point_cloud_maxmean_masked', 'point_cloud_large', 'point_cloud_maxmean_large', 'point_cloud_maxmean_large_masked', 'pointnet2', 'foldnet', 'minkowski']:
                            encoder_input = torch.cat((item['rgb'], item['depth']), 1).to(device)
                            latent_val = encoder(encoder_input)
                        elif param['encoder'] == 'minkowski':
                            # Handle both stacked tensors and lists of tensors (from custom collate)
                            if isinstance(item['partial_pcd_world'], list):
                                partial_points = [p.to(device) for p in item['partial_pcd_world']]
                                bbox_centers = [c.to(device) for c in item['bbox_center_tensor']]
                                bbox_scales = [s.to(device) for s in item['bbox_scale_tensor']]
                            else:
                                partial_points = item['partial_pcd_world'].to(device)
                                bbox_centers = item['bbox_center_tensor'].to(device)
                                bbox_scales = item['bbox_scale_tensor'].to(device)
                            latent_val = encoder(partial_points, bbox_centers, bbox_scales)
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
                                latent_val = encoder(encoder_input, mask=real_mask)
                            else:
                                latent_val = encoder(encoder_input)
                        # Use per-sample bbox for validation
                        val_box = tensor_dict_2_float_dict(item['bbox'])
                        grid_val = Grid3D(grid_density, device, precision, bbox=val_box)
                        dec_input_val = torch.cat([latent_val.expand(grid_val.points.size(0), -1), grid_val.points], dim=1)

                        pred_sdf_val = decoder(dec_input_val)

                        # Validation loss (computed from already-available supervision signals)
                        # Note: Not all datasets/configs provide all targets (e.g., target_sdf may be absent).
                        val_sample_loss = torch.tensor(0.0, device=device)
                        if param.get('supervised_3d', False) and 'latent' in item:
                            target_lat = item['latent'].to(device)
                            if target_lat.dim() == 1:
                                target_lat = target_lat.unsqueeze(0)
                            lambda_super = float(param.get('lambda_super', 1.0))
                            val_sample_loss = val_sample_loss + lambda_super * SuperLoss(latent_val, target_lat)

                        if param.get('reg_latent', False):
                            val_sample_loss = val_sample_loss + RegLatentLoss(latent_val, param["lambda_reg_latent"], e)

                        # Optional SDF loss using sample points from laser/samples.npz.
                        # pred at sample xyz (NOT the grid) so shapes match target_sdf.
                        if (param.get("3D_loss", False) and ('target_sdf' in item)
                                and ('target_sdf_xyz' in item)
                                and isinstance(item.get('target_sdf'), torch.Tensor)
                                and isinstance(item.get('target_sdf_xyz'), torch.Tensor)):
                            try:
                                xy_v = item['target_sdf_xyz'].to(device)
                                ts_v = item['target_sdf'].to(device)
                                if xy_v.dim() == 2:
                                    xy_v = xy_v.unsqueeze(0)
                                if ts_v.dim() == 2:
                                    ts_v = ts_v.unsqueeze(0)
                                z_v = latent_val if latent_val.dim() == 2 else latent_val.unsqueeze(0)
                                Bp, Nv, _ = xy_v.shape
                                z_exp_v = z_v.unsqueeze(1).expand(-1, Nv, -1)
                                dec_in_v = torch.cat([z_exp_v, xy_v], dim=-1).reshape(Bp * Nv, -1)
                                pred_at_v = decoder(dec_in_v).reshape(Bp, Nv, 1)
                                trunc = float(getattr(val_cl_dataset, 'sdf_trunc', 0.015))
                                loss_sdf_val = F.l1_loss(
                                    pred_at_v.clamp(-trunc, trunc),
                                    ts_v.clamp(-trunc, trunc),
                                )
                                val_sample_loss = val_sample_loss + float(param.get('lambda_sdf', 1.0)) * loss_sdf_val
                            except Exception as _ex_sdf:
                                # Best-effort: skip SDF loss for this val sample on error
                                pass

                        # Track only when we computed something meaningful
                        if isinstance(val_sample_loss, torch.Tensor):
                            val_loss_sum += float(val_sample_loss.detach().item())
                            val_loss_count += 1

                        if val_idx < 3 and e < 5:
                            sdf_cpu = pred_sdf_val.detach().cpu()
                            sdf_min = float(sdf_cpu.min().item())
                            sdf_max = float(sdf_cpu.max().item())
                            sdf_mean = float(sdf_cpu.mean().item())
                            thresholds = [param['threshold'], 0.01, 0.0, -0.001]
                            ratios = []
                            for t in thresholds:
                                mask = sdf_cpu < t
                                ratios.append(float(mask.float().mean().item()))
                            ratio_str = ", ".join(f"t={t:+.3f}:{r*100:.1f}%" for t, r in zip(thresholds, ratios))
                            print(f"[sdf-stats] epoch {e+1} val_idx {val_idx}: min={sdf_min:.6f}, max={sdf_max:.6f}, mean={sdf_mean:.6f} | {ratio_str}")

                        mesh_val = sdf2mesh_cuda(
                            pred_sdf_val,
                            grid_val.points,
                            t=param['threshold']
                        )

                        pred_volume = mesh_val.get_volume()

                        label = item['fruit_id'][0]
                        vol_col = 'volume_ml' if 'volume_ml' in df.columns else 'volume_metashape'
                        gt_vals = df.loc[df['label'] == label, vol_col].values
                        if len(gt_vals) > 0:
                            gt_vol = gt_vals[0]
                            # Adjust GT volume for scaled samples: V_scaled = V_gt * (s/100)^3
                            s_pct = int(item.get('scale_pct', [100])[0])
                            if s_pct != 100:
                                gt_vol = gt_vol * (s_pct / 100.0) ** 3
                            gt_volumes.append(gt_vol)
                            pred_volumes.append(round(pred_volume * 1e6, 1))
                        else:
                            print('Validation GT volume missing for:', label)

                        if overfit: break                        
                    except Exception as ex:
                        print('Validation sample failed:', ex)

                if len(gt_volumes) > 0 and len(pred_volumes) > 0:
                    mse_volume = mean_squared_error(gt_volumes, pred_volumes)
                    rmse_volume = float(np.sqrt(mse_volume))
                    print('RMSE volume: ', round(rmse_volume, 1))

                    # Additional metrics
                    pred_arr = np.array(pred_volumes)
                    gt_arr = np.array(gt_volumes)
                    # MAPE (no outlier protection): mean(|pred-gt|/gt)*100
                    valid_mask = gt_arr > 0
                    if valid_mask.any():
                        mape_volume = float(np.mean(np.abs(pred_arr[valid_mask] - gt_arr[valid_mask]) / gt_arr[valid_mask]) * 100)
                    else:
                        mape_volume = float('inf')
                    mean_pred = np.mean(pred_arr)
                    mean_gt = np.mean(gt_arr)
                    print(f'  Mean predicted: {mean_pred:.1f}, Mean GT: {mean_gt:.1f}')
                    print(f'  Pred range: [{pred_arr.min():.1f}, {pred_arr.max():.1f}]')
                    print(f'  GT range: [{gt_arr.min():.1f}, {gt_arr.max():.1f}]')
                    print(f'  MAPE volume: {mape_volume:.2f}%')
                else:
                    rmse_volume = float('inf')
                    mape_volume = float('inf')
                    print('RMSE volume: skipped (no valid validation samples)')

                # Validation loss summary
                if val_loss_count > 0:
                    val_loss_mean = float(val_loss_sum / val_loss_count)
                    print(f'Val loss (avg over {val_loss_count} samples): {val_loss_mean:.6f}')
                else:
                    val_loss_mean = float('inf')
                    print('Val loss: skipped (no loss components available)')

                # logging
                writer.add_scalar('Val/rmse_volume', rmse_volume, n_iter)
                # Also log RMSE volume with epoch as x-axis for a cleaner trend line
                writer.add_scalar('Val/rmse_volume_per_epoch', rmse_volume, e + 1)
                writer.add_scalar('Val/val_loss', val_loss_mean, n_iter)
                writer.add_scalar('Val/val_loss_per_epoch', val_loss_mean, e + 1)
                writer.add_scalar('Val/mape_volume', mape_volume, n_iter)
                writer.add_scalar('Val/mape_volume_per_epoch', mape_volume, e + 1)
                # saving best model
                if model_selection_metric in {"val_loss", "loss"}:
                    current_score = val_loss_mean
                    metric_name = "val_loss"
                elif model_selection_metric in {"val_mape", "val_volume_mape", "mape"}:
                    current_score = mape_volume
                    metric_name = "val_volume_mape"
                else:
                    current_score = rmse_volume
                    metric_name = "val_volume_rmse"

                if current_score < best_score:
                    best_score = current_score
                    best_model_path = os.path.join(param["checkpoint_dir"], f"{model_prefix}_best_model.pt")
                    save_model(encoder, decoder, e, optim, loss, best_model_path)
                    print(f'saving best model (metric={metric_name}, score={best_score:.6f})')

                # Also independently track and save best per metric
                if rmse_volume < best_rmse_score:
                    best_rmse_score = rmse_volume
                    rmse_path = os.path.join(param["checkpoint_dir"], f"{model_prefix}_best_rmse_model.pt")
                    save_model(encoder, decoder, e, optim, loss, rmse_path)
                    print(f'saving best_rmse model (val_volume_rmse={best_rmse_score:.6f})')
                if val_loss_mean < best_loss_score:
                    best_loss_score = val_loss_mean
                    loss_path = os.path.join(param["checkpoint_dir"], f"{model_prefix}_best_loss_model.pt")
                    save_model(encoder, decoder, e, optim, loss, loss_path)
                    print(f'saving best_loss model (val_loss={best_loss_score:.6f})')
                if mape_volume < best_mape_score:
                    best_mape_score = mape_volume
                    mape_path = os.path.join(param["checkpoint_dir"], f"{model_prefix}_best_mape_model.pt")
                    save_model(encoder, decoder, e, optim, loss, mape_path)
                    print(f'saving best_mape model (val_volume_mape={best_mape_score:.4f}%)')
                print()
            finally:
                if encoder_was_training:
                    encoder.train()

        # saving checkpoints
        if (e+1) % param["checkpoint_frequency"] == 0:
            checkpoint_path = os.path.join(param["checkpoint_dir"], f"{model_prefix}_epoch_{e+1:03d}.pt")
            save_model(encoder, decoder, e, optim, loss, checkpoint_path)

    # saving last model
    final_model_path = os.path.join(param["checkpoint_dir"], f"{model_prefix}_final_model.pt")
    save_model(encoder, decoder, e, optim, loss, final_model_path)

    return


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
        "--overfit",
        dest="overfit",
	    action='store_true',
        help="Overfit the network.",
    )

    arg_parser.add_argument(
        "--checkpoint_decoder",
        dest="checkpoint",
        default=None,
        help="The checkpoint weights to use. This should be a number indicated an epoch. If not provided, reads from config file.",
    )
    arg_parser.add_argument(
        "--decoder",
        dest="decoder",
	    action='store_true',
        help="Update decoder network.",
    )
    arg_parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume training (expects --resume_checkpoint, and optionally --resume_run_root).",
    )
    arg_parser.add_argument(
        "--resume_checkpoint",
        dest="resume_checkpoint",
        default=None,
        help="Path to a saved .pt checkpoint (encoder+decoder+optimizer) produced by train.py.",
    )
    arg_parser.add_argument(
        "--resume_run_root",
        dest="resume_run_root",
        default=None,
        help="Existing run root directory to continue logging into (e.g. experiments/.../<run>/).",
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
        print(f"[train] Using experiment from config: {args.experiment_directory}")
    
    if args.checkpoint is None:
        if "checkpoint_decoder" in config:
            args.checkpoint = str(config["checkpoint_decoder"])
            print(f"[train] Using checkpoint_decoder from config: {args.checkpoint}")
        else:
            args.checkpoint = "500"  # default fallback
            print(f"[train] Using default checkpoint_decoder: {args.checkpoint}")

    deep_sdf.configure_logging(args)

    # loading deepsdf model
    specs = ws.load_experiment_specifications(args.experiment_directory)
    latent_size = specs["CodeLength"]
    arch = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    path = os.path.join(args.experiment_directory, 'ModelParameters', args.checkpoint) + '.pth'
    model_state = net_utils.load_without_parallel(torch.load(path))
    decoder.load_state_dict(model_state)
    decoder = net_utils.set_require_grad(decoder, True)

    pretrain_path = os.path.join(args.experiment_directory, 'Reconstructions', args.checkpoint, 'Codes', 'complete')

    main_function(decoder=decoder,
                  pretrain=pretrain_path,
                  cfg=args.cfg,
                  latent_size=latent_size,
                  trunc_val=specs['ClampingDistance'],
		          overfit=args.overfit,
                  update_decoder=args.decoder,
                  resume=args.resume,
                  resume_checkpoint=args.resume_checkpoint,
                  resume_run_root=args.resume_run_root)