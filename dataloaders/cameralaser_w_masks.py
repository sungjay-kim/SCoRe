#!/usr/bin/env python3

import copy
import json
import numpy as np
import os
import re
import torch
import torch.utils.data
from torchvision import transforms
from PIL import Image
import cv2 as cv
import open3d as o3d

import matplotlib.pyplot as plt
import matplotlib as mpl
from torchvision.transforms.transforms import ToTensor, Resize
from torch.utils.data import DataLoader

from dataloaders import dataset_layout as layout


def collate_fn_variable_pcd(batch):
    """Custom collate that keeps variable-length tensors (target_pcd, target_sdf, target_sdf_weights, partial_pcd, partial_pcd_world) as lists."""
    if not batch:
        return {}
    
    collated = {}
    keys = batch[0].keys()
    
    for key in keys:
        values = [item[key] for item in batch]
        
        # Keep these fields as lists (variable size or may contain None)
        if key in {'target_pcd', 'target_sdf', 'target_sdf_weights', 'target_sdf_xyz', 'partial_pcd', 'partial_pcd_world'}:
            collated[key] = values
        # bbox is a dict - all samples share the same global bbox, just use the first one
        elif key == 'bbox':
            collated[key] = values[0]
        elif isinstance(values[0], torch.Tensor):
            # Stack fixed-size tensors
            try:
                collated[key] = torch.stack(values, dim=0)
            except RuntimeError:
                # Fallback: keep as list if stacking fails
                collated[key] = values
        else:
            # Non-tensor values (strings, etc.)
            collated[key] = values
    
    return collated


# from transforms import Pad
from dataloaders.transforms import Pad

mpl.rcParams['image.cmap'] = 'gray'

NAME = '0'

def imshow(img):
    # plt.imshow(img[:, :, [2, 1, 0]])
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    

## load intrinsics file
def load_intrinsics(intrinsics_file):
    with open(intrinsics_file) as json_file:
        data = json.load(json_file)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(data['width'], data['height'], data['intrinsic_matrix'][0], data['intrinsic_matrix'][4], data['intrinsic_matrix'][6], data['intrinsic_matrix'][7])
    return intrinsics, data['depth_scale']


## Histogram filter for the depth image
def histogram_filtering(dimg, mask, max_depth_range=50, max_depth_contribution=0.05):
    mask = mask.astype(np.uint8)
    mask_bool = mask.astype(bool)
    
    z = np.expand_dims(dimg, axis=2)
    z_mask = z[mask_bool]
    z_mask_filtered = z_mask[z_mask != 0]

    # Robust guard: if no valid depth inside the mask, fall back to a safe range
    if z_mask_filtered.size == 0:
        z_max_fallback = float(dimg.max()) if dimg.size > 0 else 0.0
        return 0.0, z_max_fallback

    if z_mask_filtered.size > 1: 
        z_mask_filtered_range = np.max(z_mask_filtered)-np.min(z_mask_filtered)

        if (z_mask_filtered_range > max_depth_range):
            hist, bin_edges = np.histogram(z_mask_filtered, density=False) 
            hist_peak = np.argmax(hist)
            lb = bin_edges[hist_peak]
            ub = bin_edges[hist_peak+1]

            bc = np.bincount(np.absolute(z_mask_filtered.astype(np.int64)))
            peak_id = np.argmax(bc)

            if peak_id > int(lb) and peak_id < int(ub):
                peak_id = peak_id
            else:
                bc_clip = bc[int(lb):int(ub)]
                peak_id = int(lb) + np.argmax(bc_clip)

            pixel_counts = np.zeros((10), dtype=np.int64)

            for j in range(10):
                lower_bound = peak_id-(max_depth_range - (j * 10))
                upper_bound = lower_bound + max_depth_range
                z_final = z_mask_filtered[np.where(np.logical_and(z_mask_filtered >= lower_bound, z_mask_filtered <= upper_bound))]
                pixel_counts[j] = z_final.size

            pix_id = np.argmax(pixel_counts)
            lower_bound = peak_id-(max_depth_range - (pix_id * 10))
            upper_bound = lower_bound + max_depth_range
            z_final = z_mask_filtered[np.where(np.logical_and(z_mask_filtered >= lower_bound, z_mask_filtered <= upper_bound))]
            
        else:
            z_final = z_mask_filtered

        hist_f, bin_edges_f = np.histogram(z_final, density=False)
        norm1 = hist_f / np.sum(hist_f)

        sel1 = bin_edges_f[np.where(norm1 >= max_depth_contribution)]
        sel2 = bin_edges_f[np.where(norm1 >= max_depth_contribution)[0]+1]
        edges = np.concatenate((sel1,sel2), axis=0)
        final_bins = np.unique(edges)
 
        z_min = np.min(final_bins)
        z_max = np.max(final_bins)
    else:
        # Size is 1: min==max is fine
        z_min = float(np.min(z_mask_filtered))
        z_max = float(np.max(z_mask_filtered))
    
    return z_min, z_max


def visualize_sdf(sdf_data):
    xyz = sdf_data[:, :-1]
    val = sdf_data[:, -1]

    xyz_min = xyz[val<0] + np.array([0,0,0])
    xyz_max = xyz[val>0]

    val_min = val[val<0]
    val_max = val[val>0]

    val_min += np.min(val_min)
    val_min /= np.max(val_min)

    val_max -= np.min(val_max)
    val_max /= np.max(val_max)

    colors_min = np.zeros(xyz_min.shape)
    colors_min[:, 0] =  val_min

    colors_max = np.zeros(xyz_max.shape)
    colors_max[:, 2] =  val_max

    pcd_min = o3d.geometry.PointCloud()
    pcd_min.points = o3d.utility.Vector3dVector(xyz_min)
    pcd_min.colors = o3d.utility.Vector3dVector(colors_min)

    pcd_max = o3d.geometry.PointCloud()
    pcd_max.points = o3d.utility.Vector3dVector(xyz_max)
    pcd_max.colors = o3d.utility.Vector3dVector(colors_max)

    o3d.visualization.draw_geometries([pcd_min, pcd_max])

def sample_freespace(pcd, n_samples=20000):
    xyz = np.asarray(pcd.points)
    n_xyz = np.asarray(pcd.normals)

    xyz_free = []
    mu_free = []
    for _ in range(n_samples):

        mu_freespace = np.random.uniform(0.001, 0.01)
        idx = np.random.randint(len(xyz))

        p = xyz[idx]
        pn = n_xyz[idx]

        sample = p + mu_freespace*pn

        xyz_free.append(sample)
        mu_free.append(mu_freespace)

    return np.asarray(xyz_free), np.asarray(mu_free)

def generate_deepsdf_target(pcd, mu=0.001, align_with=np.array([0.0, 1.0, 0.0])):
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction(align_with)

    # o3d.visualization.draw_geometries([pcd])

    xyz = np.asarray(pcd.points)
    n_xyz = np.asarray(pcd.normals)
    
    xyz_pos = xyz + mu*n_xyz
    xyz_neg = xyz - mu*n_xyz
    sdf_val_pos = np.repeat(mu, len(xyz))
    sdf_val_neg = np.repeat(-mu, len(xyz))

    xyz_free, sdf_val_free = sample_freespace(pcd)

    # merging positive samples from depth and from freespace
    xyz_pos = np.vstack((xyz_pos, xyz_free))
    sdf_val_pos = np.concatenate((sdf_val_pos, sdf_val_free))

    # packing deepsdf input
    sdf_pos = np.empty((len(xyz_pos), 4))
    sdf_pos[:, :3] = xyz_pos
    sdf_pos[:, 3] = sdf_val_pos 

    sdf_neg = np.empty((len(xyz_neg), 4))
    sdf_neg[:, :3] = xyz_neg
    sdf_neg[:, 3] = sdf_val_neg

    # visualize_sdf(np.vstack((sdf_pos, sdf_neg))) 

    return sdf_pos, sdf_neg

def show_cv(img, fruit_id):
    cv.imshow(fruit_id, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def _fast_subsample_xyz(xyz_full: np.ndarray, pad_size: int) -> np.ndarray:
    """Vectorized replacement for the open3d random_down_sample + select_by_index
    flow used in MaskedCameraLaserData.__getitem__. Preserves the original
    2-stage semantics (Bernoulli subset then uniform trim/upsample) using numpy.
    """
    n = int(xyz_full.shape[0])
    if n == 0:
        return np.zeros((pad_size, 3), dtype=np.float32)
    ratio = (pad_size + 100) / float(n)
    if ratio >= 1.0:
        pts = xyz_full
    else:
        mask = np.random.random(n) < ratio
        pts = xyz_full[mask]
    cur = int(pts.shape[0])
    if cur == 0:
        select_n = min(pad_size, n)
        idx = np.random.choice(n, size=select_n, replace=False)
        pts = xyz_full[idx]
        cur = int(pts.shape[0])
    if cur < pad_size:
        if cur == 0:
            return np.zeros((pad_size, 3), dtype=np.float32)
        idx = np.random.choice(cur, size=pad_size, replace=True)
        pts = pts[idx]
    elif cur > pad_size:
        idx = np.random.choice(cur, size=pad_size, replace=False)
        pts = pts[idx]
    return np.ascontiguousarray(pts, dtype=np.float32)


class MaskedCameraLaserData(torch.utils.data.Dataset):
    def __init__(self, data_source, pad_size, detection_input, normalize_depth, depth_min, depth_max, pretrain, grid_density, 
                tf=None, color_tf=None, pcd_tf=None, supervised_3d=False, split=None, sdf_loss=False, sdf_trunc=0.015, overfit=False, species=None,
                use_phenobox_pcd=False, phenobox_subdir='phenobox/pointclouds',
                use_orbbec_pcd=False, orbbec_pcd_suffix='_preprocessed',
                split_json_path=None,
                require_dataset_json=True,
                normalize_pcd=False,
                temporal_loading=False,
                partial_pcd_count=None,
                pre_pad_pcd_tf=None,
                vol_map_csv=None,
                gt_vol_similar_thr=0.05,
                caliper_weak_supervision=False,
                gt_volume_csv=None,
                caliper_correct_proxy=False,
                cache_pcds_in_ram=False,
                # Vanilla CoRe++ baseline: render the colored partial PCD to a
                # 2D RGB-D image so image-based encoders (DoubleEncoder,
                # EncoderBigPooled) can consume the same partial observation
                # PCE sees. See corepp/baselines/corepp_vanilla/render_pcd.py.
                use_pcd_render_as_rgbd=False,
                render_size=304,
                render_splat_radius=3,
                # Pre-render every PCD once at __init__ and cache the resulting
                # (rgb, depth, mask) as uint8 numpy arrays. With caching enabled
                # __getitem__ becomes a dict lookup; without it, the splat-based
                # rasterizer runs in every worker on every iteration and
                # bottlenecks training to <5% GPU utilization.
                cache_rendered_rgbd=False,
                # Fixed-scale rendering: when True, the rasterizer uses a fixed
                # xy and z half-extent (defaults 0.07 m) so the encoder reads
                # metric scale from BOTH the projected bbox area AND the
                # depth-value spread. With fixed_scale=False (legacy auto-fit),
                # a 5 cm and a 10 cm fruit produce identical rendered images,
                # destroying the size signal needed for volume regression.
                # Alternative input path: load aligned camera-frame RGB-D
                # directly from the raw 2025/2026 captures (depth_to_color
                # extrinsic warp for 2026, direct load for 2025). This is the
                # ORIGINAL CoRe++ recipe — depth carries true camera distance,
                # the fruit's appearance scales with distance, and metric size
                # is recoverable from the image without any extra channel.
                # Overrides PCD rasterization when True.
                use_raw_camera_rgbd=False,
                raw_rgbd_depth_min_mm=500.0,
                raw_rgbd_depth_max_mm=1100.0,
                raw_rgbd_bbox_margin=0.10,
                render_fixed_scale=False,
                render_fixed_xy_half_extent=0.07,
                render_fixed_z_half_extent=0.07,
                # Resample every PCD to this point count before rasterising,
                # so the renderer sees the same source density regardless of
                # the underlying frame's true sparsity. Critical for the
                # 2025-vs-2026 density gap (S 3.3k pts / frame vs P 1.1k pts).
                # None = no resampling.
                render_target_points=None,
                render_target_jitter_mm=1.5,
                ): # Default to False to match corepp_bak behavior, or make explicit
	    
        self.overfit = overfit
        self.data_source = data_source
        self.species = species
        # ... (rest of init)
        self.normalize_pcd = normalize_pcd
        # ...

	    
        self.overfit = overfit
        self.data_source = data_source
        self.species = species
        self.require_dataset_json = bool(require_dataset_json)
        # Store whether SDF/3D loss is active to conditionally include large variable-size fields
        self.sdf_loss = sdf_loss
        # Persist SDF truncation value (was previously only used by compute_target_sdf
        # via self.sdf_trunc but never stored). Needed for L1/SDFLoss clamp range.
        self.sdf_trunc = float(sdf_trunc)
        # SDF samples cache (laser/samples.npz) for sdf_loss=True. Keyed by fruit_id.
        # Loaded lazily on first access in __getitem__.
        self._sdf_samples_cache = {}
        # Per-iter subsample count for SDF loss (split half pos / half neg). Configurable
        # later via param if needed; sensible default 4000 keeps decoder forward fast.
        self.sdf_subsample = 4000
        # Initialize phenobox flags BEFORE reading filenames
        self.use_phenobox_pcd = use_phenobox_pcd
        self.phenobox_subdir = phenobox_subdir
        # Initialize orbbec point cloud flags. The public release names partial clouds by
        # session date with no suffix, so the tag is dropped there and every strip of it
        # downstream becomes a no-op.
        self.use_orbbec_pcd = use_orbbec_pcd
        self.orbbec_pcd_suffix = layout.partial_suffix(data_source, orbbec_pcd_suffix)
        self.temporal_loading = temporal_loading
        self.partial_pcd_count = partial_pcd_count

        # Vanilla CoRe++ baseline rendering flags
        self.use_pcd_render_as_rgbd = bool(use_pcd_render_as_rgbd)
        self.render_size = int(render_size)
        self.render_splat_radius = int(render_splat_radius)
        self.cache_rendered_rgbd = bool(cache_rendered_rgbd)
        self._rendered_rgbd_cache = {}  # fpath -> (rgb_u8, depth_u8, mask_u8)
        self.render_fixed_scale = bool(render_fixed_scale)
        self.render_fixed_xy_half_extent = float(render_fixed_xy_half_extent)
        self.render_fixed_z_half_extent = float(render_fixed_z_half_extent)
        self.render_target_points = None if render_target_points is None else int(render_target_points)
        self.render_target_jitter_mm = float(render_target_jitter_mm)
        self.use_raw_camera_rgbd = bool(use_raw_camera_rgbd)
        self.raw_rgbd_depth_min_mm = float(raw_rgbd_depth_min_mm)
        self.raw_rgbd_depth_max_mm = float(raw_rgbd_depth_max_mm)
        self.raw_rgbd_bbox_margin = float(raw_rgbd_bbox_margin)

        self.grid_density = grid_density
        self.supervised_3d = supervised_3d
        self.tf = tf
        self.color_tf = color_tf
        self.pcd_tf = pcd_tf
        self.pre_pad_pcd_tf = pre_pad_pcd_tf
        self.vol_map_csv = vol_map_csv
        self.gt_vol_similar_thr = float(gt_vol_similar_thr)
        # Initialized later in _build_vol_map() after files are known
        self.vol_map = {}
        self.fruit_last_vol = {}
        self.fruit_last_date = {}
        self.global_ref_date = None

        # Caliper weak supervision: treat intermediate frames as weak-labeled samples.
        # Requires vol_map_csv (caliper-derived ellipsoid volumes per (fruit, date)) AND
        # gt_volume_csv (laser-measured last-date volume per fruit). When enabled:
        #   - last_date frames remain STRONG (laser GT, full SDF + super + volume loss)
        #   - intermediate frames become WEAK (caliper proxy GT, super + volume only;
        #     scaled latent target = z_last × (V_proxy/V_laser)^(1/3); SDF skipped)
        # Intermediate frames lacking caliper data are filtered out of self.files.
        # Fruits without caliper data are unchanged (only last_date strong samples).
        self.caliper_weak_supervision = bool(caliper_weak_supervision)
        self.caliper_correct_proxy = bool(caliper_correct_proxy)
        self.gt_volume_csv = gt_volume_csv
        self.last_laser_vol = {}   # fruit_id -> V_laser (ml) from gt_volume_csv (last-date laser)
        self.is_weak_per_file = []  # parallel to self.files; True for intermediate weak frames

        # PCD RAM cache: pre-load PLY files to eliminate per-iter NFS reads.
        # Used by temporal training (no DataLoader workers) where PCD I/O dominates.
        # Key: file path (partial) or fruit_id (target). Value: o3d.PointCloud.
        # Memory: ~1024 pts × 12B per partial PCD × ~1000 files = ~12MB; targets ~50MB total.
        self.cache_pcds_in_ram = bool(cache_pcds_in_ram)
        self._partial_pcd_cache = {}
        self._target_pcd_cache = {}
        self._partial_pcd_xyz_cache = {}
        self._target_pcd_xyz_cache = {}

        self.pad_size = pad_size
        self.detection_input = detection_input
        self.normalize_depth = normalize_depth
        self.depth_min = depth_min  # in mm
        self.depth_max = depth_max  # in mm
        self.normalize_pcd = normalize_pcd

        # Split selection: by default read "<data_source>/split.json".
        # You can override via `split_json_path` (absolute path, or path relative to data_source).
        if split_json_path is None:
            split_json_path = os.path.join(self.data_source, 'split.json')
        elif not os.path.isabs(split_json_path):
            split_json_path = os.path.join(self.data_source, split_json_path)
            
        self.split_json_path = split_json_path  # Store for find_global_bbox

        with open(split_json_path) as json_file:
            self.split_ids = json.load(json_file)[split]

        # Only allow canonical sample IDs: S12, P101, P3-1 in the working tree, A1/B7/C80
        # in the public release (no suffixes like _unhealthy/_unhealthy_recon), and skip
        # IDs whose folders do not exist (e.g., after renaming).
        self.split_ids = [
            fid for fid in self.split_ids
            if layout.looks_like_fruit_id(fid) and layout.has_fruit(self.data_source, fid)
        ]

        # latents dics has to be defined before filtering not useable fruits
        self.latents_dict = self.get_latents_dict(pretrain) 
        self.split_ids = self.check_is_useable()

        # loading output of orbbec registration
        registration_outputs = self.get_intrinsics()
        self.Ks = registration_outputs['K']
        self.box = self.find_global_bbox()
        self.files = self.get_instance_filenames()

        # Eagerly preload SDF samples for all unique fruits in this split when sdf_loss=True.
        # Avoids first-epoch lazy-load NFS round-trips. Memory: ~3-5 MB per fruit × ~100-200
        # fruits = ~0.5-1 GB. Skips fruits without samples.npz; remaining lazy fallback in
        # __getitem__ still works.
        if self.sdf_loss:
            unique_fids = set()
            for f in self.files:
                unique_fids.add(layout.fruit_id_from_path(f, self.data_source))
            n_loaded = 0
            n_missing = 0
            for fid in sorted(unique_fids):
                sample_path = layout.sdf_samples(self.data_source, fid)
                if not os.path.exists(sample_path):
                    n_missing += 1
                    continue
                try:
                    npz = np.load(sample_path)
                    self._sdf_samples_cache[fid] = (
                        torch.from_numpy(npz['pos']).float(),
                        torch.from_numpy(npz['neg']).float(),
                    )
                    n_loaded += 1
                except Exception as ex:
                    n_missing += 1
            print(f"[sdf_loss] Preloaded samples.npz: {n_loaded} fruits, missing {n_missing}/{len(unique_fids)}")

        # Build per-(fruit_id, date) volume proxy map from field measurements
        # Used for temporal FT: mask SuperLoss by has_gt_latent and provide
        # vol_proxy / days_since_ref for size-aware contrastive loss.
        if self.vol_map_csv or layout.has_sessions(self.data_source):
            self._build_vol_map()

        # Caliper weak supervision: filter intermediate frames lacking caliper data,
        # load laser GT volumes, mark per-file is_weak.
        if self.caliper_weak_supervision:
            self._build_caliper_weak_supervision()

        # Pre-fill PCD RAM cache (after all file-list filters are applied).
        if self.cache_pcds_in_ram and self.use_orbbec_pcd:
            import time as _t
            _t0 = _t.time()
            n_partial = 0
            n_target = 0
            seen_fids = set()
            for fpath in self.files:
                try:
                    pcd = o3d.io.read_point_cloud(fpath)
                    self._partial_pcd_cache[fpath] = pcd
                    self._partial_pcd_xyz_cache[fpath] = np.asarray(pcd.points, dtype=np.float32)
                    n_partial += 1
                except Exception:
                    pass
                fid = layout.fruit_id_from_path(fpath, self.data_source) or None
                if fid and fid not in seen_fids:
                    seen_fids.add(fid)
                    tpath = layout.gt_cloud(self.data_source, fid)
                    if os.path.isfile(tpath):
                        try:
                            pcd_t = o3d.io.read_point_cloud(tpath)
                            self._target_pcd_cache[fid] = pcd_t
                            self._target_pcd_xyz_cache[fid] = np.asarray(pcd_t.points, dtype=np.float32)
                            n_target += 1
                        except Exception:
                            pass
            print(f"[pcd_cache] Loaded {n_partial} partial PCDs + {n_target} target PCDs in {_t.time()-_t0:.1f}s (xyz_cache enabled)")

        # Pre-load raw camera-frame RGB-D when that path is enabled. Cached as
        # uint8 numpy tuples just like the rasterised path; same lookup key
        # (fpath) so __getitem__'s render hook can serve either source from
        # the same cache dict.
        if self.use_pcd_render_as_rgbd and self.use_raw_camera_rgbd and self.cache_rendered_rgbd and (self.use_orbbec_pcd or self.use_phenobox_pcd):
            import time as _t
            from corepp.baselines.corepp_vanilla.raw_rgbd import load_aligned_rgbd
            _t0 = _t.time()
            n_cached = 0
            n_failed = 0
            for fpath in self.files:
                try:
                    fid = layout.fruit_id_from_path(fpath, self.data_source)
                    frame_id = os.path.splitext(os.path.basename(fpath))[0].replace(self.orbbec_pcd_suffix, '')
                    out = load_aligned_rgbd(
                        fid, frame_id,
                        size=self.render_size,
                        depth_min_mm=self.raw_rgbd_depth_min_mm,
                        depth_max_mm=self.raw_rgbd_depth_max_mm,
                        bbox_margin=self.raw_rgbd_bbox_margin,
                    )
                    if out is None:
                        n_failed += 1
                        continue
                    rgb_f, depth_f, mask_u8 = out
                    rgb_u8 = (np.clip(rgb_f, 0.0, 1.0) * 255).astype(np.uint8)
                    depth_u8 = (np.clip(depth_f, 0.0, 1.0) * 255).astype(np.uint8)
                    self._rendered_rgbd_cache[fpath] = (rgb_u8, depth_u8, mask_u8)
                    n_cached += 1
                except Exception as _e:
                    n_failed += 1
            mb = sum((r.nbytes + d.nbytes + m.nbytes) for r, d, m in self._rendered_rgbd_cache.values()) / 1024**2
            print(f"[raw_rgbd_cache] Pre-loaded {n_cached} aligned camera-frame RGB-D frames "
                  f"({mb:.1f} MB) in {_t.time()-_t0:.1f}s, {n_failed} failed")
            # Skip the PCD-rasterizer pre-cache below since raw RGB-D is the active path.

        # Pre-render colored partial PCDs to 2D RGB-D images and cache as uint8.
        # Used by the vanilla CoRe++ baselines (DoubleEncoder, EncoderBigPooled)
        # to avoid running the splat-based rasterizer in every dataloader worker
        # on every iteration. Memory cost: ~460 KB per frame × N frames.
        elif self.use_pcd_render_as_rgbd and self.cache_rendered_rgbd and (self.use_orbbec_pcd or self.use_phenobox_pcd):
            import time as _t
            from corepp.baselines.corepp_vanilla.render_pcd import render_pcd_to_rgbd
            _t0 = _t.time()
            n_cached = 0
            n_failed = 0
            for fpath in self.files:
                try:
                    if self.cache_pcds_in_ram and fpath in self._partial_pcd_cache:
                        pcd = self._partial_pcd_cache[fpath]
                    else:
                        pcd = o3d.io.read_point_cloud(fpath)
                    _xyz = np.asarray(pcd.points, dtype=np.float32)
                    _cols = np.asarray(pcd.colors, dtype=np.float32)
                    if _cols.size == 0 or _cols.shape[0] != _xyz.shape[0]:
                        _cols = np.full_like(_xyz, 0.5, dtype=np.float32)
                    rgb_f, depth_f, mask_u8 = render_pcd_to_rgbd(
                        _xyz, _cols,
                        size=self.render_size,
                        splat_radius=self.render_splat_radius,
                        fixed_scale=self.render_fixed_scale,
                        fixed_xy_half_extent=self.render_fixed_xy_half_extent,
                        fixed_z_half_extent=self.render_fixed_z_half_extent,
                        target_points=self.render_target_points,
                        target_jitter_mm=self.render_target_jitter_mm,
                        rng_seed=hash(fpath) & 0xFFFFFFFF,
                    )
                    # Quantize float32 [0,1] to uint8 to save 4x memory.
                    rgb_u8 = (np.clip(rgb_f, 0.0, 1.0) * 255).astype(np.uint8)
                    depth_u8 = (np.clip(depth_f, 0.0, 1.0) * 255).astype(np.uint8)
                    self._rendered_rgbd_cache[fpath] = (rgb_u8, depth_u8, mask_u8)
                    n_cached += 1
                except Exception as _e:
                    n_failed += 1
            mb = sum((r.nbytes + d.nbytes + m.nbytes) for r, d, m in self._rendered_rgbd_cache.values()) / 1024**2
            print(f"[render_cache] Pre-rendered {n_cached} RGB-D frames "
                  f"({mb:.1f} MB) in {_t.time()-_t0:.1f}s, {n_failed} failed")




    def _build_vol_map(self):
        """Load field_size_data csv and compute per-frame volume proxy, has_gt_latent, days_since_ref.

        Populates:
          self.vol_map[(fid, date_str)] = vol_ml (ellipsoid proxy)
          self.fruit_last_vol[fid] = last-date vol proxy (fallback: latest measured)
          self.fruit_last_date[fid] = last-date (yyyymmdd)
          self.global_ref_date = earliest date across all fruits (yyyymmdd)
        """
        import pandas as pd
        try:
            from temporal.ellipsoid_proxy import compute_ellipsoid_volume
        except Exception:
            # relative import fallback — keep loader resilient
            compute_ellipsoid_volume = lambda h, w: (4.0 / 3.0) * 3.141592653589793 * (w / 2.0) ** 2 * (h / 2.0)

        if layout.has_sessions(self.data_source):
            # Public release: caliper readings live in the single sessions table.
            rows = layout.read_sessions(self.data_source)
            for r in rows:
                try:
                    h, w = float(r['height_mm']), float(r['width_mm'])
                    if h <= 0 or w <= 0:
                        continue
                    self.vol_map[(r['sample_id'], r['session_date'])] = \
                        compute_ellipsoid_volume(h, w) / 1000.0
                except (ValueError, KeyError):
                    continue
            print(f"[vol_map] Loaded {len(self.vol_map)} caliper sessions from "
                  f"{layout.sessions_csv(self.data_source)}")
        else:
            csv_path = self.vol_map_csv
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(self.data_source, csv_path)
            if not os.path.exists(csv_path):
                print(f"[vol_map] WARNING: not found: {csv_path}. Skipping vol_map build.")
                return

            df = pd.read_csv(csv_path)
            df['date'] = df['date'].astype(str)
            for _, r in df.iterrows():
                try:
                    h, w = float(r['height']), float(r['width'])
                    if h <= 0 or w <= 0: continue
                    vol_ml = compute_ellipsoid_volume(h, w) / 1000.0
                    self.vol_map[(str(r['fruit_id']), r['date'])] = vol_ml
                except Exception:
                    continue

        # Determine last-date per fruit from self.files (the orbbec frames actually used)
        # Working tree names frames YYYYMMDD_..., the release names them YYYYMMDD.ply.
        date_pat = re.compile(r"^(\d{8})")
        fid_dates = {}
        all_dates = []
        for fpath in self.files:
            fid = layout.fruit_id_from_path(fpath, self.data_source)
            fname = os.path.basename(fpath)
            m = date_pat.match(fname)
            if m:
                d = m.group(1)
                fid_dates.setdefault(fid, []).append(d)
                all_dates.append(d)

        for fid, ds in fid_dates.items():
            ds_sorted = sorted(set(ds))
            # Prefer the measured last-date; fallback to latest measurement available
            last_d = ds_sorted[-1]
            if (fid, last_d) in self.vol_map:
                self.fruit_last_vol[fid] = self.vol_map[(fid, last_d)]
                self.fruit_last_date[fid] = last_d
            else:
                # find latest measurement available for this fid
                measured = [d for d in ds_sorted if (fid, d) in self.vol_map]
                if measured:
                    latest = measured[-1]
                    self.fruit_last_vol[fid] = self.vol_map[(fid, latest)]
                    self.fruit_last_date[fid] = latest

        self.global_ref_date = min(all_dates) if all_dates else None
        # Precompute per-file has_gt and vol for use by samplers / collate logic.
        self.has_gt_per_file = []
        self.vol_per_file = []
        self.vol_last_per_file = []  # per-fruit last-date vol, replicated per file
        for fpath in self.files:
            fid = layout.fruit_id_from_path(fpath, self.data_source)
            fname = os.path.basename(fpath)
            m = date_pat.match(fname)
            date_str = m.group(1) if m else None
            vol, has_gt = self._frame_vol_and_has_gt(fid, date_str)
            self.has_gt_per_file.append(bool(has_gt))
            self.vol_per_file.append(float(vol))
            self.vol_last_per_file.append(float(self.fruit_last_vol.get(fid, -1.0)))
        n_has_gt = sum(self.has_gt_per_file)
        print(f"[vol_map] frames with vol: {len(self.vol_map)}  fruits with last_vol: {len(self.fruit_last_vol)}  "
              f"has_gt per-file: {n_has_gt}/{len(self.has_gt_per_file)} = {100.0*n_has_gt/max(1,len(self.has_gt_per_file)):.1f}%  "
              f"global_ref_date: {self.global_ref_date}")

    def _build_caliper_weak_supervision(self):
        """Caliper weak supervision setup.

        Requires self.vol_map (already built) and gt_volume_csv (laser GT per fruit).
        Drops intermediate frames lacking caliper data; remaining intermediate frames
        are marked is_weak=True. Last-date frames remain strong supervision.

        Populates:
            self.last_laser_vol[fid] = V_laser (ml) from gt_volume_csv
            self.is_weak_per_file = [bool, ...] parallel to self.files
        """
        import pandas as pd
        # Need vol_map for proxy volumes
        if not self.vol_map:
            print("[caliper_weak] WARNING: vol_map empty; pass vol_map_csv to enable. Skipping caliper weak supervision.")
            return

        # Load laser GT volumes per fruit
        if layout.has_sessions(self.data_source):
            for r in layout.read_sessions(self.data_source):
                try:
                    v = float(r['harvest_volume_ml'])
                except (ValueError, KeyError):
                    continue
                if v > 0:
                    self.last_laser_vol[r['sample_id']] = v
            print(f"[caliper_weak] Loaded harvest volumes: {len(self.last_laser_vol)} fruits "
                  f"from {layout.sessions_csv(self.data_source)}")
        elif self.gt_volume_csv:
            csv_path = self.gt_volume_csv
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(self.data_source, csv_path)
            if os.path.exists(csv_path):
                gt_df = pd.read_csv(csv_path)
                for _, r in gt_df.iterrows():
                    fid = str(r['label']) if 'label' in gt_df.columns else str(r.get('fruit_id', ''))
                    vol = r.get('volume_ml', None)
                    try:
                        v = float(vol)
                        if v > 0:
                            self.last_laser_vol[fid] = v
                    except Exception:
                        pass
                print(f"[caliper_weak] Loaded laser GT volumes: {len(self.last_laser_vol)} fruits from {csv_path}")
            else:
                print(f"[caliper_weak] WARNING: gt_volume_csv not found: {csv_path}. Falling back to caliper-derived V_laser_proxy.")
        else:
            print("[caliper_weak] No gt_volume_csv; using caliper-derived V_proxy_last as V_laser proxy.")

        # Multiplicative correction of per-frame ellipsoid proxy: assumes a constant
        # shape factor across growth stages, which is more biologically plausible than
        # additive correction (additive would say a 1ml infant fruit needs +49ml fixed
        # offset → 50ml, vs multiplicative which scales it as ratio·ellipsoid).
        #   ratio(f) = laser(f) / ellipsoid_at_last(f)
        #   corrected(f, d) = ellipsoid(f, d) * ratio(f)
        # This makes intermediate-frame weak GT match the multiplicative definition used
        # for eval (corrected_gt_mult_ml).
        if bool(getattr(self, 'caliper_correct_proxy', False)) and self.last_laser_vol:
            n_corrected = 0
            n_skipped = 0
            ratio_min, ratio_max, ratio_sum = float('inf'), float('-inf'), 0.0
            for fid, laser in self.last_laser_vol.items():
                last_d = self.fruit_last_date.get(fid)
                if not last_d:
                    n_skipped += 1; continue
                last_ellip = self.vol_map.get((fid, last_d))
                if last_ellip is None or last_ellip <= 0:
                    n_skipped += 1; continue
                ratio = laser / last_ellip
                ratio_min = min(ratio_min, ratio); ratio_max = max(ratio_max, ratio); ratio_sum += ratio
                for k in [k for k in self.vol_map.keys() if k[0] == fid]:
                    new_v = self.vol_map[k] * ratio
                    if new_v > 0:
                        self.vol_map[k] = new_v
                self.fruit_last_vol[fid] = laser  # corrected last = laser exactly
                n_corrected += 1
            if n_corrected > 0:
                avg_ratio = ratio_sum / n_corrected
                print(f"[caliper_weak] Applied corrected_proxy (multiplicative) to {n_corrected} fruits "
                      f"(skipped {n_skipped}). Per-fruit ratio: min={ratio_min:.2f} max={ratio_max:.2f} mean={avg_ratio:.2f}")

        # Filter self.files: keep last camera-date frames always (laser GT applies regardless of
        # caliper availability); drop intermediate frames lacking caliper measurement.
        # NOTE: "last" = max camera observation date per fruit (in self.files), NOT last caliper
        # date — last camera observation may be after last caliper measurement.
        # Working tree names frames YYYYMMDD_..., the release names them YYYYMMDD.ply.
        date_pat = re.compile(r"^(\d{8})")
        # Pre-compute max camera date per fruit from self.files
        last_camera_date_by_fid = {}
        for fpath in self.files:
            fid = layout.fruit_id_from_path(fpath, self.data_source)
            fname = os.path.basename(fpath)
            m = date_pat.match(fname)
            if not m:
                continue
            d = m.group(1)
            if fid not in last_camera_date_by_fid or d > last_camera_date_by_fid[fid]:
                last_camera_date_by_fid[fid] = d

        kept = []
        is_weak_kept = []
        n_drop_no_caliper = 0
        n_strong = 0
        n_weak = 0
        for fpath in self.files:
            fid = layout.fruit_id_from_path(fpath, self.data_source)
            fname = os.path.basename(fpath)
            m = date_pat.match(fname)
            if not m:
                kept.append(fpath); is_weak_kept.append(False); n_strong += 1
                continue
            date_str = m.group(1)
            last_cam_d = last_camera_date_by_fid.get(fid)
            is_last = (last_cam_d is not None and date_str == last_cam_d)
            if is_last:
                # Strong (last camera frame): keep regardless of caliper
                kept.append(fpath); is_weak_kept.append(False); n_strong += 1
            else:
                # Intermediate: keep only if caliper-derived volume is available
                if (fid, date_str) in self.vol_map and self.vol_map[(fid, date_str)] > 0:
                    kept.append(fpath); is_weak_kept.append(True); n_weak += 1
                else:
                    n_drop_no_caliper += 1
        self.files = kept
        self.is_weak_per_file = is_weak_kept

        # Sync downstream lists if they were already built by _build_vol_map (rebuild for filtered set)
        if hasattr(self, 'has_gt_per_file') and self.has_gt_per_file:
            new_has_gt, new_vol, new_vol_last = [], [], []
            for fpath in self.files:
                fid = layout.fruit_id_from_path(fpath, self.data_source)
                fname = os.path.basename(fpath)
                m = date_pat.match(fname)
                date_str = m.group(1) if m else None
                vol, has_gt = self._frame_vol_and_has_gt(fid, date_str)
                new_has_gt.append(bool(has_gt))
                new_vol.append(float(vol))
                new_vol_last.append(float(self.fruit_last_vol.get(fid, -1.0)))
            self.has_gt_per_file = new_has_gt
            self.vol_per_file = new_vol
            self.vol_last_per_file = new_vol_last

        print(f"[caliper_weak] Active. files: strong={n_strong}, weak={n_weak}, "
              f"dropped (intermediate w/o caliper)={n_drop_no_caliper}, "
              f"laser_vol fruits={len(self.last_laser_vol)}")

    def _parse_date_from_frame_id(self, frame_id):
        m = re.match(r"^(\d{8})", str(frame_id))
        return m.group(1) if m else None

    def _days_since_ref(self, date_str):
        if not self.global_ref_date or not date_str:
            return -1
        try:
            from datetime import datetime
            d = datetime.strptime(date_str, "%Y%m%d")
            r = datetime.strptime(self.global_ref_date, "%Y%m%d")
            return (d - r).days
        except Exception:
            return -1

    def _frame_vol_and_has_gt(self, fruit_id, date_str):
        """Returns (vol_proxy, has_gt_latent) for a given frame.
        has_gt_latent: True if |vol-vol_last|/vol_last < thr (includes last-date since diff=0).
        """
        vol = self.vol_map.get((fruit_id, date_str), -1.0) if date_str else -1.0
        last_v = self.fruit_last_vol.get(fruit_id)
        has_gt = False
        if vol is not None and vol > 0 and last_v is not None and last_v > 0:
            rel = abs(vol - last_v) / last_v
            has_gt = rel < self.gt_vol_similar_thr
        return float(vol if vol is not None else -1.0), bool(has_gt)

    def get_latents_dict(self, path):
        "create dictionary of pairs fruit_id:latent given pretrained model"
        latent_dictionary = {}
        if not path or not os.path.exists(path) or not os.path.isdir(path):
            # If path is invalid or is a file (e.g. checkpoint), skip loading pre-optimized latents
            return latent_dictionary
        
        try:
            for fname in os.listdir(path):
                if not fname.endswith('.pth'): continue
                try:
                    latent = torch.load(os.path.join(path, fname), map_location='cpu')
                    key = fname[:-4]
                    latent_dictionary[key] = latent
                except Exception:
                    pass
        except OSError:
            pass
        return latent_dictionary

    @staticmethod
    def preprocess_images(rgb, depth, mask, intrinsic_file, detection_input="box"):
        """ 
        Crop images and depths around the fruit
        
        Args:
            rgb: full image as loaded from opencv
            depth: full depth as generated from orbbec camera
            mask: semantic mask
                        
        Returns: 
            rgb: rgb cropped around the fruit
            depth: depth cropped around the fruit
            mask: mask cropped around the fruit
        """

        # create partial point cloud
        intrinsics, depth_scale = load_intrinsics(intrinsic_file)
        # Ensure depth is 2D: handle cases where depth is provided as multi-channel (e.g., XYZ[D])
        if depth.ndim == 3:
            if depth.shape[2] == 1:
                depth = depth[..., 0]
            elif depth.shape[2] >= 3:
                # Use Z channel as depth
                depth = depth[..., 2]
            else:
                # Fallback to last channel
                depth = depth[..., -1]

        img_mask = np.multiply(rgb, np.expand_dims(mask, axis=2))
        dimg_mask = np.multiply(depth, mask)
        z_min, z_max = histogram_filtering(depth, mask, 50, 0.05)
        dimg_mask[dimg_mask < z_min] = 0
        dimg_mask[dimg_mask > z_max] = 0

        rgb_mask = o3d.geometry.Image((img_mask).astype(np.uint8))
        depth_mask = o3d.geometry.Image(dimg_mask)
        rgbd_mask = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_mask, depth_mask, depth_scale=depth_scale, depth_trunc=0.4, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_mask, intrinsics)
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_filtered = pcd.select_by_index(ind)
        pcd_filtered.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # cropping
        offset = 20
        indices = np.where(mask)
        min_i = indices[0].min() - offset
        min_j = indices[1].min() - offset
        max_i = indices[0].max() + offset
        max_j = indices[1].max() + offset

        if min_i < 0: min_i = 0
        if min_j < 0: min_j = 0

        rgb = rgb[min_i:max_i, min_j:max_j, :]
        depth = depth[min_i:max_i, min_j:max_j]
        mask = mask[min_i:max_i, min_j:max_j]

        ## mask the RGB and depth image and apply histogram filtering on the depth image
        if detection_input == "mask":
            depth[depth < z_min] = 0
            depth[depth > z_max] = 0
            depth = depth * mask
            mask = np.clip(depth, 0.0, 1.0)
            rgb = rgb * np.expand_dims(mask.astype(bool), 2)
        
        w = max_i - min_i
        h = max_j - min_j

        return rgb, depth, mask, (min_i, min_j), (w,h), np.ones(mask.shape), pcd_filtered

    @staticmethod
    def load_K(path):
        """ 
        Load intrinsic params
        
        Args:
            path: path to json
                        
        Returns: 
            k: intrinsic matrix
        """
        f = open(path,'r')
        data = json.load(f)['intrinsic_matrix']
        k = np.reshape(data, (3, 3), order='F') 
        return k

    @staticmethod
    def bbox2dict(bb):
        """
        Convert bounding box from array with shapes 2x3 to dictionary

        Args:
            bb: bounding box extreme points

        Returns:
            box: bounding box as dictionary
        """

        x_min, y_min, z_min = bb[0] 
        x_max, y_max, z_max = bb[1]

        box = {}
        box['xmin'] = x_min
        box['xmax'] = x_max
        box['ymin'] = y_min
        box['ymax'] = y_max
        box['zmin'] = z_min
        box['zmax'] = z_max
        return box

    def update_intrinsic(self, crop_origin, shape_origin, k):
        pad_x = self.pad_size - shape_origin[0]
        pad_y = self.pad_size - shape_origin[1]
        
        intrinsic = copy.deepcopy(k)

        intrinsic[0,2] -= crop_origin[1] - pad_y/2 + 0.5
        intrinsic[1,2] -= crop_origin[0] - pad_x/2 + 0.5
        return intrinsic

    def find_global_bbox(self):
        # Compute bbox across ALL splits (train/val/test) to ensure consistent normalization
        # regardless of which split is currently loaded.
        
        all_ids = set()
        
        # Load all splits from the split json
        if self.split_json_path and os.path.exists(self.split_json_path):
             with open(self.split_json_path) as f:
                all_splits = json.load(f)
                for split_name, ids in all_splits.items():
                    if isinstance(ids, list):
                        for fid in ids:
                            if isinstance(fid, str):
                                all_ids.add(fid)
        
        # Filter strictly by folder existence, similar to __init__

        valid_ids = []
        for fid in all_ids:
            if layout.looks_like_fruit_id(fid) and layout.has_fruit(self.data_source, fid):
                valid_ids.append(fid)
                
        # If fallback needed (no json or empty), use local split_ids
        if not valid_ids:
            valid_ids = self.split_ids

        dmax = 0
        for fruit_id in valid_ids:
            target_pcd_fname = layout.gt_cloud(self.data_source, fruit_id)
            if not os.path.exists(target_pcd_fname):
                continue
            target_pcd = o3d.io.read_point_cloud(target_pcd_fname)
            box = target_pcd.get_axis_aligned_bounding_box()
            dx, dy, dz = np.abs(box.get_max_bound() - box.get_min_bound())
            local_dmax = max(dx,dy,dz)
            if local_dmax > dmax:
                dmax = local_dmax
        
        global_bbox = {'xmin': -dmax/2, 'xmax': dmax/2, 
                       'ymin': -dmax/2, 'ymax': dmax/2, 
                       'zmin': -dmax/2, 'zmax': dmax/2}    
        return global_bbox
        
    def get_instance_filenames(self):
        """ 
        Load image names
        
        Returns: 
            files: list of file names ( of rgb frames) to be used for training/testing
        """

        files = []
        for count_id, id in enumerate(self.split_ids):

            ####################### 
            if self.overfit:
                if count_id == 1: break
            #######################  

            if self.use_phenobox_pcd:
                pc_dir = os.path.join(self.data_source, id, self.phenobox_subdir)
                if not os.path.isdir(pc_dir):
                    continue
                ply_files = [f for f in os.listdir(pc_dir) if f.lower().endswith('.ply')]
                ply_files.sort()

                if self.partial_pcd_count is not None and len(ply_files) > self.partial_pcd_count:
                    import random
                    # Deterministically select subset based on partial_pcd_count
                    ply_files = random.sample(ply_files, self.partial_pcd_count)
                    ply_files.sort()

                for fname in ply_files:
                    files.append(os.path.join(pc_dir, fname))
            elif self.use_orbbec_pcd:
                # Load preprocessed point clouds from orbbec/pointclouds
                pc_dir = layout.partial_dir(self.data_source, id)
                if not os.path.isdir(pc_dir):
                    continue
                
                # Get all files with correct suffix
                candidates = [f for f in os.listdir(pc_dir) 
                            if f.lower().endswith(f'{self.orbbec_pcd_suffix}.ply')]
                
                if self.temporal_loading:
                     ply_files = candidates
                else:
                    # Filter to only keep files from the LATEST date. Names are
                    # YYYYMMDD_... in the working tree and YYYYMMDD.ply in the release.
                    latest_date = None
                    date_pattern = re.compile(r"^(\d{8})")

                    for f in candidates:
                        match = date_pattern.match(f)
                        if match:
                            date_str = match.group(1)
                            if latest_date is None or date_str > latest_date:
                                latest_date = date_str

                    if latest_date:
                        ply_files = [f for f in candidates if f.startswith(latest_date)]
                    else:
                        # Fallback: if no dates found, use all files (legacy behavior)
                        ply_files = candidates

                ply_files.sort()
                for fname in ply_files:
                    files.append(os.path.join(pc_dir, fname))
            else:
                lst_images = os.listdir(self.data_source+id+'/orbbec/masks/')
                lst_images.sort()
                for count_img, fname in enumerate(lst_images):
                    files.append(self.data_source+id+'/orbbec/color/'+fname)

        files.sort() 
        return files

    def get_intrinsics(self):
        """ 
        Load registration parameters
        
        Args:
      
        Returns: 
            data: dictionary containing intrinsic, boudning box 
                  and poses of each frame of each fruit
        """       
        Ks = {}
        for fruit in self.split_ids:
            path = os.path.join(self.data_source, fruit)
            k_path = os.path.join(path, 'orbbec/intrinsic.json')
            # Phenobox / point-cloud-only pipelines don't require camera intrinsics.
            # Some samples (e.g., later Sxx) may not have orbbec/ metadata at all.
            try:
                k = self.load_K(k_path)
            except FileNotFoundError:
                k = np.eye(3, dtype=np.float32)
            Ks[fruit] = k
        
        return {'K': Ks}

    def compute_target_sdf(self, rgb, depth, pose, k):

        bbox = self.global_bbox

        grid_density_complex = self.grid_density * 1j
        X, Y, Z = np.mgrid[bbox['xmin']:bbox['xmax']:grid_density_complex, bbox['ymin']:bbox['ymax']:grid_density_complex, bbox['zmin']:bbox['zmax']:grid_density_complex]
        grid = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1).reshape((-1, 3))

        bb = o3d.geometry.AxisAlignedBoundingBox()
        bb = bb.create_from_points(o3d.utility.Vector3dVector(grid))

        grid_pcd = o3d.geometry.PointCloud()
        grid_pcd.points = o3d.utility.Vector3dVector(grid)
        grid_pcd.paint_uniform_color(np.array((.5,.5,.5)))

        invpose = np.linalg.inv(pose)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb),
                                                                  o3d.geometry.Image(depth),
                                                                  depth_scale = 1000,
                                                                  depth_trunc=1.0,
                                                                  convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(height=rgb.shape[0],
                                 width=rgb.shape[1],
                                 fx=k[0,0],
                                 fy=k[1,1],
                                 cx=k[0,2],
                                 cy=k[1,2],
                                 )

        volume_len = self.global_bbox['xmax'] - self.global_bbox['xmin']
        volume = o3d.pipelines.integration.UniformTSDFVolume(length=volume_len,
                                                             resolution=self.grid_density,
                                                             sdf_trunc= self.sdf_trunc,
                                                             origin = np.full((3,1),-volume_len/2),
                                                             color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)

        volume.integrate(rgbd, intrinsic, invpose)
        tsdf = volume.extract_volume_tsdf()

        val = np.asarray(tsdf)[:,0]
        weight = np.asarray(tsdf)[:,1]

        val = val.reshape((self.grid_density**3,1))
        weight = weight.reshape((self.grid_density**3,1))
        return val, weight, grid

    def check_is_useable(self):
        useable_fruits = []
        for fruit in self.split_ids:
            # print('checking {} usability...'.format(fruit))
            dataset_json = os.path.join(self.data_source, fruit, 'dataset.json')
            if not os.path.isfile(dataset_json):
                # Many datasets don't have dataset.json for all samples.
                # When require_dataset_json=False, allow samples without this metadata file.
                if self.require_dataset_json:
                    continue
                useable_fruits.append(fruit)
                continue
            try:
                with open(dataset_json) as json_file:
                    specs = json.load(json_file)
                if specs.get('is_useable', False):
                    useable_fruits.append(fruit)
            except Exception:
                # Best-effort: skip malformed JSON instead of crashing
                continue
        # Filter by latent availability if supervised_3d is active
        if self.supervised_3d and self.latents_dict:
            useable_fruits = [f for f in useable_fruits if f in self.latents_dict]
            
        return useable_fruits

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        _fast_pcd = os.environ.get("TCORE_PCD_FAST", "0") not in ("0", "", "false", "False")
        _fast_partial_xyz = None
        _fast_target_xyz = None

        # ugly stuff to get consistent ids
        if self.use_phenobox_pcd:
            # files[idx] points to a phenobox partial PLY
            fruit_id = layout.fruit_id_from_path(self.files[idx], self.data_source)
            frame_id = os.path.splitext(os.path.basename(self.files[idx]))[0]
            target_pcd_fname = layout.gt_cloud(self.data_source, fruit_id)
            target_pcd = o3d.io.read_point_cloud(target_pcd_fname)
            partial_pcd = o3d.io.read_point_cloud(self.files[idx])
            crop_dim = (self.pad_size, self.pad_size)
            # dummy placeholders to satisfy downstream keys (not used by point-cloud encoders)
            # create tensors in expected shapes: rgb [3,H,W], depth [1,H,W], masks [1,H,W]
            rgb = torch.zeros((3, self.pad_size, self.pad_size), dtype=torch.float32)
            depth = torch.zeros((1, self.pad_size, self.pad_size), dtype=torch.float32)
            mask = torch.ones((1, self.pad_size, self.pad_size), dtype=torch.uint8)
            padding_mask = torch.ones((1, self.pad_size, self.pad_size), dtype=torch.uint8)
            k = self.Ks.get(fruit_id, np.eye(3))
            crop_origin = (0, 0)
        elif self.use_orbbec_pcd:
            # files[idx] points to a preprocessed orbbec PLY
            fruit_id = layout.fruit_id_from_path(self.files[idx], self.data_source)
            frame_id = os.path.splitext(os.path.basename(self.files[idx]))[0]
            # Remove the suffix to get the base frame_id
            frame_id = frame_id.replace(self.orbbec_pcd_suffix, '')
            target_pcd_fname = layout.gt_cloud(self.data_source, fruit_id)
            if self.cache_pcds_in_ram and fruit_id in self._target_pcd_cache:
                target_pcd = self._target_pcd_cache[fruit_id]
            else:
                target_pcd = o3d.io.read_point_cloud(target_pcd_fname)
            if self.cache_pcds_in_ram and self.files[idx] in self._partial_pcd_cache:
                partial_pcd = self._partial_pcd_cache[self.files[idx]]
            else:
                partial_pcd = o3d.io.read_point_cloud(self.files[idx])
            if _fast_pcd and self.files[idx] in self._partial_pcd_xyz_cache:
                _fast_partial_xyz = _fast_subsample_xyz(
                    self._partial_pcd_xyz_cache[self.files[idx]], self.pad_size
                )
            if _fast_pcd and fruit_id in self._target_pcd_xyz_cache:
                _fast_target_xyz = self._target_pcd_xyz_cache[fruit_id]
            crop_dim = (self.pad_size, self.pad_size)
            # dummy placeholders to satisfy downstream keys (not used by point-cloud encoders)
            rgb = torch.zeros((3, self.pad_size, self.pad_size), dtype=torch.float32)
            depth = torch.zeros((1, self.pad_size, self.pad_size), dtype=torch.float32)
            mask = torch.ones((1, self.pad_size, self.pad_size), dtype=torch.uint8)
            padding_mask = torch.ones((1, self.pad_size, self.pad_size), dtype=torch.uint8)
            k = self.Ks.get(fruit_id, np.eye(3))
            crop_origin = (0, 0)
        else:
            fruit_id = layout.fruit_id_from_path(self.files[idx], self.data_source)
            frame_id = self.files[idx].split('/')[-1][:-4]
            
            target_pcd_fname = layout.gt_cloud(self.data_source, fruit_id)
            target_pcd = o3d.io.read_point_cloud(target_pcd_fname)

            image_path = self.files[idx]
            depth_path = image_path.replace('color', 'depth')
            depth_path = depth_path.replace('png', 'npy')
            mask_path = image_path.replace('color', 'masks')

            orbbec_dir = os.path.dirname(os.path.dirname(self.files[idx]))
            intrinsic_file = os.path.join(orbbec_dir, "intrinsic.json")

            rgb = cv.imread(image_path)
            rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
            depth = np.load(depth_path).astype(np.float32)
            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE) // 255

            # getting intrinsic and camera pose
            k = self.Ks[fruit_id]
            rgb, depth, mask, crop_origin, crop_dim, padding_mask, partial_pcd = self.preprocess_images(rgb, depth, mask, intrinsic_file, self.detection_input)

        ## downsample the partial cloud robustly
        if _fast_partial_xyz is not None:
            # fast path: vectorized numpy subsample already produced pad_size×3 xyz
            down_pcd = None
            num_points = int(self._partial_pcd_xyz_cache[self.files[idx]].shape[0])
        elif (num_points := len(partial_pcd.points)) == 0:
            # Fallback: create a dummy cloud at the origin to keep pipeline running
            down_pcd = o3d.geometry.PointCloud()
            down_pcd.points = o3d.utility.Vector3dVector(np.zeros((self.pad_size, 3), dtype=np.float32))
        else:
            ratio = (self.pad_size + 100) / float(num_points)
            ratio = min(1.0, max(0.0, ratio))
            down_pcd = partial_pcd.random_down_sample(ratio)

            # Ensure exactly pad_size points by sampling with/without replacement
            current_n = len(down_pcd.points)
            if current_n == 0:
                # If random_down_sample returned empty, sample directly from original
                select_n = min(self.pad_size, num_points)
                indices = np.random.choice(num_points, size=select_n, replace=False)
                down_pcd = partial_pcd.select_by_index(indices.tolist())
                current_n = len(down_pcd.points)

            if current_n < self.pad_size:
                # Upsample with replacement from the available points
                indices = np.random.choice(current_n, size=self.pad_size, replace=True)
                down_pcd = down_pcd.select_by_index(indices.tolist())
            elif current_n > self.pad_size:
                # Downsample without replacement to the target size
                indices = np.random.choice(current_n, size=self.pad_size, replace=False)
                down_pcd = down_pcd.select_by_index(indices.tolist())

        intrinsic = self.update_intrinsic(crop_origin, depth.shape, k) if not self.use_phenobox_pcd else k

        item = {
            'dimension': crop_dim,
            'fruit_id': fruit_id,
            'frame_id': frame_id,
            # 'pose': pose,
            'K': intrinsic,
            'bbox': self.box,
        }

        # Volume proxy / has_gt_latent / days_since_ref (temporal FT support).
        # Populated only when vol_map was built (vol_map_csv given in constructor).
        _date_str = self._parse_date_from_frame_id(frame_id) if hasattr(self, '_parse_date_from_frame_id') else None
        if (self.vol_map_csv or layout.has_sessions(self.data_source)) and _date_str:
            _vol, _has_gt = self._frame_vol_and_has_gt(fruit_id, _date_str)
            _days = self._days_since_ref(_date_str)
        else:
            _vol, _has_gt, _days = -1.0, False, -1
        item['vol_proxy'] = torch.tensor(_vol, dtype=torch.float32)
        item['has_gt_latent'] = torch.tensor(_has_gt, dtype=torch.bool)
        item['days_since_ref'] = torch.tensor(_days, dtype=torch.long)

        # SDF loss supervision via pre-computed laser/samples.npz (DeepSDF format).
        # Loads pos/neg samples lazily, subsamples per-iter, and stores xyz + sdf
        # for use in train.py's SDF loss block. Only active when sdf_loss=True.
        if self.sdf_loss:
            sample_path = layout.sdf_samples(self.data_source, fruit_id)
            if os.path.exists(sample_path):
                if fruit_id not in self._sdf_samples_cache:
                    npz = np.load(sample_path)
                    self._sdf_samples_cache[fruit_id] = (
                        torch.from_numpy(npz['pos']).float(),  # (Npos, 4) [x,y,z,sdf]
                        torch.from_numpy(npz['neg']).float(),
                    )
                pos_t, neg_t = self._sdf_samples_cache[fruit_id]
                half = self.sdf_subsample // 2
                # Drop NaNs (defensive — same as deepsdf data.py)
                pos_t = pos_t[~torch.isnan(pos_t).any(dim=1)]
                neg_t = neg_t[~torch.isnan(neg_t).any(dim=1)]
                if pos_t.shape[0] >= half and neg_t.shape[0] >= half:
                    pos_idx = torch.randint(0, pos_t.shape[0], (half,))
                    neg_idx = torch.randint(0, neg_t.shape[0], (half,))
                    samples = torch.cat([pos_t[pos_idx], neg_t[neg_idx]], dim=0)  # (subsample, 4)
                    item['target_sdf_xyz'] = samples[:, :3]                       # (N, 3)
                    item['target_sdf'] = samples[:, 3:4]                          # (N, 1)
                    item['target_sdf_weights'] = torch.ones(samples.shape[0], 1)  # (N, 1)
                else:
                    # Not enough samples — skip SDF supervision for this item
                    item['target_sdf_xyz'] = None
                    item['target_sdf'] = None
                    item['target_sdf_weights'] = None
            else:
                item['target_sdf_xyz'] = None
                item['target_sdf'] = None
                item['target_sdf_weights'] = None

        if self.normalize_depth:
            depth_original = copy.deepcopy(depth)
            depth = (depth_original - self.depth_min) / (self.depth_max - self.depth_min)
            depth[depth_original == 0] = 0
        else:
            depth = depth / self.depth_max

        if self.color_tf and not self.use_phenobox_pcd and not self.use_orbbec_pcd:
            rgb = torch.from_numpy(np.array(self.color_tf(rgb))).permute(2,0,1)
        else:
            if self.tf and not self.use_phenobox_pcd and not self.use_orbbec_pcd:
                rgb = torch.from_numpy(np.array(self.tf(rgb))).permute(2,0,1)

        if self.tf and not self.use_phenobox_pcd and not self.use_orbbec_pcd:
            depth = torch.from_numpy(np.array(self.tf(depth))).unsqueeze(dim=0) 
            mask = torch.from_numpy(np.array(self.tf(mask))).unsqueeze(dim=0)
            padding_mask = torch.from_numpy(np.array(self.tf(padding_mask))).unsqueeze(dim=0)

        item['rgb'] = rgb.float()/255 
        item['depth'] = depth.float()
        item['mask'] = mask
        item['padding_mask'] = padding_mask
        
        # Compute bbox normalization parameters once for reuse
        bbox_center = np.array([
            (self.box['xmin'] + self.box['xmax']) / 2,
            (self.box['ymin'] + self.box['ymax']) / 2,
            (self.box['zmin'] + self.box['zmax']) / 2
        ], dtype=np.float32)
        bbox_size = np.array([
            self.box['xmax'] - self.box['xmin'],
            self.box['ymax'] - self.box['ymin'],
            self.box['zmax'] - self.box['zmin']
        ], dtype=np.float32)
        bbox_size = np.maximum(bbox_size, 1e-6)  # Avoid division by zero
        bbox_half = bbox_size / 2.0
        
        # Only include target_pcd when 3D loss is used; it has variable length and breaks batching otherwise
        if self.sdf_loss:
            if _fast_target_xyz is not None:
                target_pcd_raw = _fast_target_xyz
            else:
                target_pcd_raw = np.asarray(target_pcd.points, dtype=np.float32)
            if target_pcd_raw.shape[0] > 0:
                # Normalize target_pcd to match partial_pcd normalization
                target_pcd_normalized = (target_pcd_raw - bbox_center) / bbox_half
                item['target_pcd'] = torch.from_numpy(target_pcd_normalized)
            else:
                item['target_pcd'] = torch.from_numpy(target_pcd_raw)

        # -------------------------------------------------------------------------
        # 1. Prepare Raw Point Cloud
        # -------------------------------------------------------------------------
        if _fast_partial_xyz is not None:
            partial_pcd_raw = _fast_partial_xyz
        else:
            partial_pcd_raw = np.asarray(down_pcd.points, dtype=np.float32)

        # Pre-padding augmentation (e.g. random occlusion on REAL points only,
        # before replicate-padding so dropped points are not re-added via replication).
        if self.pre_pad_pcd_tf is not None and partial_pcd_raw.shape[0] > 0:
            partial_pcd_raw = self.pre_pad_pcd_tf(partial_pcd_raw)

        # Force exact point count. Preserve real-vs-padded separation:
        #   real points occupy indices [0, n_real), padding/replication fills [n_real, pad_size).
        # This lets encoders apply masked pooling (e.g. masked mean) over only real pts.
        n_real_actual = partial_pcd_raw.shape[0]
        if self.pad_size > 0:
            current_n = n_real_actual
            if current_n < self.pad_size:
                if current_n > 0:
                    try:
                        fill_indices = np.random.choice(current_n, size=self.pad_size - current_n, replace=True)
                        partial_pcd_raw = np.concatenate([partial_pcd_raw, partial_pcd_raw[fill_indices]], axis=0)
                    except ValueError:
                        partial_pcd_raw = np.zeros((self.pad_size, 3), dtype=np.float32)
                        n_real_actual = 0
                else:
                    partial_pcd_raw = np.zeros((self.pad_size, 3), dtype=np.float32)
                    n_real_actual = 0
            elif current_n > self.pad_size:
                indices = np.random.choice(current_n, size=self.pad_size, replace=False)
                partial_pcd_raw = partial_pcd_raw[indices]
                n_real_actual = self.pad_size

        # -------------------------------------------------------------------------
        # 2. Augmentation (on raw coords)
        # -------------------------------------------------------------------------
        # NOTE: Apply augmentation only to the input point cloud.
        if self.pcd_tf:
            partial_pcd_raw = self.pcd_tf(partial_pcd_raw)

        # -------------------------------------------------------------------------
        # 3. Normalization (Conditional)
        # -------------------------------------------------------------------------
        if self.normalize_pcd:
            if partial_pcd_raw.shape[0] > 0:
                # Normalize: center at origin and scale to fit in [-1, 1]
                # WARNING: This removes global scale information!
                partial_pcd_normalized = (partial_pcd_raw - bbox_center) / bbox_half
            else:
                partial_pcd_normalized = partial_pcd_raw
        else:
             # Use raw physical coordinates (essential for learning volume/scale)
            partial_pcd_normalized = partial_pcd_raw

        # -------------------------------------------------------------------------
        # 4. Pack Tensors
        # -------------------------------------------------------------------------
        item['partial_pcd'] = torch.from_numpy(partial_pcd_normalized)
        item['partial_pcd_world'] = torch.from_numpy(partial_pcd_raw)
        item['bbox_center_tensor'] = torch.from_numpy(bbox_center)
        item['bbox_scale_tensor'] = torch.from_numpy(bbox_half)
        # Real-point mask: True for indices in [0, n_real_actual), False elsewhere (padding)
        _real_mask_np = np.zeros(self.pad_size if self.pad_size > 0 else partial_pcd_raw.shape[0], dtype=bool)
        _real_mask_np[:min(n_real_actual, _real_mask_np.shape[0])] = True
        item['partial_pcd_real_mask'] = torch.from_numpy(_real_mask_np)
        item['n_real'] = int(n_real_actual)

        if self.supervised_3d:
            try:
                trained_latent = self.latents_dict[fruit_id]
                item['latent'] = trained_latent.squeeze()
            except KeyError:
                # print(f"Warning: Latent for {fruit_id} not found in __getitem__")
                item['latent'] = torch.zeros(32) # Make assumption about size or crash
                pass

        # ------------------------------------------------------------------
        # Caliper weak supervision: mark intermediate frames as weak,
        # set proxy_volume_ml, scale target latent by (V_proxy/V_laser)^(1/3),
        # and disable SDF supervision (no shape GT for intermediate dates).
        # ------------------------------------------------------------------
        _is_weak_flag = False
        _proxy_vol = -1.0
        if self.caliper_weak_supervision:
            date_str = self._parse_date_from_frame_id(frame_id)
            last_d = self.fruit_last_date.get(fruit_id)
            is_intermediate = (date_str is not None and last_d is not None and date_str != last_d)
            if is_intermediate:
                v_proxy = self.vol_map.get((fruit_id, date_str), 0.0)
                # Prefer laser GT volume; fallback to caliper-derived V_proxy_last
                v_ref = self.last_laser_vol.get(fruit_id, self.fruit_last_vol.get(fruit_id, 0.0))
                if v_proxy > 0 and v_ref > 0:
                    _is_weak_flag = True
                    _proxy_vol = float(v_proxy)
                    scale = (v_proxy / v_ref) ** (1.0 / 3.0)
                    if 'latent' in item:
                        item['latent'] = item['latent'] * float(scale)
                    # Disable SDF supervision for weak samples (no shape GT).
                    # Only set None when SDF keys were already created by the sdf_loss block;
                    # otherwise (3D_loss=false) leave keys absent so default collate doesn't see
                    # mixed presence/absence between samples.
                    if self.sdf_loss:
                        item['target_sdf'] = None
                        item['target_sdf_xyz'] = None
                        item['target_sdf_weights'] = None
        item['is_weak'] = torch.tensor(_is_weak_flag, dtype=torch.bool)
        item['proxy_volume_ml'] = torch.tensor(_proxy_vol, dtype=torch.float32)

        # Load Ground Truth Size (BBox Extents) for Size Stream Supervision
        bbox_path = layout.bounding_box(self.data_source, fruit_id)
        try:
            if os.path.exists(bbox_path):
                data = np.load(bbox_path)
                if "bb" in data:
                     bb = data["bb"]
                elif "arr_0" in data:
                     bb = data["arr_0"]
                else:
                     bb = np.zeros((2, 3)) # Fallback
                
                bb = np.asarray(bb, dtype=np.float32)
                if bb.size > 0 and bb.ndim == 2 and bb.shape[1] == 3:
                    mins = bb.min(axis=0)
                    maxs = bb.max(axis=0)
                    ext = maxs - mins
                    # Normalize by 2500mm
                    ext_norm = ext / 2500.0
                    item['size'] = torch.from_numpy(ext_norm).float()
                else:
                     item['size'] = torch.zeros(3)
            else:
                 item['size'] = torch.zeros(3)
        except Exception:
             item['size'] = torch.zeros(3)

        # Vanilla CoRe++ baseline: render the just-loaded colored partial PCD
        # to a 2D RGB-D image (replacing the zero placeholders set in the
        # use_orbbec_pcd / use_phenobox_pcd branches). The image-based encoders
        # (DoubleEncoder, EncoderBigPooled) then consume the SAME underlying
        # partial observation as the point-cloud encoders, with only the
        # encoder architecture as the comparison variable.
        if self.use_pcd_render_as_rgbd and (self.use_orbbec_pcd or self.use_phenobox_pcd):
            fpath = self.files[idx]
            try:
                cached = self._rendered_rgbd_cache.get(fpath) if self.cache_rendered_rgbd else None
                if cached is not None:
                    rgb_u8, depth_u8, mask_u8 = cached
                    # uint8 → float32 [0, 1] (depth/rgb), keep mask uint8.
                    item['rgb'] = torch.from_numpy(rgb_u8.transpose(2, 0, 1).astype(np.float32) / 255.0).contiguous()
                    item['depth'] = torch.from_numpy(depth_u8[None, ...].astype(np.float32) / 255.0).contiguous()
                    item['mask'] = torch.from_numpy(mask_u8[None, ...]).contiguous().to(torch.uint8)
                elif self.use_raw_camera_rgbd:
                    # Cache miss → load aligned camera-frame RGB-D on the fly.
                    from corepp.baselines.corepp_vanilla.raw_rgbd import load_aligned_rgbd
                    fid = layout.fruit_id_from_path(fpath, self.data_source)
                    frame_id_raw = os.path.splitext(os.path.basename(fpath))[0].replace(self.orbbec_pcd_suffix, '')
                    out = load_aligned_rgbd(
                        fid, frame_id_raw,
                        size=self.render_size,
                        depth_min_mm=self.raw_rgbd_depth_min_mm,
                        depth_max_mm=self.raw_rgbd_depth_max_mm,
                        bbox_margin=self.raw_rgbd_bbox_margin,
                    )
                    if out is None:
                        raise FileNotFoundError(f"raw RGB-D missing for {fid}/{frame_id_raw}")
                    rgb_arr, depth_arr, mask_arr = out
                    item['rgb'] = torch.from_numpy(rgb_arr.transpose(2, 0, 1)).contiguous().float()
                    item['depth'] = torch.from_numpy(depth_arr[None, ...]).contiguous().float()
                    item['mask'] = torch.from_numpy(mask_arr[None, ...]).contiguous().to(torch.uint8)
                else:
                    from corepp.baselines.corepp_vanilla.render_pcd import render_pcd_to_rgbd
                    _xyz = np.asarray(partial_pcd.points, dtype=np.float32)
                    _cols = np.asarray(partial_pcd.colors, dtype=np.float32)
                    if _cols.size == 0 or _cols.shape[0] != _xyz.shape[0]:
                        _cols = np.full_like(_xyz, 0.5, dtype=np.float32)
                    rgb_arr, depth_arr, mask_arr = render_pcd_to_rgbd(
                        _xyz, _cols,
                        size=self.render_size,
                        splat_radius=self.render_splat_radius,
                        fixed_scale=self.render_fixed_scale,
                        fixed_xy_half_extent=self.render_fixed_xy_half_extent,
                        fixed_z_half_extent=self.render_fixed_z_half_extent,
                        target_points=self.render_target_points,
                        target_jitter_mm=self.render_target_jitter_mm,
                        rng_seed=hash(self.files[idx]) & 0xFFFFFFFF,
                    )
                    item['rgb'] = torch.from_numpy(rgb_arr.transpose(2, 0, 1)).contiguous().float()
                    item['depth'] = torch.from_numpy(depth_arr[None, ...]).contiguous().float()
                    item['mask'] = torch.from_numpy(mask_arr[None, ...]).contiguous().to(torch.uint8)
                item['padding_mask'] = torch.ones((1, self.render_size, self.render_size), dtype=torch.uint8)
            except Exception as _e:
                print(f"[render_pcd] failed for {item.get('fruit_id', '?')}/{item.get('frame_id', '?')}: {_e}")

        # Detach + drop requires_grad on all tensors so DataLoader workers (num_workers>0)
        # can pickle the item without "Cowardly refusing to serialize non-leaf tensor" errors.
        for _k, _v in list(item.items()):
            if isinstance(_v, torch.Tensor):
                if _v.requires_grad or _v.grad_fn is not None:
                    item[_k] = _v.detach().requires_grad_(False)
        return item


if __name__ == '__main__':

    # tfs = [Pad(size=128), ToTensor()]
    tfs = [Pad(size=500)]
    tf = transforms.Compose(tfs)
    cl_dataset = MaskedCameraLaserData(data_source="/home/federico/Datasets/shape_completion/potato/Potato/", 
                                tf=tf, supervised_3d=True,
                                pretrain='/home/federico/Datasets/shape_completion/potato/deepsdf_potato_weights/', 
                                pad_size=250,
                                sdf_loss=False,
                                grid_density=50,
                                split='train',
                                species='Potato')

    dataset = DataLoader(cl_dataset, batch_size=1, shuffle=False)

    c = 0
    for item in iter(dataset):
        # import ipdb;ipdb.set_trace()
        ids = item['fruit_id']
        c+=1
        print(c*32)
