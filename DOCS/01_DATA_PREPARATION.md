# Step 1: Data Preparation

Complete guide for preparing data from GT meshes to training-ready datasets.

---

## Overview

This step converts raw 3D meshes into training data for DeepSDF and encoders.

**Input**: GT meshes from Structure-from-Motion (SfM)  
**Output**: Postprocessed + centered/scaled PCDs, augmented data, DeepSDF samples, synthetic partial PCDs

---

## Pipeline Flow

```
GT Mesh (SfM)
    ↓
Point Cloud (data/kmelon)
    ↓
Postprocessed + Centered/Scaled PCD (data/kmelon_centered)
    ↓
├─→ Augmented Data (data/kmelon_augment)
├─→ DeepSDF Training Data
└─→ Synthetic Partial PCDs (phenobox/)
```

---

## Step 1.1: GT Mesh to Point Cloud

Convert SfM meshes to point clouds.

### Input
- **Location**: `data/3DKmelonDemo/2_sfm/1_mesh/`
- **Format**: Mesh files (PLY/OBJ)

### Script
```bash
# cd data_preparation
python3 data_preparation/pcd_from_sfm.py \
    --src ./data/3DKmelonDemo/2_sfm/1_mesh \
    --dst ./data/kmelon \
    --skip_existing \
    --quiet_skip
```

### Output
- **Location**: `data/kmelon/`
- **Format**: Point cloud files (`.ply`)

### What it does
- Loads mesh files
- Samples points uniformly from mesh surface
- Preserves color information if available
- Saves as point cloud files

---

## Step 1.2: Postprocess + Center/Scale Point Clouds

Postprocess point clouds (remove black support/stripe + fill + reorient) and output a **recentered + rescaled** version for SDF preparation.

### Input
- **Location**: `data/kmelon/`

### Script
```bash
# From corepp/ (recommended)
python3 data_preparation/postprocess_kmelon_pointclouds.py \
    --src_root ./data/kmelon \
    --dst_root ./data/kmelon_centered \
    --s_min 1 --s_max 70 \
    --center_method mean \
    --scale_default 0.18 \
    --scale_alt 0.22 --scale_alt_min 71 --scale_alt_max 120 \
    --dark_mode fixed --dark_thr 0.08 \
    --poisson_depth 7 --density_keep_percentile 5 --n_points_out 20000 \
    --skip_no_color
```

### Output
- **Location**:
  - `data/kmelon_centered/S*/laser/fruit.ply` (for each sample; postprocessed + centered + scaled; use this for downstream steps)
  - `data/kmelon_centered/S*/laser/fruit_aligned.ply` (cleaned + filled + PCA-aligned, before final recenter/scale)
  - `data/kmelon_centered/S*/tf/centering.json` (recentering + scale metadata)
  - `data/kmelon_centered/S*/tf/bounding_box.npz` (AABB corners of the centered+scaled PLY)

### What it does
- Removes dark (black) support/stripe points (optional, thresholded by luminance)
- Optionally denoises + keeps the largest cluster
- Fills missing regions via Poisson surface reconstruction
- Reorients so the fruit’s longest PCA axis becomes vertical (+Z)
- Recenters to origin and rescales for SDF preparation:
  - default **scale = 0.18**
  - samples **S71–S120 scale = 0.22**

### Why centering?
- DeepSDF expects shapes centered at origin
- Enables consistent global bounding box across all samples
- Improves network training stability

---

## Step 1.3: Data Augmentation

Augment data to increase training samples.

### Input
- **Location**: `data/kmelon_centered/`

### Script
```bash
# cd data_preparation
python3 data_preparation/augment.py \
    --src ./data/kmelon_centered \
    --dst ./data/kmelon_augment \
    --json_config_filename data_preparation/augment.json \
    --skip_existing \
    --quiet_skip
```

### Output
- **Location**: `data/kmelon_augment/`
- **Format**: Augmented versions of each sample (contains both `.ply` and `.npz` samples)

### Augmentation Types
1. **Rotation**: Random rotations around vertical axis
2. **Scaling**: Slight uniform scaling (0.9x - 1.1x)
3. **Jittering**: Small random noise to point positions

### Parameters
Parameters are defined in `augment.json`:
- `no_of_augmentations`: Number of augmented versions per sample (default: 10)
- `max_rotation_angle_degree`: Max rotation in degrees
- `min_scalefactor` / `max_scalefactor`: Scaling range

---

## Step 1.4: Prepare DeepSDF Training Data

Convert point clouds to SDF samples for DeepSDF training.

### Script
```bash
# cd data_preparation
python3 data_preparation/prepare_deepsdf_training_data.py \
    --src ./data/kmelon_centered \
    --no_of_samples 250000 \
    --skip_existing \
    --quiet_skip
```

### What it does
1. For each point cloud:
   - Samples points near surface (on-surface samples)
   - Samples points in free space (off-surface samples)
   - Computes signed distance for each sample
2. Saves as `.npz` files for DeepSDF training (saved alongside the source ply files)

---

## (Recommended) Step 1.4b: Add Uniform “Guaranteed Outside” Samples

The default SDF sampling is dominated by near-surface samples. This can under-constrain the decoder in the **far field**, which may lead to **sign flips near the marching-cubes grid boundary** (often seen as an unwanted “outer shell” around the shape).

This optional step appends extra **uniform samples that are guaranteed outside** the object to each `laser/samples.npz`.

### Script

```bash
# Adds +10% outside samples (relative to existing pos count), with backups
python3 data_preparation/add_uniform_outside_samples.py \
    --src ./data/kmelon_centered \
    --outside_ratio 0.10 \
    --cube_half_extent 0.5 \
    --outside_radius 0.30 \
    --sdf_value 0.1 \
    --backup
```

### Output / behavior
- Modifies in-place: `data/kmelon_centered/S*/laser/samples.npz`
- If `--backup` is set: creates `samples.npz.bak` alongside each file (one-time)
- Adds metadata keys inside the `.npz` (e.g. `outside_added`, `outside_added_count`, etc.)

### Undo / safety

```bash
# Restore all samples.npz from their backups
python3 data_preparation/add_uniform_outside_samples.py \
    --src ./data/kmelon_centered \
    --restore
```

### When you should use this
- Recommended if you observe **outer-shell artifacts**, unstable far-field SDF signs, or reconstructions that “close” incorrectly at the grid boundary.
- If your reconstructions are already stable, you can skip it.

### Output Structure
```
data/kmelon_centered/
├── S1/
│   ├── laser/
│   │   ├── fruit.ply
│   │   └── samples.npz
├── S2/
│   └── ...
```

### Run for Both Datasets
```bash
# For centered data (creates samples.npz inside each subject folder)
python3 prepare_deepsdf_training_data.py \
    --src ../data/kmelon_centered

# For augmented data
# augment.py already generates samples.npz, so this step is skipped for augmented data.
```

---

## Step 1.5: Generate Synthetic Partial Point Clouds

Create realistic partial views for pretraining encoders.

### Purpose
Simulate real sensor data (like greenhouse cameras) from complete point clouds for encoder pretraining.

### Script
```bash
cd data_preparation
python3 generate_partial_pointclouds.py \
    --dataset_root ../data/kmelon_centered \
    --all \
    --num 15 \
    --seed 42
```

### What it does
1. **Viewpoint sampling**: Simulates camera positions
2. **Occlusion simulation**: Adds realistic occlusions (leaves, stems)
3. **Point distribution**: Uniformizes point density
4. **Quality control**: Ensures target point count range

### Output Structure
```
data/kmelon_centered/
└── S1/
    ├── laser/
    │   └── fruit.ply          # Complete GT
    └── phenobox/
        └── pointclouds/
            ├── partial_00.ply  # Synthetic partial view 1
            ├── partial_01.ply  # Synthetic partial view 2
            └── ...
            └── partial_14.ply  # Synthetic partial view 15
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num` | 15 | Number of partial views per sample |
| `--min_ratio` | 0.1 | Min partial size (% of full cloud) |
| `--max_ratio` | 0.4 | Max partial size (% of full cloud) |
| `--seed` | 42 | Random seed for reproducibility |
| `--occluder_scale` | 3.0 | Scale factor for occlusion objects |

### Occlusion Types
- **Leaves**: Most common, flat circular occluders
- **Thin stems**: Narrow cylindrical stems
- **Thick stems**: Thicker cylindrical branches

### Advanced Usage

**Process specific samples:**
```bash
python3 generate_partial_pointclouds.py \
    --dataset_root ../data/kmelon_centered \
    --samples S1,S2,S3 \
    --num 15
```

**Less aggressive occlusion:**
```bash
python3 generate_partial_pointclouds.py \
    --dataset_root ../data/kmelon_centered \
    --all \
    --num 15 \
    --occlusion_prob 0.3 \
    --occluders_max 1
```

**Higher density partials:**
```bash
python3 generate_partial_pointclouds.py \
    --dataset_root ../data/kmelon_centered \
    --all \
    --num 10 \
    --upsample_factor 1.5
```

---


---

## Step 1.6: Preprocess Orbbec Input Data

Prepare real-world sensor data for finetuning or testing.

### Purpose
Orbbec depth data often contains noise and is not aligned with the canonical coordinate system. This step cleans and centers the partial point clouds.

### Script
```bash
python3 preprocess_greenhouse_pcd.py
```

### What it does
1.  **Iterates** through all samples in `data/kmelon_centered/*/orbbec/pointclouds/`
2.  **Denoises**: Removes statistical outliers.
3.  **Centers**: Aligns the partial view to the fruit center (approximate).
4.  **Saves**: Creates new files with suffix `_preprocessed.ply`.

### Output
```
data/kmelon_centered/S1/orbbec/pointclouds/
├── 20250501_masked_color_points.ply             # Raw
└── 20250501_masked_color_points_preprocessed.ply # Cleaned (Use this!)
```

**Note**: To use these files in training, ensure your config has `"orbbec_pcd_suffix": "_preprocessed"`.

---

## Complete Pipeline Script


Run all steps in sequence:

```bash
#!/bin/bash
# Complete data preparation pipeline

cd data_preparation

# Step 1: Mesh to PCD
echo "Step 1: Converting meshes to point clouds..."
python3 pcd_from_sfm.py \
    --src ../data/3DKmelonDemo/2_sfm/1_mesh \
    --dst ../data/kmelon

# Step 2: Center PCDs
echo "Step 2: Postprocess + center/scale point clouds..."
python3 postprocess_kmelon_pointclouds.py \
    --src_root ../data/kmelon \
    --dst_root ../data/kmelon_centered \
    --scale_default 0.18 \
    --scale_alt 0.22 --scale_alt_min 71 --scale_alt_max 120

# Step 3: Augment data
echo "Step 3: Augmenting data..."
python3 augment.py \
    --src ../data/kmelon_centered \
    --dst ../data/kmelon_augment \
    --json_config_filename augment.json

# Step 4: Prepare DeepSDF data
echo "Step 4: Preparing DeepSDF training data..."
# For centered data
python3 prepare_deepsdf_training_data.py \
    --src ../data/kmelon_centered \
    --no_of_samples 250000

# (Recommended) Step 4b: Add uniform outside samples (helps prevent outer-shell artifacts)
python3 add_uniform_outside_samples.py \
    --src ../data/kmelon_centered \
    --outside_ratio 0.10 \
    --backup

# Augmented data is already handled by augment.py

# Step 5: Generate synthetic partial PCDs
echo "Step 5: Generating synthetic partial point clouds..."
python3 generate_partial_pointclouds.py \
    --dataset_root ../data/kmelon_centered \
    --all \
    --num 15 \
    --seed 42

echo "Data preparation complete!"
```

---

## Verification

### Check Output Files

```bash
# Check centered data
ls data/kmelon_centered/*/laser/fruit.ply

# Check SDF samples
ls data/kmelon_centered/*/SdfSamples/*.npz

# Check synthetic partials
ls data/kmelon_centered/*/phenobox/pointclouds/partial_*.ply

# Count files
find data/kmelon_centered -name "fruit.ply" | wc -l
find data/kmelon_centered -name "*.npz" | wc -l
find data/kmelon_centered -name "partial_*.ply" | wc -l
```

### Expected Output
- Centered PCDs: N samples (one per fruit)
- SDF samples: N `.npz` files
- Partial PCDs: N × 15 files (15 views per fruit)

---

## Data Format Details

### Point Cloud Files (`.ply`)
- **Vertices**: 3D coordinates (X, Y, Z)
- **Colors**: RGB values (0-255) if available
- **Format**: Binary PLY (compressed)

### SDF Sample Files (`.npz`)
- **Keys**:
  - `pos`: Sample positions (N, 3)
  - `neg`: Sample positions for negative SDF
  - `sdf`: Signed distance values (N,)

### Coordinate System
- **Origin**: `(0, 0, 0)` (center of fruit)
- **Units**: Meters
- **Axes**: Right-handed coordinate system

---

## Postprocess details (same as Step 1.2)

If your reconstructed `fruit.ply` includes black points from a support/turntable and/or a black stripe on the fruit surface, you can postprocess the point clouds by:
- removing dark points based on RGB luminance
- filling missing regions via Poisson reconstruction (smooth completion)
- reorienting so the fruit’s **longest axis becomes vertical (+Z)** via PCA

### Script

```bash
python3 data_preparation/postprocess_kmelon_pointclouds.py \
    --src_root ./data/kmelon \
    --dst_root ./data/kmelon_centered \
    --s_min 1 --s_max 70 \
    --dark_mode fixed --dark_thr 0.08 \
    --poisson_depth 7 \
    --density_keep_percentile 5 \
    --n_points_out 20000 \
    --skip_no_color
```

### Output
- `data/kmelon_centered/S*/laser/fruit.ply`: postprocessed + centered + scaled output (default filename for downstream compatibility)
- `data/kmelon_centered/S*/laser/fruit_aligned.ply`: cleaned + filled + PCA-aligned point cloud (before final recenter/scale)
- `data/kmelon_centered/S*/laser/postprocess_transform.json`: applied transforms and parameters (including scale)
- `data/kmelon_centered/postprocess_summary.json`: per-sample summary (removed points, etc.)
- `data/kmelon_centered/S*/tf/centering.json`: recenter + scale metadata
- `data/kmelon_centered/S*/tf/bounding_box.npz`: AABB corners for downstream cropping

### Notes / knobs
- `--dark_thr`: increase if black support is not fully removed; decrease if too much fruit is removed.
- `--skip_existing`: skip samples if the final output already exists (useful for resuming).
- `--fill_z_keep_fraction`: set to e.g. `0.95` to prevent Poisson fill from “hallucinating” the bottom ~5% (support region).
- `--density_keep_percentile`: set to `0` to disable density pruning (can help avoid reintroducing holes).
- `--write_intermediate`: writes debug PLYs (`fruit_dark_removed.ply`, `fruit_clustered.ply`, `fruit_clustered_aligned.ply`, `fruit_filled_all.ply`) per sample.
- Scale factors: default `--scale_default 0.18`, and `--scale_alt 0.22 --scale_alt_min 71 --scale_alt_max 120` for S71–S120.

---

## Troubleshooting

### Issue: "No mesh files found"
**Solution**: Check input directory path and file extensions

### Issue: "Center computation failed"
**Solution**: Ensure point cloud has sufficient points (>100)

### Issue: "SDF sampling too slow"
**Solution**: Reduce `--num_samples` parameter

### Issue: "Partial PCDs too sparse"
**Solution**: Increase `--upsample_factor` or adjust `--min_ratio`

---

## Next Step

Once data preparation is complete, proceed to:
→ **[Step 2: DeepSDF Training](02_DEEPSDF_TRAINING.md)**

---

## Related Scripts

- `data_preparation/pcd_from_sfm.py` - Mesh to PCD conversion
- `data_preparation/postprocess_kmelon_pointclouds.py` - Postprocess + center/scale script
- `data_preparation/augment.py` - Data augmentation
- `data_preparation/prepare_deepsdf_training_data.py` - SDF sampling
- `data_preparation/add_uniform_outside_samples.py` - Add guaranteed-outside samples to `samples.npz` (recommended for stable far-field SDF)
- `data_preparation/generate_partial_pointclouds.py` - Partial PCD generation

---

**Last Updated**: October 31, 2025

