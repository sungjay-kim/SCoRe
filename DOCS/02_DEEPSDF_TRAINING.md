# Step 2: DeepSDF Training

Train DeepSDF decoder to learn shape representations with different latent dimensions.

---

## Overview

DeepSDF learns to map latent codes to signed distance functions (SDFs) representing 3D shapes.

**Input**: SDF samples from Step 1  
**Output**: Trained decoder + latent codes for each shape

---

## Quick Start

### Train Single Latent Size

```bash
python3 train_deep_sdf.py -e deepsdf/experiments/kmelon_latent32
```

### Train All Latent Sizes

```bash
for latent in 16 32 64 128; do
    python3 train_deep_sdf.py -e deepsdf/experiments/kmelon_latent${latent}
done
```

### Size-Aware Weight Variant (32D)
Uses an auxiliary size head supervised by fruit weight (log-transformed target).
```bash
python3 train_deep_sdf.py -e deepsdf/experiments/kmelon_latent32_sizeaware_weight
```
Spec highlights:
- Latent 32D, 601 epochs, snapshots every 10 (+ extra at 0, 500, 1500)
- LR schedules (step): 0.0005 halved every 100 epochs; second schedule entry 0.001 halved every 100
- Batch: ScenesPerBatch=64, SamplesPerScene=16384, DataLoaderThreads=16
- Decoder: 8×512 MLP, latent injected at layer 4, weight norm on all layers, dropout 0.2 on all layers, no tanh, no xyz_in_all
- Regularization: CodeRegularization lambda=1e-4, CodeBound=1.0, ClampingDistance=0.1
- Size head: enabled, lambda=0.2, hidden dims [128,128], L2 loss
- Size target: `data/3DKmelonDemo/ground_truth.csv`, columns `label`/`weight_g_inctack`, log transform with epsilon 1e-12

---

## Latent Size Selection

Different latent dimensions offer different trade-offs:

| Latent Size | Pros | Cons | Use Case |
|-------------|------|------|----------|
| **16D** | Fast training, less memory | Limited expressiveness | Simple shapes, quick experiments |
| **32D** | Balanced (recommended) | - | General purpose |
| **64D** | More expressive | Slower training | Complex shapes |
| **128D** | Maximum detail | High memory, overfitting risk | Very complex shapes |

**Recommendation**: Start with **32D**, then experiment with others.

---

## Experiment Setup

### Directory Structure

```
deepsdf/experiments/
├── kmelon_latent16/
│   ├── specs.json              # Configuration
│   ├── ModelParameters/        # Saved checkpoints
│   ├── LatentCodes/           # Learned latent codes
│   ├── OptimizerParameters/   # Optimizer states
│   └── Logs.pth               # Training logs
├── kmelon_latent32/
├── kmelon_latent64/
└── kmelon_latent128/
```

### Configuration Files

Each experiment has a `specs.json` with key parameters:

```json
{
    "CodeLength": 32,                    // Latent dimension
    "CodeBound": 1.0,                    // L2 norm bound
    "CodeRegularizationLambda": 0.0001,  // Regularization weight
    "NumEpochs": 301,                    // Training epochs
    "SamplesPerScene": 16384,            // Points per shape
    "ScenesPerBatch": 64,                // Batch size
    "ClampingDistance": 0.1,             // Max SDF distance
    "LearningRateSchedule": [...]        // LR schedule
}
```

---

## Parameter Scaling

When changing latent size, **CodeBound** must be scaled appropriately:

### Formula
```
CodeBound = baseline × √(latent_size / 32)
```

### Values for Each Latent Size

| Latent Size | CodeBound | Calculation |
|-------------|-----------|-------------|
| 16D | 0.71 | `1.0 × √(16/32)` |
| 32D | 1.0 | `1.0 × √(32/32)` (baseline) |
| 64D | 1.41 | `1.0 × √(64/32)` |
| 128D | 2.0 | `1.0 × √(128/32)` |

**Why?** Higher dimensional spaces naturally have larger L2 norms. Scaling prevents over-constraining smaller latent spaces.

---

## Training Process

### Step 2.1: Train DeepSDF Decoder

```bash
python3 train_deep_sdf.py -e deepsdf/experiments/kmelon_latent32
```

**What it does:**
1. Loads SDF samples from `data/kmelon_centered/`
2. Initializes decoder network
3. Learns latent codes for each shape
4. Optimizes decoder to predict SDF values
5. Saves checkpoints every 10 epochs

**Training time:** ~2-4 hours per experiment (depends on GPU)

### Step 2.2: Reconstruct Shapes

After training, reconstruct 3D meshes from learned latent codes:

```bash
bash run_scripts_reconstruct.sh
```

This script runs reconstruction for all experiments and latent sizes.

**What it does:**
1. Loads trained decoder
2. For each shape's latent code:
   - Samples SDF values on a 3D grid
   - Extracts isosurface at SDF=0 (marching cubes)
   - Saves as mesh file

**Output:** Reconstructed meshes in `deepsdf/experiments/kmelon_latent*/Reconstructions/`

---

## Note: Mesh extraction centering bug (fixed)

If you observe a small, consistent translation bias in reconstructed meshes (e.g. \(\approx\) **2–3mm** in X/Y when using `N=192`), this can be caused by a **grid indexing bug** in the mesh extraction code path (`deep_sdf.mesh.create_mesh`).

### Symptom
- Reconstructed meshes have a near-constant centroid offset (often \(\approx -0.5 \cdot\) voxel size) even when GT point clouds (`laser/fruit.ply`) are centered.

### Root cause
- `create_mesh()` previously computed grid indices using **float division** (`/`) instead of integer floor division, which biases the sampling grid and produces a systematic translation.

### Fix
- Patched `corepp/deepsdf/deep_sdf/mesh.py` to:
  - compute indices with integer floor division
  - map indices to coordinates using consistent `(x,y,z)` origins

After this fix, `create_mesh()` no longer introduces the half-voxel translation bias.

---

## Note: Ghost outer-shell from mis-oriented surface normals (fixed 2026-04-12)

If you observe **ghost zero-crossings** or a faint **outer shell** ~1–3 cm from the true surface in reconstructed meshes — often **connected to the main body** as a bump or double layer, and mostly visible on S-series samples — the cause is **inward-flipped surface normals** in the source `laser/fruit.ply`, which propagate into mislabeled positive SDF samples during training data generation.

### Symptom
- Val precision stalls at ~45–55% while recall stays ~90% (low F1 ~60%).
- Reconstructed meshes have **two connected components**, or a single surface with a spurious outer bump.
- Problem affects S samples more than P, because S PLYs on this dataset have **no stored normals** while P PLYs do.
- Typically progresses over many epochs without improvement — ghost shell gets baked into the decoder's far-field SDF.

### Root cause
1. Historical data-preparation pipeline stripped normals from `S*/laser/fruit.ply` at some point (the intact originals are kept as `S*/laser/fruit_bak.ply`). P samples were unaffected because their processing pipeline preserved normals.
2. [`prepare_deepsdf_training_data.py`](../data_preparation/deepsdf/prepare_deepsdf_training_data.py) used to call `estimate_normals()` **without orienting** them, so on PLYs with no stored normals the estimator produced ~3% inward-flipped normals.
3. `generate_tsdf_samples()` uses each point's outward normal to place positive (outside-surface) SDF samples. Flipped normals move those samples **inside** the surface, but they are still labeled `sdf > 0` — the decoder learns a second positive region inside the mesh → ghost zero-crossing → outer shell.
4. Secondary factor: [`augment.py`](../data_preparation/deepsdf/augment.py) had its own normal-orientation logic (`orient_normals_towards_camera_location` + flip), so augmented samples were partially protected. This masked the severity of the bug during earlier S-only training (v1) but surfaced as ghost shells once S+P mixed training (v2) increased decoder capacity demands.

### Fix
- **`prepare_deepsdf_training_data.py`** and **`augment.py`** now always force a robust outward-oriented normal pipeline:
  1. `estimate_normals(radius=0.01, max_nn=30)` + `normalize_normals()`
  2. `orient_normals_consistent_tangent_plane(k=30)` for topology-propagated consistency
  3. Global sign check: if `mean(normal · (p - centroid)) < 0`, flip all normals
- This forces outward normals even when the source PLY has correct *stored* normals (some samples had stored normals that were themselves inconsistent, e.g. P112).

### Verification
After the fix, regenerate all `samples.npz` (centered + augmented) and verify:
```python
# For each sample: fraction of pos samples lying inside the surface (w.r.t. nearest
# point's outward normal) should be ~0% for spherical fruits, <6% for highly concave.
# Before fix: S samples = 3.2% inward, causing outer-shell artifacts.
# After fix:  S samples = 0.0% inward, P samples = 0-6% (concave regions only).
```

### Training impact
Scratch retraining after the normal fix showed an immediate jump:
- **Before fix**: val precision stuck at ~47% even at epoch 150.
- **After fix**: val precision = **77.9% at epoch 10** (≈ 30%p improvement in 1/15th the training time).

### How to detect this in a new dataset
- Load any `{sample}/laser/fruit.ply` and check `len(pcd.normals)`. If 0, re-estimating without orientation is unsafe.
- Load `samples.npz`, take `pos[:, :3]`, and for each point check whether it lies inside or outside the surface by dot product with the nearest surface point's **stored** normal. If >1% fall inside, the normals were already mis-oriented when SDF samples were computed.

### Related files
- [`corepp/data_preparation/deepsdf/prepare_deepsdf_training_data.py`](../data_preparation/deepsdf/prepare_deepsdf_training_data.py)
- [`corepp/data_preparation/deepsdf/augment.py`](../data_preparation/deepsdf/augment.py)

### Step 2.3: Compute Metrics

Evaluate reconstruction quality:

```bash
bash run_scripts_metrics.sh
```

**Metrics computed:**
- **Chamfer Distance**: Average point-to-surface distance
- **Hausdorff Distance**: Maximum distance error
- **Volume Error**: Difference in volume

**Output:** `reconstruction_metrics.csv` in each experiment directory

### Step 2.4: Select Best Weights

Based on metrics, identify the best checkpoint:

```bash
# Check metrics summary
cat deepsdf/experiments/kmelon_latent32/reconstruction_metrics.csv

# Look for lowest Chamfer distance
sort -t',' -k3 -n deepsdf/experiments/kmelon_latent32/reconstruction_metrics.csv | head -5
```

Typically, the **final checkpoint (epoch 301)** or **last few epochs** work best.

---

## Reconstruction for Encoder Training

After selecting the best weights, prepare reconstructions for encoder training:

```bash
python3 reconstruct_deep_sdf.py \
    --experiment deepsdf/experiments/kmelon_latent32 \
    --checkpoint 301 \
    --resolution 256 \
    --output_dir data/reconstructions/latent32
```

### Parameters
- `--experiment`: DeepSDF experiment directory
- `--checkpoint`: Epoch number to use
- `--resolution`: Grid resolution for marching cubes (higher = more detail)
- `--output_dir`: Where to save reconstructed meshes

---

## Preparing Data for Encoder Training

Organize reconstructed data for encoder training:

```bash
cd data_preparation

# Organize directory structure
python3 organize_data.py \
    --reconstruction_dir ../data/reconstructions/latent32 \
    --laser_dir ../data/kmelon_centered \
    --output_dir ../data/encoder_training/latent32

# Copy necessary files
python3 copy_file.py \
    --source ../data/kmelon_centered \
    --dest ../data/encoder_training/latent32
```

**Output structure:**
```
data/encoder_training/latent32/
├── S1/
│   ├── laser/
│   │   └── fruit.ply        # GT point cloud
│   ├── reconstruction/
│   │   └── mesh.ply         # DeepSDF reconstruction
│   └── phenobox/
│       └── pointclouds/     # Synthetic partial views
└── ...
```

---

## Monitoring Training

### TensorBoard (if enabled)
```bash
tensorboard --logdir deepsdf/experiments/ --port 6006
```

### Check Logs
```bash
# Training progress
tail -f deepsdf/experiments/kmelon_latent32/train.log

# Load training history
python3 -c "import torch; logs = torch.load('deepsdf/experiments/kmelon_latent32/Logs.pth'); print(logs.keys())"
```

### Key Metrics to Watch
- **Loss decreasing steadily**: Good training
- **Latent magnitude < CodeBound**: Codes not saturated
- **No NaN values**: Stable training

---

## Comparing Latent Sizes

After training all sizes, compare their performance:

```bash
# Compare reconstruction quality
for latent in 16 32 64 128; do
    echo "=== Latent ${latent} ==="
    tail -1 deepsdf/experiments/kmelon_latent${latent}/reconstruction_metrics.csv
done
```

**What to look for:**
- Lower Chamfer distance = better reconstruction
- Check if 128D significantly outperforms 32D (may not be worth the cost)
- Ensure 16D isn't underfitting

---

## Automation Scripts

### `run_scripts_reconstruct.sh`
```bash
#!/bin/bash
# Reconstruct meshes for all trained models

for latent in 16 32 64 128; do
    echo "Reconstructing latent ${latent}..."
    python3 reconstruct_deep_sdf.py \
        --experiment deepsdf/experiments/kmelon_latent${latent} \
        --checkpoint 301 \
        --resolution 256
done
```

### `run_scripts_metrics.sh`
```bash
#!/bin/bash
# Compute reconstruction metrics for all models

for latent in 16 32 64 128; do
    echo "Computing metrics for latent ${latent}..."
    python3 compute_reconstruction_metrics.py \
        --experiment deepsdf/experiments/kmelon_latent${latent} \
        --gt_dir data/kmelon_centered
done
```

### `check_reconstruction_progress.sh`
```bash
#!/bin/bash
# Check training progress for all experiments

for latent in 16 32 64 128; do
    echo "=== Latent ${latent} ==="
    tail -5 deepsdf/experiments/kmelon_latent${latent}/train.log
    echo ""
done
```

---

## Troubleshooting

### Issue: Training diverges (NaN loss)
**Solutions:**
1. Decrease learning rate in `specs.json`
2. Increase `CodeBound`
3. Decrease `CodeRegularizationLambda`

### Issue: Latent codes saturate at bound
**Symptom:** Mean latent magnitude ≈ CodeBound

**Solutions:**
1. Increase `CodeBound` (e.g., multiply by 1.5)
2. Indicates insufficient capacity - try larger latent size

### Issue: Poor reconstruction quality
**Solutions:**
1. Train longer (more epochs)
2. Increase `SamplesPerScene` (more points per shape)
3. Try larger latent size
4. Check if input data is properly centered

### Issue: Out of memory
**Solutions:**
1. Decrease `ScenesPerBatch` (smaller batch size)
2. Decrease `SamplesPerScene`
3. Use smaller latent size
4. Use smaller GPU memory with gradient checkpointing

---

## Expected Results

### Training Time
- **16D**: ~1.5 hours
- **32D**: ~2 hours
- **64D**: ~3 hours
- **128D**: ~4 hours

(On single NVIDIA RTX 3090)

### Memory Usage
- **16D**: ~4 GB
- **32D**: ~6 GB
- **64D**: ~8 GB
- **128D**: ~10 GB

### Typical Chamfer Distance (lower is better)
- **16D**: 0.003 - 0.005
- **32D**: 0.002 - 0.004
- **64D**: 0.001 - 0.003
- **128D**: 0.001 - 0.003

---

## Next Step

Once DeepSDF training is complete and you've selected the best weights, proceed to:
→ **[Step 3: Encoder Training](03_ENCODER_TRAINING.md)**

---

## Related Files

- `train_deep_sdf.py` - Main training script
- `reconstruct_deep_sdf.py` - Reconstruction script
- `compute_reconstruction_metrics.py` - Metrics computation
- `run_scripts_reconstruct.sh` - Batch reconstruction
- `run_scripts_metrics.sh` - Batch metrics
- `check_reconstruction_progress.sh` - Progress monitoring

---

**Last Updated**: October 31, 2025

