# Troubleshooting Guide

Common issues and solutions for the TCORE pipeline.

---

## Quick Diagnosis

**Identify your issue:**
- [Data Preparation Issues](#data-preparation-issues)
- [DeepSDF Training Issues](#deepsdf-training-issues)
- [Encoder Training Issues](#encoder-training-issues)
- [Testing/Evaluation Issues](#testing-evaluation-issues)
- [Visualization Issues](#visualization-issues)
- [System/Hardware Issues](#system-hardware-issues)

---

## Data Preparation Issues

### No mesh files found
**Symptom:** "No mesh files found in directory"

**Solutions:**
1. Check input directory path is correct
2. Verify file extensions (`.ply`, `.obj`, `.off`)
3. Check file permissions
4. Use absolute paths instead of relative

**Example:**
```bash
# Wrong
python3 pcd_from_sfm.py --input_dir ../data/meshes

# Right
python3 pcd_from_sfm.py --input_dir /home/user/tcore/corepp/data/3DKmelonDemo/2_sfm/1_mesh
```

### Center computation failed
**Symptom:** Error during centering process

**Solutions:**
1. Ensure point cloud has sufficient points (>100)
2. Check if input file is valid PLY format
3. Verify point cloud is not empty

**Check file:**
```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("file.ply")
print(f"Points: {len(pcd.points)}")
```

### SDF sampling too slow
**Symptom:** `prepare_deepsdf_training_data.py` takes too long

**Solutions:**
1. Reduce `--num_samples` (default: 250000)
2. Use fewer samples per shape
3. Process in parallel

**Example:**
```bash
# Reduce samples
python3 prepare_deepsdf_training_data.py \
    --data_dir ../data/kmelon_centered \
    --num_samples 100000
```

### Partial PCDs too sparse
**Symptom:** Generated partial point clouds have too few points

**Solutions:**
1. Increase `--upsample_factor`
2. Adjust `--min_ratio` and `--max_ratio`
3. Reduce occlusion probability

**Example:**
```bash
python3 generate_partial_pointclouds.py \
    --dataset_root ../data/kmelon_centered \
    --all \
    --num 15 \
    --upsample_factor 1.5 \
    --min_ratio 0.2 \
    --max_ratio 0.5
```

---

## DeepSDF Training Issues

### Training diverges (NaN loss)
**Symptom:** Loss becomes NaN or infinity

**Solutions:**
1. Decrease learning rate in `specs.json`
2. Increase `CodeBound`
3. Decrease `CodeRegularizationLambda`
4. Check if input data has NaN values

**Fix specs.json:**
```json
{
    "LearningRateSchedule": [
        {"Type": "Step", "Initial": 0.0005, ...}  // Reduced from 0.001
    ],
    "CodeBound": 1.5,  // Increased from 1.0
    "CodeRegularizationLambda": 0.00005  // Reduced from 0.0001
}
```

### Latent codes saturate at bound
**Symptom:** Mean latent magnitude ≈ `CodeBound`

**Solutions:**
1. Increase `CodeBound` (multiply by 1.5)
2. Try larger latent size
3. Check if data is too complex

**Check saturation:**
```python
import torch
logs = torch.load('deepsdf/experiments/kmelon_latent32/Logs.pth')
latent_mags = logs['latent_magnitude']
print(f"Mean: {latent_mags[-1].mean()}")
print(f"CodeBound: 1.0")  # From specs.json
```

### Poor reconstruction quality
**Symptom:** Reconstructed meshes don't match GT

**Solutions:**
1. Train longer (more epochs)
2. Increase `SamplesPerScene`
3. Try larger latent size
4. Check if input data is properly centered
5. Verify SDF samples are correct

**Check reconstruction:**
```bash
python3 compute_reconstruction_metrics.py \
    --experiment deepsdf/experiments/kmelon_latent32 \
    --gt_dir data/kmelon_centered
```

### Out of memory during DeepSDF training
**Symptom:** CUDA out of memory error

**Solutions:**
1. Decrease `ScenesPerBatch`
2. Decrease `SamplesPerScene`
3. Use smaller latent size
4. Enable gradient checkpointing

**Fix specs.json:**
```json
{
    "ScenesPerBatch": 32,  // Reduced from 64
    "SamplesPerScene": 8192  // Reduced from 16384
}
```

---

## Encoder Training Issues

### Pretrained encoder file not found
**Symptom:** "Pretrained encoder file not found"

**Solutions:**
1. Use absolute path in config
2. Verify file exists
3. Check filename (`.pt` extension)

**Fix config:**
```json
{
    "pretrained_encoder": "/home/user/tcore/corepp/experiments/kmelon_latent32/20251031_120000_pretrain/logs/checkpoints/minkowski_phenobox_best_model.pt"
}
```

### Poor finetuning performance
**Symptom:** Model doesn't improve during finetuning

**Solutions:**
1. Lower learning rate (try 0.00001)
2. Train longer (50+ epochs)
3. Check preprocessing was applied
4. Verify pretrained model quality
5. Ensure data paths are correct

**Check config:**
```json
{
    "lr": 0.0001,  // 10x lower than pretrain
    "use_orbbec_pcd": true,
    "orbbec_pcd_suffix": "_preprocessed"  // Important!
}
```

### Overfitting during finetuning
**Symptom:** Training loss low, validation loss high

**Solutions:**
1. Lower learning rate
2. Fewer epochs (20-30)
3. Early stopping
4. Data augmentation

**Monitor in TensorBoard:**
```bash
tensorboard --logdir experiments/kmelon_latent32/ --port 6006
```

### Training very slow
**Symptom:** Each epoch takes very long

**Solutions:**
1. Decrease batch size
2. Reduce validation frequency
3. Use fewer validation samples
4. Enable mixed precision training

**Fix config:**
```json
{
    "batch_size": 8,  // Reduced from 12
    "validation_frequency": 5  // Every 5 epochs instead of 1
}
```

### Encoder not learning (loss plateau)
**Symptom:** Loss doesn't decrease after initial drop

**Solutions:**
1. Check if decoder is properly frozen
2. Increase learning rate
3. Try different optimizer
4. Check if data has sufficient variation
5. Verify gradients are flowing

**Debug:**
```python
# In train.py, add:
for name, param in encoder.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm()}")
```

### Encoder predicts mean/average volume (Low Variance)
**Symptom:** Predicted shapes change, but their **volume/size** is always the same (the dataset mean).

**Cause:** The input point clouds are being normalized to `[-1, 1]`, destroying physical scale information. This is default for Minkowski but wrong for PCE/FoldNet.

**Solutions:**
1. Ensure your config does NOT use `normalize_pcd: true` if using PCE.
2. Explicitly set `"normalize_pcd": false` in config.
3. Check `verify_normalization_switch.py` to confirm input stats.

**Fix config:**
```json
{
    "encoder": "point_cloud",
    "normalize_pcd": false  // CRITICAL for volume estimation
}
```


### Data loading errors
**Symptom:** Error loading partial/target point clouds

**Solutions:**
1. Check file paths in dataset
2. Verify file format (PLY)
3. Ensure files are not corrupted
4. Check coordinate system alignment

**Verify data:**
```bash
# Check if files exist
find data/kmelon_centered/*/phenobox/pointclouds/ -name "*.ply" | wc -l

# Check file size
du -sh data/kmelon_centered/*/phenobox/pointclouds/partial_00.ply
```

---

## Testing Evaluation Issues

### Test crashes/OOM
**Symptom:** Out of memory during testing

**Solutions:**
1. Decrease batch size
2. Reduce marching cubes resolution
3. Test fewer samples at once
4. Use CPU inference

**Reduce resolution:**
```bash
python3 test.py \
    -e deepsdf/experiments/kmelon_latent32 \
    -c configs/kmelon/minkowski_finetune.json \
    --resolution 128  # Reduced from 256
```

### Poor test performance
**Symptom:** Reconstruction quality bad on test set

**Solutions:**
1. Check if encoder was properly trained
2. Verify decoder checkpoint is correct
3. Ensure test data is properly preprocessed
4. Try different checkpoint (not just best)
5. Check coordinate system alignment

**Debug:**
```bash
# Test with different checkpoint
python3 test.py \
    -e deepsdf/experiments/kmelon_latent32 \
    -c configs/kmelon/minkowski_finetune.json \
    --checkpoint_folder experiments/.../logs/checkpoints \
    --checkpoint_name minkowski_finetune_epoch_040.pt
```

### Size metrics don't match GT
**Symptom:** Large errors in length/width measurements

**Solutions:**
1. Apply bias correction (`--correction_csv`)
2. Verify GT CSV format
3. Check mesh filename parsing
4. Visualize PCA axes
5. Check units (mm vs cm)

**Apply correction:**
```bash
# Step 1: Quantify bias
python3 compare_hand_vs_mesh_size.py \
    --gt_csv data/3DKmelonDemo/size_data.csv \
    --gt_mesh_dir data/3DKmelonDemo \
    --output comparison_hand_vs_mesh.csv

# Step 2: Evaluate with correction
python3 evaluate_size_metrics.py \
    --mesh_dir experiments/.../inf/meshes \
    --gt_csv data/3DKmelonDemo/size_data.csv \
    --output size_evaluation.csv \
    --correction_csv comparison_hand_vs_mesh.csv
```

### All predictions look similar
**Symptom:** Low variance in predicted shapes/sizes

**Solutions:**
1. Check for overfitting (train vs val)
2. Ensure sufficient training data diversity
3. Try larger latent size
4. Verify decoder is properly loaded
5. Check if encoder output varies

**Debug:**
```python
# Check latent code variation
import numpy as np
latent_codes = []
for file in latent_code_files:
    code = np.load(file)
    latent_codes.append(code)
latent_codes = np.stack(latent_codes)
print(f"Mean: {latent_codes.mean(axis=0)}")
print(f"Std: {latent_codes.std(axis=0)}")
# Low std = no variation!
```

---

## Visualization Issues

### GIF generation fails
**Symptom:** Error creating GIF files

**Solutions:**
1. Check if mesh file exists and is valid
2. Install required packages: `pip install imageio pillow`
3. Reduce resolution or fps
4. Check disk space

**Test:**
```bash
python3 create_promo_gif.py \
    --mesh_file experiments/.../inf/meshes/S1_0_mesh.ply \
    --output test.gif \
    --resolution 400 \
    --fps 15
```

### Colored point clouds look wrong
**Symptom:** Colors don't represent errors correctly

**Solutions:**
1. Verify GT point cloud is properly loaded
2. Check `visualization_max_distance` setting
3. Try different colormap
4. Ensure point clouds are in same coordinate system

**Fix config:**
```json
{
    "save_colored_mesh": true,
    "visualization_colormap": "viridis",  // Try different
    "visualization_max_distance": 0.02  // Adjust scale
}
```

### SDF visualization shows no surface
**Symptom:** Blank or wrong visualization

**Solutions:**
1. Check if latent code was properly loaded
2. Verify decoder checkpoint is correct
3. Check if sample ID exists
4. Try different slice position

**Debug:**
```bash
python3 visualize_sdf_simple.py \
    --experiment deepsdf/experiments/kmelon_latent32 \
    --checkpoint 301 \
    --sample S1 \
    --slice_axis z \
    --slice_position 0.0  # Try -0.5, 0.0, 0.5
```

### Renderings are too dark/bright
**Symptom:** Poor lighting in visualizations

**Solutions:**
1. Adjust lighting parameters in script
2. Increase/decrease ambient light
3. Add more light sources
4. Adjust camera distance

---

## System Hardware Issues

### CUDA out of memory
**Symptom:** `RuntimeError: CUDA out of memory`

**Immediate fix:**
```bash
# Clear cache
python3 -c "import torch; torch.cuda.empty_cache()"
```

**Long-term solutions:**
1. Decrease batch size
2. Use gradient accumulation
3. Enable mixed precision training
4. Use smaller models
5. Close other GPU processes

**Check GPU usage:**
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### Training hangs/freezes
**Symptom:** No progress, no output

**Solutions:**
1. Check data loading (might be stuck)
2. Verify GPU is responding (`nvidia-smi`)
3. Check if deadlock in data loading
4. Reduce number of data loading workers

**Debug:**
```bash
# Kill and restart
pkill -9 python3

# Check GPU
nvidia-smi

# Restart with debugging
python3 -u train.py ... 2>&1 | tee train.log
```

### Disk space full
**Symptom:** "No space left on device"

**Solutions:**
1. Clean up old experiments
2. Remove intermediate files
3. Compress/archive results
4. Use different output directory

**Clean up:**
```bash
# Remove incomplete runs
bash clean_incomplete_runs.sh

# Remove old TensorBoard logs
find experiments -name "events.out.tfevents.*" -mtime +30 -delete

# Compress old experiments
tar -czf experiments_backup.tar.gz experiments/kmelon_latent32/2024*/
rm -rf experiments/kmelon_latent32/2024*/
```

### Multi-GPU issues
**Symptom:** Errors with multiple GPUs

**Solutions:**
1. Specify single GPU: `CUDA_VISIBLE_DEVICES=0`
2. Check if multi-GPU is properly implemented
3. Verify all GPUs are accessible

**Test:**
```python
import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

---

## Common Workflow Issues

### Wrong data being loaded
**Symptom:** Model trains on wrong dataset

**Solutions:**
1. Check `data_dir` in config
2. Verify `use_phenobox_pcd` / `use_orbbec_pcd` flags
3. Check `orbbec_pcd_suffix`

**Debug:**
```python
# In dataloader, add:
print(f"Loading data from: {self.data_dir}")
print(f"use_phenobox_pcd: {self.use_phenobox_pcd}")
print(f"use_orbbec_pcd: {self.use_orbbec_pcd}")
print(f"First file: {self.files[0]}")
```

### Configs not found
**Symptom:** "Config file not found"

**Solutions:**
1. Use absolute paths
2. Check working directory
3. Verify file exists

**Fix:**
```bash
# Use absolute path
python3 train.py \
    -e /home/user/tcore/corepp/deepsdf/experiments/kmelon_latent32 \
    -c /home/user/tcore/corepp/configs/kmelon/minkowski_phenobox.json
```

### Can't reproduce results
**Symptom:** Different results each run

**Solutions:**
1. Set random seeds
2. Disable data augmentation
3. Use same hardware
4. Check for non-deterministic operations

**Set seeds:**
```python
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Getting Help

### Before Asking for Help

1. **Check logs**: Look at console output and log files
2. **Verify setup**: Ensure dependencies are installed
3. **Try simple case**: Test with minimal example
4. **Search documentation**: Check relevant doc sections

### Provide This Information

When asking for help, include:
1. **Error message**: Full error traceback
2. **Command**: Exact command you ran
3. **Environment**: GPU, CUDA version, PyTorch version
4. **Config**: Relevant configuration parameters
5. **Logs**: Last 50 lines of log file

**Get system info:**
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## Prevention Tips

### Before Training

- ✅ Validate configs
- ✅ Check data exists and is correct format
- ✅ Test with small dataset first
- ✅ Monitor disk space and GPU memory
- ✅ Back up previous results

### During Training

- ✅ Monitor TensorBoard regularly
- ✅ Check logs for warnings
- ✅ Watch GPU utilization
- ✅ Save checkpoints frequently

### After Training

- ✅ Validate outputs before deleting inputs
- ✅ Back up important results
- ✅ Document successful configurations
- ✅ Clean up intermediate files

---

## Quick Fixes

### Reset Everything

```bash
# Nuclear option: start fresh
rm -rf experiments/kmelon_latent32/20251031_*/
rm -rf __pycache__
python3 -c "import torch; torch.cuda.empty_cache()"
```

### Check System Health

```bash
#!/bin/bash
echo "=== System Check ==="
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo ""
echo "Disk:"
df -h | grep -E "Filesystem|/home"
echo ""
echo "Python:"
python3 --version
echo ""
echo "PyTorch:"
python3 -c "import torch; print(torch.__version__)"
```

---

**Last Updated**: October 31, 2025

