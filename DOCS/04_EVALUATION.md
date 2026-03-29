# Step 4: Evaluation

Test trained models and evaluate reconstruction quality with multiple metrics.

---

## Overview

Evaluate encoder performance on:
1. **Shape metrics** - Chamfer distance, F1 score (when 3D GT available)
2. **Size metrics** - Length and width errors (for historical time-series data)

---

## Quick Start

### Test on Standard Data (with 3D GT)

```bash
python3 test.py \
    -e deepsdf/experiments/kmelon_latent32 \
    -c configs/kmelon/latent_32/pce_finetune_orbbec.json \
    --checkpoint_folder experiments/kmelon_latent32/TRAIN_RUN/logs/checkpoints
```

### Test on Time-Series Data

```bash
python3 test.py \
    -e deepsdf/experiments/kmelon_latent32 \
    -c configs/kmelon/minkowski_finetune_orbbec.json \
    --checkpoint_folder experiments/kmelon_latent32/TRAIN_RUN/logs/checkpoints \
    --data_dir ./data/kmelon_centered_ts/
```

---

## Evaluation Metrics

### Shape Completion Metrics (with 3D GT)

When ground truth 3D meshes/point clouds are available:

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **Chamfer Distance** | Average point-to-surface distance | [0, ∞) | Lower |
| **Precision** | % of pred points close to GT | [0, 1] | Higher |
| **Recall** | % of GT points close to pred | [0, 1] | Higher |
| **F1 Score** | Harmonic mean of P & R | [0, 1] | Higher |
| **Volume Error** | Absolute volume difference | [0, ∞) | Lower |

### Size Metrics (for time-series/historical data)

When only 2D hand measurements available:

| Metric | Description | Units | Better |
|--------|-------------|-------|--------|
| **Length Error** | Predicted vs measured length | mm | Lower |
| **Width Error** | Predicted vs measured width | mm | Lower |
| **MAE** | Mean absolute error | mm | Lower |
| **RMSE** | Root mean squared error | mm | Lower |

---

## Testing Procedure

### Step 4.1: Run Inference

```bash
python3 test.py \
    -e deepsdf/experiments/kmelon_latent32 \
    -c configs/kmelon/minkowski_finetune_orbbec.json \
    --checkpoint_folder experiments/kmelon_latent32/20251031_143000_finetune/logs/checkpoints
```

**What it does:**
1. Loads trained encoder
2. Loads frozen DeepSDF decoder
3. For each test sample:
   - Encodes partial PCD → latent code
   - Decodes latent code → SDF field
   - Extracts mesh (marching cubes)
   - Compares with GT
4. Saves results and meshes

**Output:**
```
experiments/kmelon_latent32/20251031_143000_finetune/
├── inf/
│   ├── meshes/
│   │   ├── S1_0_mesh.ply
│   │   ├── S2_0_mesh.ply
│   │   └── ...
│   └── latent_codes/
│       ├── S1_0_latent.npy
│       └── ...
└── shape_completion_results.csv
```

### Step 4.2: Evaluate Size Metrics

For time-series or historical data:

```bash
python3 evaluate_size_metrics.py \
    --mesh_dir experiments/kmelon_latent32/20251031_143000_finetune/inf/meshes \
    --gt_csv data/3DKmelonDemo/size_data.csv \
    --output experiments/kmelon_latent32/20251031_143000_finetune/size_evaluation.csv
```

**What it does:**
1. Loads predicted meshes
2. Computes length/width using PCA (principal component analysis)
3. Matches to ground truth measurements by (sample, date)
4. Computes error metrics
5. Saves detailed results

**Output:**
```
experiments/kmelon_latent32/20251031_143000_finetune/
├── size_evaluation.csv          # Detailed per-sample results
└── size_evaluation_summary.txt  # Aggregate statistics
```

---

## Time-Series Evaluation Workflow

For evaluating on historical data without 3D GT:

### Setup

**Training data:** `data/kmelon_centered/` (recent data with 3D GT)  
**Test data:** `data/kmelon_centered_ts/` (historical + recent, only has length/width GT)

### Step 1: Train on Recent Data

```bash
python3 train.py \
    -e deepsdf/experiments/kmelon_latent32 \
    -c configs/kmelon/minkowski_orbbec_fromscratch.json \
    --checkpoint_decoder 301
```

### Step 2: Test on Time-Series Data

```bash
python3 test.py \
    -e deepsdf/experiments/kmelon_latent32 \
    -c configs/kmelon/minkowski_orbbec_fromscratch.json \
    --checkpoint_folder experiments/kmelon_latent32/TRAIN_RUN/logs/checkpoints \
    --data_dir ./data/kmelon_centered_ts/
```

**Key:** `--data_dir` overrides the config to test on different data

### Step 3: Evaluate Size Metrics

```bash
python3 evaluate_size_metrics.py \
    --mesh_dir experiments/kmelon_latent32/TEST_RUN/inf/meshes \
    --gt_csv data/3DKmelonDemo/size_data.csv \
    --output experiments/kmelon_latent32/TEST_RUN/size_evaluation.csv
```

### Why This Approach?

✅ **No code pollution** - Simple CLI override  
✅ **Flexible** - Can test on any dataset  
✅ **Clear** - Command shows exactly which data is used  
✅ **Maintainable** - Easy to understand and modify

---

## Size Metrics Details

### How Length & Width are Computed

Uses **PCA (Principal Component Analysis)** on mesh vertices:

1. **Length**: Extent along 1st principal component (longest axis)
2. **Width**: Extent along 2nd principal component (perpendicular to length)
3. **Height**: Extent along 3rd principal component (shortest axis)

**Advantages:**
- ✅ Automatically finds natural orientation
- ✅ Robust to rotation/translation
- ✅ Consistent across samples
- ✅ Matches biological interpretation

### Measurement Bias Correction

PCA measurements have systematic bias vs hand measurements. To correct:

#### Step 1: Quantify Bias

```bash
python3 compare_hand_vs_mesh_size.py \
    --gt_csv data/3DKmelonDemo/size_data.csv \
    --gt_mesh_dir data/3DKmelonDemo \
    --output comparison_hand_vs_mesh.csv
```

**Output:** Learns the bias between PCA and manual measurements

#### Step 2: Apply Correction

```bash
python3 evaluate_size_metrics.py \
    --mesh_dir experiments/.../inf/meshes \
    --gt_csv data/3DKmelonDemo/size_data.csv \
    --output size_evaluation.csv \
    --correction_csv comparison_hand_vs_mesh.csv
```

**Correction model:**
```
corrected_value = slope × measured_value + intercept
```

Trained separately for length and width.

---

## Comprehensive Analysis

Analyze all experiments together:

```bash
python3 analyze_all_results.py
```

**What it does:**
1. Collects results from all experiments
2. Compares encoder performance (Minkowski vs PCE vs Pool)
3. Evaluates finetuning effectiveness
4. Detects overfitting patterns
5. Generates visualizations

**Output:** `analysis/`
```
analysis/
├── comprehensive_analysis_report.txt
├── shape_completion_summary.csv
├── size_evaluation_summary.csv
├── overfitting_analysis.csv
├── shape_completion_comparison.png
└── size_errors_by_date.png
```

### Key Analysis Components

#### 1. Encoder Comparison
Compares Minkowski vs PCE vs Pool encoders across latent sizes.

**Metrics:**
- F1 Score
- Chamfer Distance
- Inference time

#### 2. Finetuning Effectiveness
Compares pretrain+finetune vs from-scratch.

**Percentage improvement:**
```
Improvement = (Finetune_F1 - FromScratch_F1) / FromScratch_F1 × 100%
```

#### 3. Overfitting Detection
Analyzes errors across time (fruit growth stages).

**Method:**
1. Split data by date: Early (small fruits), Mid, Late (large fruits)
2. Compute MAE for each period
3. **Overfitting Score** = (Early MAE - Late MAE)
   - Positive → worse on small fruits (overfitting to large)
   - Negative → worse on large fruits (underfitting)
   - > 5mm → severe overfitting

**Interpretation:**
If model predicts similar sizes for all fruits, it learned to output average size (overfitting to training distribution).

---

## Automation Scripts

### Test All Configurations

`run_test_timeseries.sh`:
```bash
#!/bin/bash
# Test all encoder configs on time-series data

for latent in 16 64 128; do
    for encoder in minkowski pce; do
        for strategy in finetune fromscratch; do
            echo "=== ${encoder}_${strategy}_latent${latent} ==="
            python3 test.py \
                -e deepsdf/experiments/kmelon_latent${latent} \
                -c configs/kmelon/${encoder}_${strategy}.json \
                --checkpoint_folder experiments/.../logs/checkpoints \
                --data_dir ./data/kmelon_centered_ts/
        done
    done
done
```

### Evaluate All Size Metrics

`run_size_evaluations.sh`:
```bash
#!/bin/bash
# Compute size metrics for all experiments

for latent in 16 64 128; do
    for encoder in minkowski pce; do
        for strategy in finetune fromscratch; do
            python3 evaluate_size_metrics.py \
                --mesh_dir experiments/kmelon_latent${latent}/.../${encoder}_${strategy}/inf/meshes \
                --gt_csv data/3DKmelonDemo/size_data.csv \
                --output experiments/kmelon_latent${latent}/.../${encoder}_${strategy}/size_evaluation.csv
        done
    done
done
```

### Complete Analysis Pipeline

`run_complete_analysis.sh`:
```bash
#!/bin/bash
# Run complete evaluation pipeline

# Step 1: Test all configs
bash run_test_timeseries.sh

# Step 2: Evaluate size metrics
bash run_size_evaluations.sh

# Step 3: Comprehensive analysis
python3 analyze_all_results.py

echo "Analysis complete! Check analysis/ directory."
```

---

## Visualizations

### Colored Point Clouds

Visualize per-point errors with color-coded point clouds:

**Config:**
```json
{
    "save_colored_mesh": true,
    "visualization_colormap": "jet",
    "visualization_max_distance": 0.02
}
```

**Output:** `{sample}_colored.ply` files with RGB colors representing distance errors
- **Blue** = close to GT (low error)
- **Red** = far from GT (high error)

**View in:** CloudCompare, MeshLab, or Open3D

### Analysis Plots

`analyze_all_results.py` generates:

1. **Shape Completion Comparison**
   - F1 scores across configs
   - Chamfer distances across configs

2. **Size Errors by Date**
   - Length/width errors over time
   - Helps identify overfitting patterns

---

## Comparing Models

### By Latent Size

```bash
# Compare all latent sizes for Minkowski finetune
for latent in 16 32 64 128; do
    echo "=== Latent ${latent} ==="
    tail -1 experiments/kmelon_latent${latent}/.../shape_completion_results.csv | cut -d',' -f2-5
done
```

### By Encoder Type

```bash
# Compare encoders for latent 32
for encoder in minkowski pce super3d; do
    echo "=== ${encoder} ==="
    tail -1 experiments/kmelon_latent32/.../${encoder}_finetune/shape_completion_results.csv
done
```

### By Training Strategy

```bash
# Compare finetune vs from-scratch
echo "=== Finetune ==="
tail -1 experiments/kmelon_latent32/.../minkowski_finetune/shape_completion_results.csv

echo "=== From Scratch ==="
tail -1 experiments/kmelon_latent32/.../minkowski_fromscratch/shape_completion_results.csv
```

---

## Troubleshooting

### Issue: Poor reconstruction quality
**Solutions:**
1. Check if encoder was properly trained (loss converged)
2. Verify decoder checkpoint is correct
3. Check if test data is properly preprocessed
4. Try different checkpoint (not just best, try final)

### Issue: Size metrics don't match GT
**Solutions:**
1. Apply bias correction (use `--correction_csv`)
2. Verify GT CSV format matches expected
3. Check mesh filename parsing (sample, date extraction)
4. Visualize PCA axes to verify extraction

### Issue: Test crashes/OOM
**Solutions:**
1. Decrease batch size in config
2. Reduce marching cubes resolution
3. Test fewer samples at once
4. Use CPU inference (slower but more memory)

### Issue: All predictions look similar
**Symptom:** Low variance in predicted sizes/shapes

**Solutions:**
1. Check for overfitting (train vs val loss gap)
2. Ensure sufficient training data diversity
3. **CRITICAL**: Check normalization settings. If using PCE/FoldNet, `normalize_pcd` MUST be `false` to learn volume. If it's `true`, the encoder sees all objects as unit size.
4. Try larger latent size
4. Check if decoder is properly loaded

---

## Expected Results

### Shape Completion (F1 Score, higher is better)
- **Minkowski Finetune**: 0.75 - 0.85
- **Minkowski From Scratch**: 0.70 - 0.80
- **PCE Finetune**: 0.70 - 0.80
- **PCE From Scratch**: 0.65 - 0.75

### Chamfer Distance (lower is better)
- **Minkowski Finetune**: 0.004 - 0.008
- **Minkowski From Scratch**: 0.005 - 0.010
- **PCE Finetune**: 0.006 - 0.012
- **PCE From Scratch**: 0.008 - 0.015

### Size Metrics (MAE, lower is better)
- **Length Error**: 3-8 mm
- **Width Error**: 2-6 mm
- **With correction**: ~30-50% improvement

---

## Next Step

Once evaluation is complete and you have results, proceed to:
→ **[Step 5: Visualization](05_VISUALIZATION.md)**

---

## Related Files

- `test.py` - Main testing script
- `evaluate_size_metrics.py` - Size metrics evaluation
- `compare_hand_vs_mesh_size.py` - Bias quantification
- `analyze_all_results.py` - Comprehensive analysis
- `run_test_timeseries.sh` - Batch testing
- `run_size_evaluations.sh` - Batch size evaluation
- `run_complete_analysis.sh` - Complete pipeline

---

**Last Updated**: October 31, 2025

