# Step 6: Automation

Automate the complete pipeline with shell scripts for reproducible experiments.

---

## Overview

Automation scripts to run the complete pipeline or specific stages:
1. **Complete Pipeline** - From data to results
2. **Training Pipeline** - DeepSDF + Encoder training
3. **Evaluation Pipeline** - Testing + metrics
4. **Visualization Pipeline** - Generate all visualizations

---

## Quick Start

### Run Complete Pipeline

```bash
# First, activate the environment
./activate_corepp.sh

# Then run the pipeline
bash run_complete_pipeline.sh
```

This runs everything from data preparation to final analysis.

---

## Complete Pipeline Script

### `run_complete_pipeline.sh`

```bash
#!/bin/bash
# Complete pipeline: Data → DeepSDF → Encoder → Evaluation → Visualization

set -e  # Exit on error

echo "=========================================="
echo "TCORE Complete Pipeline"
echo "=========================================="

# ==========================================
# Stage 1: Data Preparation
# ==========================================
echo ""
echo "Stage 1/6: Data Preparation"
echo "=========================================="

cd data_preparation

# Mesh to PCD
python3 pcd_from_sfm.py \
    --src ../data/3DKmelonDemo/2_sfm/1_mesh \
    --dst ../data/kmelon

# Center PCDs
python3 ../corepp/data_preparation/postprocess_kmelon_pointclouds.py \
    --src_root ../data/kmelon \
    --dst_root ../data/kmelon_postprocessed \
    --s_min 1 --s_max 70 \
    --dark_mode fixed --dark_thr 0.08 \
    --poisson_depth 7 --density_keep_percentile 5 --n_points_out 20000 \
    --skip_no_color \
    --scale_default 0.18 \
    --scale_alt 0.22 --scale_alt_min 71 --scale_alt_max 120

# Augment data
python3 augment.py \
    --src ../data/kmelon_postprocessed \
    --dst ../data/kmelon_augment \
    --json_config_filename augment.json

# Prepare DeepSDF data
python3 prepare_deepsdf_training_data.py \
    --src ../data/kmelon_postprocessed \
    --no_of_samples 250000

# Generate synthetic partial PCDs
python3 generate_partial_pointclouds.py \
    --dataset_root ../data/kmelon_postprocessed \
    --all \
    --num 15 \
    --seed 42

cd ..

# ==========================================
# Stage 2: DeepSDF Training
# ==========================================
echo ""
echo "Stage 2/6: DeepSDF Training"
echo "=========================================="

for latent in 16 32 64 128; do
    echo "Training DeepSDF with latent size ${latent}..."
    python3 train_deep_sdf.py -e deepsdf/experiments/kmelon_latent${latent}
done

# Reconstruct shapes
bash run_scripts_reconstruct.sh

# Compute metrics
bash run_scripts_metrics.sh

# ==========================================
# Stage 3: Preprocess Orbbec Data
# ==========================================
echo ""
echo "Stage 3/6: Preprocess Orbbec Data"
echo "=========================================="

python3 preprocess_greenhouse_pcd.py

# ==========================================
# Stage 4: Encoder Training
# ==========================================
echo ""
echo "Stage 4/6: Encoder Training"
echo "=========================================="

for latent in 16 32 64 128; do
    echo "Training encoders for latent size ${latent}..."
    
    # Minkowski: Pretrain + Finetune
    python3 train.py \
        -e deepsdf/experiments/kmelon_latent${latent} \
        -c configs/kmelon/minkowski_phenobox.json \
        --checkpoint_decoder 301
    
    python3 train.py \
        -e deepsdf/experiments/kmelon_latent${latent} \
        -c configs/kmelon/minkowski_finetune_orbbec.json \
        --checkpoint_decoder 301
    
    # PCE: Pretrain + Finetune
    python3 train.py \
        -e deepsdf/experiments/kmelon_latent${latent} \
        -c configs/kmelon/pce_phenobox.json \
        --checkpoint_decoder 301
    
    python3 train.py \
        -e deepsdf/experiments/kmelon_latent${latent} \
        -c configs/kmelon/pce_finetune_orbbec.json \
        --checkpoint_decoder 301
done

# ==========================================
# Stage 5: Evaluation
# ==========================================
echo ""
echo "Stage 5/6: Evaluation"
echo "=========================================="

# Test on time-series data
bash run_test_timeseries.sh

# Evaluate size metrics
bash run_size_evaluations.sh

# Comprehensive analysis
python3 analyze_all_results.py

# ==========================================
# Stage 6: Visualization
# ==========================================
echo ""
echo "Stage 6/6: Visualization"
echo "=========================================="

bash generate_all_visualizations.sh

# ==========================================
# Complete!
# ==========================================
echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Results:"
echo "  - Analysis: ./analysis/"
echo "  - Visualizations: ./visualizations/"
echo "  - Experiments: ./experiments/"
echo "=========================================="
```

---

## Stage-Specific Scripts

### DeepSDF Training Pipeline

`run_deepsdf_pipeline.sh`:
```bash
#!/bin/bash
# Train DeepSDF for all latent sizes and compute metrics

set -e

LATENT_SIZES="16 32 64 128"

# Train all DeepSDF models
for latent in ${LATENT_SIZES}; do
    echo "=== Training DeepSDF Latent ${latent} ==="
    python3 train_deep_sdf.py -e deepsdf/experiments/kmelon_latent${latent}
done

# Reconstruct shapes
echo "=== Reconstructing shapes ==="
bash run_scripts_reconstruct.sh

# Compute metrics
echo "=== Computing reconstruction metrics ==="
bash run_scripts_metrics.sh

# Display results
echo "=== Results Summary ==="
for latent in ${LATENT_SIZES}; do
    echo "Latent ${latent}:"
    tail -1 deepsdf/experiments/kmelon_latent${latent}/reconstruction_metrics.csv
done
```

### Encoder Training Pipeline

`run_encoder_training_pipeline.sh`:
```bash
#!/bin/bash
# Train all encoder configurations

set -e

LATENT_SIZES="16 32 64 128"
ENCODERS="minkowski pce"
DECODER_CHECKPOINT=301

# Preprocess Orbbec data (only once)
echo "=== Preprocessing Orbbec data ==="
python3 preprocess_greenhouse_pcd.py

# Train all configurations
for latent in ${LATENT_SIZES}; do
    for encoder in ${ENCODERS}; do
        echo ""
        echo "=========================================="
        echo "Training: ${encoder} with latent ${latent}"
        echo "=========================================="
        
        # Pretrain on synthetic
        echo "--- Pretraining on synthetic data ---"
        python3 train.py \
            -e deepsdf/experiments/kmelon_latent${latent} \
            -c configs/kmelon/${encoder}_phenobox.json \
            --checkpoint_decoder ${DECODER_CHECKPOINT}
        
        # Finetune on real
        echo "--- Finetuning on real data ---"
        python3 train.py \
            -e deepsdf/experiments/kmelon_latent${latent} \
            -c configs/kmelon/${encoder}_finetune_orbbec.json \
            --checkpoint_decoder ${DECODER_CHECKPOINT}
        
        # Train from scratch (optional)
        echo "--- Training from scratch ---"
        python3 train.py \
            -e deepsdf/experiments/kmelon_latent${latent} \
            -c configs/kmelon/${encoder}_orbbec_fromscratch.json \
            --checkpoint_decoder ${DECODER_CHECKPOINT}
    done
done

echo ""
echo "=========================================="
echo "Encoder training complete!"
echo "=========================================="
```

### Evaluation Pipeline

`run_evaluation_pipeline.sh`:
```bash
#!/bin/bash
# Complete evaluation pipeline

set -e

echo "=========================================="
echo "Evaluation Pipeline"
echo "=========================================="

# Step 1: Test all configurations
echo ""
echo "Step 1/3: Running inference on test data..."
bash run_test_timeseries.sh

# Step 2: Compute size metrics
echo ""
echo "Step 2/3: Computing size metrics..."
bash run_size_evaluations.sh

# Step 3: Comprehensive analysis
echo ""
echo "Step 3/3: Running comprehensive analysis..."
python3 analyze_all_results.py

# Display summary
echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results available in:"
echo "  - analysis/comprehensive_analysis_report.txt"
echo "  - analysis/*.csv"
echo "  - analysis/*.png"
echo "=========================================="
```

### Visualization Pipeline

`generate_all_visualizations.sh`:
```bash
#!/bin/bash
# Generate all visualizations

set -e

OUTPUT_DIR="visualizations"
mkdir -p ${OUTPUT_DIR}/{gifs,sdf,colored,comparison}

echo "=========================================="
echo "Visualization Generation"
echo "=========================================="

# Find all test experiments
EXPERIMENTS=$(find experiments/kmelon_latent* -type d -name "inf" | sed 's|/inf||')

for experiment in ${EXPERIMENTS}; do
    echo ""
    echo "Processing: ${experiment}"
    
    exp_name=$(basename $(dirname ${experiment}))
    MESH_DIR="${experiment}/inf/meshes"
    
    if [ ! -d "${MESH_DIR}" ]; then
        echo "  Skipping (no meshes found)"
        continue
    fi
    
    # Create GIFs
    echo "  - Creating GIFs..."
    for mesh in ${MESH_DIR}/*_mesh.ply; do
        if [ -f "${mesh}" ]; then
            basename=$(basename ${mesh} _mesh.ply)
            python3 create_promo_gif.py \
                --mesh_file ${mesh} \
                --output ${OUTPUT_DIR}/gifs/${exp_name}_${basename}.gif \
                --duration 3.0 \
                --fps 30 \
                --quiet
        fi
    done
    
    # Copy colored point clouds
    echo "  - Copying colored visualizations..."
    cp ${MESH_DIR}/*_colored.ply ${OUTPUT_DIR}/colored/ 2>/dev/null || true
    
done

# Create comparison plots
echo ""
echo "Creating comparison plots..."
python3 demo_visualization.py \
    --output ${OUTPUT_DIR}/comparison/demo_comparison.png

echo ""
echo "=========================================="
echo "Visualization Complete!"
echo "=========================================="
echo "Output in: ${OUTPUT_DIR}/"
echo "=========================================="
```

---

## Utility Scripts

### Check Training Progress

`check_training_progress.sh`:
```bash
#!/bin/bash
# Check training progress across all experiments

echo "=========================================="
echo "Training Progress Check"
echo "=========================================="

# DeepSDF
echo ""
echo "=== DeepSDF Training ==="
for latent in 16 32 64 128; do
    exp_dir="deepsdf/experiments/kmelon_latent${latent}"
    if [ -f "${exp_dir}/train.log" ]; then
        echo ""
        echo "Latent ${latent}:"
        tail -3 ${exp_dir}/train.log
    fi
done

# Encoder training
echo ""
echo "=== Encoder Training ==="
for exp in experiments/kmelon_latent*/20*/train.log; do
    if [ -f "${exp}" ]; then
        exp_name=$(dirname $(dirname ${exp}))
        echo ""
        echo "$(basename ${exp_name}):"
        tail -3 ${exp}
    fi
done
```

### Clean Incomplete Runs

`clean_incomplete_runs.sh`:
```bash
#!/bin/bash
# Remove incomplete training runs

echo "Checking for incomplete runs..."

# Find directories without final checkpoint
for exp_dir in experiments/kmelon_latent*/20*/; do
    if [ -d "${exp_dir}" ]; then
        checkpoint_dir="${exp_dir}/logs/checkpoints"
        if [ ! -d "${checkpoint_dir}" ] || [ -z "$(ls -A ${checkpoint_dir} 2>/dev/null)" ]; then
            echo "Removing incomplete: ${exp_dir}"
            rm -rf ${exp_dir}
        fi
    fi
done

echo "Cleanup complete!"
```

### Backup Results

`backup_results.sh`:
```bash
#!/bin/bash
# Backup experiment results

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

echo "Creating backup in: ${BACKUP_DIR}"

mkdir -p ${BACKUP_DIR}

# Backup configs
cp -r configs ${BACKUP_DIR}/

# Backup analysis results
cp -r analysis ${BACKUP_DIR}/

# Backup visualizations
cp -r visualizations ${BACKUP_DIR}/

# Backup experiment summaries (not full checkpoints)
for exp in experiments/kmelon_latent*/20*/; do
    if [ -d "${exp}" ]; then
        exp_name=$(basename $(dirname ${exp}))
        run_name=$(basename ${exp})
        
        mkdir -p ${BACKUP_DIR}/experiments/${exp_name}/${run_name}
        
        # Copy CSV results
        cp ${exp}/*.csv ${BACKUP_DIR}/experiments/${exp_name}/${run_name}/ 2>/dev/null || true
        
        # Copy config
        cp ${exp}/config.json ${BACKUP_DIR}/experiments/${exp_name}/${run_name}/ 2>/dev/null || true
        
        # Copy best model only
        cp ${exp}/logs/checkpoints/*best*.pt ${BACKUP_DIR}/experiments/${exp_name}/${run_name}/ 2>/dev/null || true
    fi
done

echo "Backup complete!"
echo "Location: ${BACKUP_DIR}"
```

---

## Configuration Management

### Generate Config Variations

`generate_configs.sh`:
```bash
#!/bin/bash
# Generate config files for different latent sizes

LATENT_SIZES="16 32 64 128"
ENCODERS="minkowski pce"
STRATEGIES="phenobox finetune_orbbec orbbec_fromscratch"

for encoder in ${ENCODERS}; do
    for strategy in ${STRATEGIES}; do
        template="configs/kmelon/${encoder}_${strategy}.json"
        
        if [ ! -f "${template}" ]; then
            continue
        fi
        
        for latent in ${LATENT_SIZES}; do
            output_dir="configs/kmelon/latent_${latent}"
            mkdir -p ${output_dir}
            
            # Update latent_size in config
            cat ${template} | \
                sed "s/\"latent_size\": [0-9]*/\"latent_size\": ${latent}/" > \
                ${output_dir}/${encoder}_${strategy}.json
            
            echo "Generated: ${output_dir}/${encoder}_${strategy}.json"
        done
    done
done
```

### Validate Configs

`validate_configs.sh`:
```bash
#!/bin/bash
# Validate all configuration files

echo "Validating configuration files..."

for config in configs/kmelon/**/*.json; do
    echo "Checking: ${config}"
    
    # Check JSON syntax
    if ! python3 -c "import json; json.load(open('${config}'))" 2>/dev/null; then
        echo "  ERROR: Invalid JSON syntax"
        continue
    fi
    
    # Check required fields
    required_fields="encoder latent_size batch_size lr epoch"
    for field in ${required_fields}; do
        if ! grep -q "\"${field}\"" ${config}; then
            echo "  WARNING: Missing field '${field}'"
        fi
    done
    
    echo "  OK"
done

echo "Validation complete!"
```

---

## Experiment Management

### List All Experiments

`list_experiments.sh`:
```bash
#!/bin/bash
# List all experiments with their status

echo "=========================================="
echo "Experiment Summary"
echo "=========================================="

for latent_dir in experiments/kmelon_latent*; do
    if [ ! -d "${latent_dir}" ]; then
        continue
    fi
    
    latent=$(basename ${latent_dir} | sed 's/kmelon_latent//')
    
    echo ""
    echo "Latent ${latent}:"
    echo "---"
    
    for exp in ${latent_dir}/20*/; do
        if [ ! -d "${exp}" ]; then
            continue
        fi
        
        run_name=$(basename ${exp})
        
        # Check for completion
        if [ -f "${exp}/logs/checkpoints/*_final_model.pt" ]; then
            status="✓ Complete"
        elif [ -f "${exp}/train.log" ]; then
            status="⏳ Running"
        else
            status="✗ Failed/Incomplete"
        fi
        
        # Check for results
        if [ -f "${exp}/shape_completion_results.csv" ]; then
            results=" [Has results]"
        else
            results=""
        fi
        
        echo "  ${run_name}: ${status}${results}"
    done
done
```

### Compare Experiment Results

`compare_experiments.sh`:
```bash
#!/bin/bash
# Compare results across experiments

echo "=========================================="
echo "Experiment Comparison"
echo "=========================================="

# Compare by latent size
echo ""
echo "=== By Latent Size (Minkowski Finetune) ==="
printf "%-10s %-15s %-15s %-10s\n" "Latent" "Chamfer" "F1" "Time(s)"
echo "-----------------------------------------------------------"

for latent in 16 32 64 128; do
    result_file=$(find experiments/kmelon_latent${latent} -name "shape_completion_results.csv" | grep "finetune" | head -1)
    
    if [ -f "${result_file}" ]; then
        chamfer=$(tail -1 ${result_file} | cut -d',' -f3)
        f1=$(tail -1 ${result_file} | cut -d',' -f5)
        time=$(tail -1 ${result_file} | cut -d',' -f7)
        
        printf "%-10s %-15s %-15s %-10s\n" "${latent}D" "${chamfer}" "${f1}" "${time}"
    fi
done

# Compare by encoder
echo ""
echo "=== By Encoder (Latent 32, Finetune) ==="
printf "%-15s %-15s %-15s\n" "Encoder" "Chamfer" "F1"
echo "-----------------------------------------------"

for encoder in minkowski pce super3d; do
    result_file=$(find experiments/kmelon_latent32 -name "shape_completion_results.csv" | grep "${encoder}" | grep "finetune" | head -1)
    
    if [ -f "${result_file}" ]; then
        chamfer=$(tail -1 ${result_file} | cut -d',' -f3)
        f1=$(tail -1 ${result_file} | cut -d',' -f5)
        
        printf "%-15s %-15s %-15s\n" "${encoder}" "${chamfer}" "${f1}"
    fi
done
```

---

## Parallel Execution

For faster execution on multi-GPU systems:

### Parallel Training

`run_parallel_training.sh`:
```bash
#!/bin/bash
# Train multiple experiments in parallel

LATENT_SIZES=(16 32 64 128)
GPUS=(0 1 2 3)

# Function to train on specific GPU
train_on_gpu() {
    latent=$1
    gpu=$2
    
    export CUDA_VISIBLE_DEVICES=${gpu}
    
    python3 train_deep_sdf.py \
        -e deepsdf/experiments/kmelon_latent${latent} \
        > logs/deepsdf_latent${latent}.log 2>&1
}

# Launch training jobs
for i in ${!LATENT_SIZES[@]}; do
    latent=${LATENT_SIZES[$i]}
    gpu=${GPUS[$i]}
    
    echo "Launching latent ${latent} on GPU ${gpu}"
    train_on_gpu ${latent} ${gpu} &
done

# Wait for all to complete
wait

echo "All training complete!"
```

---

## Troubleshooting

### Issue: Scripts fail midway
**Solution:** Use `set -e` at start of scripts to exit on error

### Issue: Out of disk space
**Solution:** Run `clean_incomplete_runs.sh` and remove old experiments

### Issue: Configs not found
**Solution:** Check paths are relative to script location, use absolute paths

### Issue: GPU out of memory during parallel runs
**Solution:** Reduce batch size or number of parallel jobs

---

## Best Practices

1. **Always backup before major changes**
   ```bash
   bash backup_results.sh
   ```

2. **Monitor progress regularly**
   ```bash
   watch -n 60 bash check_training_progress.sh
   ```

3. **Validate configs before long runs**
   ```bash
   bash validate_configs.sh
   ```

4. **Clean up failed runs**
   ```bash
   bash clean_incomplete_runs.sh
   ```

5. **Use tmux/screen for long runs**
   ```bash
   tmux new -s training
   bash run_complete_pipeline.sh
   # Ctrl+B, D to detach
   ```

---

## Related Files

- `run_scripts_reconstruct.sh` - DeepSDF reconstruction
- `run_scripts_metrics.sh` - Reconstruction metrics
- `run_test_timeseries.sh` - Batch testing
- `run_size_evaluations.sh` - Size metric evaluation
- `run_complete_analysis.sh` - Complete analysis
- `check_reconstruction_progress.sh` - Monitor reconstruction

---

**Last Updated**: October 31, 2025

