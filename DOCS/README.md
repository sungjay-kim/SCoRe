# TCORE Documentation

Complete workflow guide for 3D fruit shape completion using DeepSDF and point cloud encoders.

---

## 📖 Quick Navigation

Follow this workflow in order for a complete pipeline:

| Step | Document | Description |
|------|----------|-------------|
| **1** | [Data Preparation](01_DATA_PREPARATION.md) | Mesh → PCD → Centering → Augmentation → Partial PCDs |
| **2** | [DeepSDF Training](02_DEEPSDF_TRAINING.md) | Train DeepSDF decoder with various latent sizes |
| **3** | [Encoder Training](03_ENCODER_TRAINING.md) | Train encoders (Pool/Minkowski/PCE) with pretrain/finetune |
| **4** | [Evaluation](04_EVALUATION.md) | Test and evaluate with shape & size metrics |
| **5** | [Visualization](05_VISUALIZATION.md) | Visualize SDFs and generate presentation GIFs |
| **6** | [Automation](06_AUTOMATION.md) | Automated scripts for complete pipeline |

**Having issues?** → See [Troubleshooting](TROUBLESHOOTING.md)

---

## 🚀 Quick Start

### For New Users

1. **Prepare your data**: Follow [Step 1: Data Preparation](01_DATA_PREPARATION.md)
2. **Train DeepSDF**: Follow [Step 2: DeepSDF Training](02_DEEPSDF_TRAINING.md)
3. **Train encoder**: Follow [Step 3: Encoder Training](03_ENCODER_TRAINING.md)
4. **Evaluate**: Follow [Step 4: Evaluation](04_EVALUATION.md)

### For Quick Testing

If you already have trained models, jump to:
- [Testing procedure](04_EVALUATION.md#testing-procedure)
- [Visualization](05_VISUALIZATION.md)

---

## 🎯 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  1. DATA PREPARATION                                            │
├─────────────────────────────────────────────────────────────────┤
│  GT Mesh → Point Cloud → Centering → Augmentation              │
│  → DeepSDF Data → Synthetic Partial PCDs                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. DEEPSDF TRAINING                                            │
├─────────────────────────────────────────────────────────────────┤
│  Train SDF decoder with different latent sizes (16/32/64/128)  │
│  → Reconstruct shapes → Compute metrics → Select best weights  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. ENCODER TRAINING                                            │
├─────────────────────────────────────────────────────────────────┤
│  Train encoder (Pool/Minkowski/PCE) to predict latent codes    │
│  Strategy: Pretrain on synthetic → Finetune on real data       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. EVALUATION                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Test on time-series data                                      │
│  Metrics: Chamfer distance, F1, Precision, Recall              │
│  Size metrics: Length & width errors                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. VISUALIZATION                                               │
├─────────────────────────────────────────────────────────────────┤
│  Visualize SDF reconstructions                                  │
│  Generate promotional GIFs for presentations                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Available Encoders

| Encoder | Config Prefix | Description | Best For |
|---------|--------------|-------------|----------|
| **Minkowski** | `minkowski_*` | Sparse convolution encoder | Partial point clouds (recommended) |
| **PCE** | `pce_*` | PointNet-style encoder | Dense point clouds |
| **Pool (Super3D)** | `super3d` | RGBD pooling encoder | RGB-D images |

---

## 📊 Data Types

| Type | Name | Description | Location |
|------|------|-------------|----------|
| **D1** | Orbbec | Greenhouse partial PCDs (noisy) | `data/kmelon_centered/*/orbbec/` |
| **D2** | Laser | Complete laser scans (GT) | `data/kmelon_centered/*/laser/fruit.ply` |
| **D3** | Phenobox | Synthetic partial PCDs | `data/kmelon_centered/*/phenobox/` |

---

## 🔧 Configuration Files

All configuration files are located in: `configs/kmelon/`

### Minkowski Encoder
- `minkowski_phenobox.json` - Pretrain on synthetic (D3)
- `minkowski_finetune_orbbec.json` - Finetune on real data (D1)
- `minkowski_orbbec_fromscratch.json` - Train from scratch on D1

### PCE Encoder
- `pce_phenobox.json` - Pretrain on synthetic (D3)
- `pce_finetune_orbbec.json` - Finetune on real data (D1)
- `pce_orbbec_fromscratch.json` - Train from scratch on D1

---

## 📁 Directory Structure

```
corepp/
├── data/
│   ├── 3DKmelonDemo/          # Original GT meshes from SfM
│   ├── kmelon/                # Processed point clouds
│   ├── kmelon_centered/       # Centered data for training
│   ├── kmelon_augment/        # Augmented data
│   └── kmelon_centered_ts/    # Time-series data for evaluation
│
├── deepsdf/
│   └── experiments/
│       ├── kmelon_latent16/   # 16D latent space experiments
│       ├── kmelon_latent32/   # 32D latent space experiments
│       ├── kmelon_latent64/   # 64D latent space experiments
│       └── kmelon_latent128/  # 128D latent space experiments
│
├── experiments/               # Encoder training experiments
│   ├── kmelon_latent16/
│   ├── kmelon_latent32/
│   ├── kmelon_latent64/
│   └── kmelon_latent128/
│
├── configs/kmelon/           # Configuration files
├── data_preparation/         # Data preprocessing scripts
├── DOCS/                     # Documentation (you are here!)
└── analysis/                 # Analysis results
```

---

## 🎓 Key Concepts

### Latent Space Dimensions
- **16D**: Compact, fast, may underfit
- **32D**: Balanced (baseline)
- **64D**: More expressive
- **128D**: Highest capacity, risk of overfitting

### Training Strategies
- **From Scratch**: Train directly on target data
- **Pretrain + Finetune**: Pretrain on synthetic, finetune on real (recommended)

### Coordinate Systems
All data is centered at origin `(0, 0, 0)` for consistent normalization

---

## 💡 Tips

1. **First time setup?** Follow all 6 steps in order
2. **Already have data?** Skip to Step 2 (DeepSDF)
3. **Have trained DeepSDF?** Skip to Step 3 (Encoder)
4. **Want to test?** Go to Step 4 (Evaluation)
5. **Need pretty pictures?** Check Step 5 (Visualization)

---

## 📝 Citation

If you use this code, please cite:
```
[Your citation here]
```

---

## 📧 Support

For questions or issues:
1. Check [Troubleshooting](TROUBLESHOOTING.md)
2. Review the relevant step documentation
3. Check logs in experiment directories

---

**Last Updated**: October 31, 2025

