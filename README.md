# SCoRe: Sequential CoRe for 3D Shape Completion of Oriental Melon

> **Parameter-free temporal latent fusion** that extends single-frame CoRe-based 3D shape completion (Blok et al., 2025) into a per-fruit sequence model. The fusion adds **no learned parameters at deployment**; at the first visit, the output reduces to the single-frame encoder output. Used to track per-fruit volume across the growing season for oriental melon (*Cucumis melo* L. var. *makuwa*).

![Last-frame mesh comparison on one representative test fruit: partial PCD input, reference GT, SCoRe, CoRe++ (PCE), T-CoRe.](assets/figures/mesh_rotation_demo.gif)

---

## Overview

A robotic platform revisits each fruit every 2 to 3 days. Each visit produces one partial point cloud from an Orbbec Femto Mega RGB-D sensor, and SCoRe accumulates information across visits in the latent space of a frozen DeepSDF decoder. The trajectory, rather than any single snapshot, becomes the relevant signal for downstream growth monitoring and harvest decisions.

![Three-stage pipeline. Stage 1 trains the DeepSDF decoder on per-fruit ground-truth meshes; Stage 2 trains the point cloud encoder against the Stage 1 per-fruit latents; Stage 3 (inference only) feeds each session's encoder output z_t together with the running fused state μ_{t-1} through the temporal latent fusion module, and decodes the fused latent μ_t into the predicted 3D shape completion.](assets/figures/pipeline_stages.png)

- **Time-aware EMA** over the full latent — irregular session intervals are absorbed by weighting each new observation by how long it has been since the last visit.
- **Hard monotone clamp** on the leading principal direction of the within-fruit consecutive-frame latent change Δz = z_{t+1} − z_t, encoding the prior that fruit volume does not decrease before harvest.
- **Frozen DeepSDF decoder** — only the encoder is trained. Stage 3 (fusion) is inference time only and learns nothing.
- **Weak supervision** at fine-tuning via an **occupancy volume** V_occ = Σ σ(−SDF/T) · v_voxel, supervised against a per-fruit-corrected ellipsoid GT V_ell on intermediate frames.

![Temporal latent fusion module (Stage 3): (a) an offline PCA pre-pass identifies the leading growth axis v_pc1 from per-fruit consecutive-frame latent differences Δz; (b) the time-aware EMA smooths z_t into μ̃_t; (c) the growth-wise monotone clamp projects μ̃_t onto that axis and enforces a running maximum against the prior s_{t-1}; the clamped projection is recomposed with the orthogonal residual to form μ_t.](assets/figures/fusion_module.png)

---

## Repository Layout

```
corepp/
├── data_preparation/       # PCD generation, splits, augmentations, metadata
├── deepsdf/                # DeepSDF decoder (Park et al. 2019)
├── networks/               # Encoders (PCE, PCE-large, Minkowski) + decoder wrappers
├── temporal/               # SCoRe pipeline (inference path only)
│   ├── test.py             # inference: single-frame OR sequential (EMA + monotone clamp)
│   ├── train.py            # encoder pretrain + fine-tune (last-frame super + V_occ weak sup)
│   ├── core.py             # encoder builder, occlusion ops, fusion runtime
│   └── utils.py            # temporal metadata index
├── analysis/               # offline latent / size analyses
├── scripts/                # PCA pre-pass, ablation aggregation, figure builders
├── configs/kmelon/         # experiment configs (latent_32 = reference; latent_64 = earlier campaign)
├── DOCS/                   # detailed walkthroughs
├── train.py / test.py      # static CoRe++ baseline entry points
└── train_deep_sdf.py       # DeepSDF decoder training
```

See [`CHANGELOG.md`](CHANGELOG.md) for the full diff against upstream `corepp`.

---

## Pipeline (3 Stages)

| # | Stage | Script | Reference config (D = 32, PCE-large, max + mean) |
|---|---|---|---|
| 1 | DeepSDF decoder | `train_deep_sdf.py` | `deepsdf/experiments/.../specs.json`, decoder epoch 380 |
| 2a | Encoder pretrain (138 synth meshes) | `train.py` | `configs/kmelon/latent_32/pretrain.json` |
| 2b | Encoder fine-tune (102 real fruits) | `train.py` | `configs/kmelon/latent_32/finetune.json` |
| 3 | Temporal latent fusion *(inference only)* | `temporal/test.py` | `configs/kmelon/latent_32/temporal.json` |

Stage 3 is parameter-free. The same fine-tuned encoder checkpoint produces both the single-frame CoRe++ baseline and the SCoRe sequence output; the only difference at inference is whether the fusion is applied. See [`configs/kmelon/latent_32/README.md`](configs/kmelon/latent_32/README.md) for the configs and [`DOCS/00_PUBLIC_DATASET.md`](DOCS/00_PUBLIC_DATASET.md) to reproduce without retraining.

---

## Headline Results (last frame, no added occlusion)

| Method | RMSE (mL) ↓ | MAPE (%) ↓ | F1 @ 5 mm (%) ↑ | d_CD (mm) ↓ |
|---|---:|---:|---:|---:|
| 2D De-occlusion + Reg (conventional image-based reference) | 95.0 | 31.21 | — | — |
| T-CoRe (Magistri et al. 2024) | 130.4 | 34.88 | 60.97 | 4.71 |
| CoRe++ (PCE-large, D = 32) | 87.4 | 24.93 | 66.13 | 4.47 |
| **SCoRe (this work)** | **50.6** | **15.19** | **83.00** | **2.95** |
| DeepSDF decoder oracle, latent optimised against the GT mesh | 8.7 | 1.84 | 98.99 | 1.22 |

Under heavy 60 % occlusion of the last frame, SCoRe holds last-frame RMSE at 70.3 mL
(CoRe++ 116.7 mL) and F1 @ 5 mm at 77.5 %. The decoder oracle auto-decodes the
ground-truth mesh and is not a deployable method.

![Per-fruit mesh time series for a representative test fruit across the growing season. Rows: partial PCD input, CoRe++ (PCE) single-frame mesh, SCoRe sequential mesh. The PCE row jitters frame-to-frame on the same dates that SCoRe smooths into a nearly monotone growth sequence; the last-day column is error-colored against the 3D reconstruction GT.](assets/figures/mesh_timeseries.png)

---

## Quick Start

### Environment

```bash
# Activate the corepp micromamba env (see ENV_SETUP.md for first-time setup)
source activate_corepp.sh
```

### Train the encoder (Stages 2a → 2b)

```bash
# Pretrain on 138 synthetic-occlusion meshes
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg configs/kmelon/latent_32/pretrain.json

# Fine-tune on real partial PCDs from the 102 in-domain training fruits
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg configs/kmelon/latent_32/finetune.json
```

### Inference: single-frame baseline vs. SCoRe

```bash
# CoRe++ baseline (no fusion)
python temporal/test.py \
    --cfg configs/kmelon/latent_32/finetune.json \
    --checkpoint <finetune_run>/logs/checkpoints/best_model.pt \
    --no_temporal

# SCoRe (Stage 3: EMA over the latent + monotone clamp on PC1)
python temporal/test.py \
    --cfg configs/kmelon/latent_32/temporal.json \
    --checkpoint <finetune_run>/logs/checkpoints/best_model.pt \
    --fusion_mode ema_max --ema_tau 3.0
```

The PC1 growth axis is computed once per encoder by an offline PCA on within-fruit consecutive-frame Δz (see `scripts/`), then reused at inference. The fusion flags are required, since `--fusion_mode` defaults to a different module; [`configs/kmelon/latent_32/README.md`](configs/kmelon/latent_32/README.md) explains them.

---

## Dataset

210 oriental melon fruits across two seasons and three greenhouses at the Seongju Oriental Melon Research Institute, Korea:

| Season | Greenhouse | Cultivar | Window | Fruits | Sample ids |
|:---:|:---:|:---:|:---:|---:|:---:|
| 2025 | A | *Eliteggul* | 1 May – 4 Jun 2025 | 61 | `A1`–`A61` |
| 2026 | B | *Alchanggul* | 11 Mar – 16 Apr 2026 | 69 | `B1`–`B69` |
| 2026 | C | *Alchanggul* | 11 Mar – 16 Apr 2026 | 80 | `C1`–`C80` |

Split: 102 train / 28 validation / 80 *Alchanggul* test (greenhouse C, April 2026 last-frame window). Ground truth is a per-fruit multi-view 3D reconstruction at harvest (last frame); intermediate frames carry a per-fruit-corrected caliper ellipsoid volume as a surrogate reference.

**Download: [`Sungjay/3DKoreanMelon`](https://huggingface.co/datasets/Sungjay/3DKoreanMelon)** · DOI [10.57967/hf/9982](https://doi.org/10.57967/hf/9982) · CC BY 4.0 · ~334 MB

```bash
hf download Sungjay/3DKoreanMelon --repo-type dataset --local-dir data/3DKoreanMelon
```

This code reads that layout directly, with no conversion step. See [`DOCS/00_PUBLIC_DATASET.md`](DOCS/00_PUBLIC_DATASET.md) for the run command and the trained weights.

---

## Acknowledgements & Citation

SCoRe builds on **CoRe++** by Blok et al.:

```BibTeX
@article{BLOK2025109673,
    title   = {High-throughput 3D shape completion of potato tubers on a harvester},
    author  = {Pieter M. Blok and Federico Magistri and Cyrill Stachniss and Haozhou Wang and James Burridge and Wei Guo},
    journal = {Computers and Electronics in Agriculture},
    volume  = {228},
    pages   = {109673},
    year    = {2025},
    doi     = {10.1016/j.compag.2024.109673},
}
```

CoRe++ itself extends Magistri's original CoRe ([magistri2022ral-iros](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/magistri2022ral-iros.pdf)), and the shape prior is the DeepSDF decoder of Park et al. ([CVPR 2019](https://arxiv.org/abs/1901.05103)).

---

## License

Inherits the upstream CoRe++ license — see [`LICENSE`](LICENSE).
