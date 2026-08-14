# Running on the public dataset

This repository reads two dataset layouts. The one described here is the published
release; the internal working tree is covered in
[`01_DATA_PREPARATION.md`](01_DATA_PREPARATION.md). Nothing needs converting or
symlinking — point a config at the downloaded folder and the loader adapts.

## Get the data

```bash
hf download Sungjay/3DKoreanMelon --repo-type dataset \
    --local-dir data/3DKoreanMelon
```

About 334 MB: 2,738 partial point clouds over 210 fruits, one post-harvest reference scan
per fruit, and a single metadata table.

```
data/3DKoreanMelon/
├── processed/<sample_id>/<YYYYMMDD>.ply    partial cloud, one per imaging session
├── gt/<sample_id>/pointcloud.ply           post-harvest reference scan
└── metadata/
    ├── sessions.csv                        one row per fruit-session
    ├── splits.json                         train / val / test fruit ids
    └── camera/                             intrinsics and depth-to-colour extrinsics
```

Sample ids are `A1`–`A61` (greenhouse A, 2025), `B1`–`B69` and `C1`–`C80` (greenhouse B
and C, 2026). Greenhouse C is the held-out test set.

## Get the weights

Trained parameters ship with the code rather than the dataset, as a release asset — they
are only meaningful alongside the revision that produced them. About 30 MB:

```bash
curl -L -o score-weights.tar.gz \
    https://github.com/sungjay-kim/SCoRe/releases/download/v1.0/score-weights-v1.0.tar.gz
tar xzf score-weights.tar.gz          # unpacks into weights/
```

```
weights/
├── MANIFEST.md                                   what each file is and what pairs with what
├── score_temporal.pt                             point cloud encoder + temporal module
├── aniso_latent64.npz                            latent anisotropy basis
└── deepsdf_latent64/
    ├── specs.json                                decoder architecture
    ├── ModelParameters/470.pth                   decoder at epoch 470
    └── Reconstructions/470/Codes/complete/       per-fruit latent codes
```

`score_temporal.pt` and `aniso_latent64.npz` belong together — the anisotropy basis
describes one specific encoder's latent space. See `weights/MANIFEST.md`.

## Run

```bash
python temporal/test.py \
    --cfg configs/public/score_public_dataset.json \
    --checkpoint weights/score_temporal.pt \
    --fusion_mode ema_max_pc12 --ema_tau 3.5 \
    --min_n_points 300 --test_occlusion_levels 0
```

[`configs/public/score_public_dataset.json`](../configs/public/score_public_dataset.json)
is the reference configuration: every path in it is relative, so it works from a fresh
clone once the data and weights are in place.

**The fusion flags are required.** SCoRe is the parameter-free rule that
`--fusion_mode ema_max_pc12 --ema_tau 3.5` selects: a time-aware EMA over the latent with
a monotone clamp on the first two principal directions. `--fusion_mode` defaults to
`kalman`, which runs the trained Kalman module SCoRe was developed against — a different
model that produces different numbers without complaining.

Expected on load: 210 fruits, 2,738 frames across the three splits (1,212 train, 347
validation, 1,179 test), and 2,312 caliper sessions read from `sessions.csv`. Evaluation
covers the 79 test fruits whose final observation passes the 300-point filter, and should
report chamfer distance 2.76 mm, F1 at 5 mm 86.6 %, volume RMSE 47.2 mL, MAPE 14.1 %.

## What changes between the two layouts

The release groups files by kind and names them by session date; the working tree groups
by fruit and tags filenames with `_preprocessed`. Every path decision goes through
[`dataloaders/dataset_layout.py`](../dataloaders/dataset_layout.py), which detects the
layout from the directory structure, so no config flag selects between them.

| | working tree | public release |
|---|---|---|
| partial clouds | `<id>/orbbec/pointclouds/<frame>_preprocessed.ply` | `processed/<id>/<YYYYMMDD>.ply` |
| reference scan | `<id>/laser/fruit.ply` | `gt/<id>/pointcloud.ply` |
| session dates | `temporal_metadata.csv` | `metadata/sessions.csv` |
| caliper readings | `field_size_data_v2.csv` | `metadata/sessions.csv` |
| harvest volumes | `ground_truth.csv` | `metadata/sessions.csv` |

The `timestamp_csv`, `vol_map_csv`, `gt_volume_csv`, and `val_gt_csv` config keys address
the working tree's separate tables. Leave them out when pointing at the release: the
loader reads `sessions.csv` for all four.

## Training from scratch

Fine-tuning the encoder with SDF supervision needs `samples.npz` beside each reference
scan. These are derived files and are not shipped; generate them once:

```bash
python data_preparation/deepsdf/prepare_deepsdf_training_data.py \
    --src data/3DKoreanMelon/gt --skip_existing
```

The script reads each `pointcloud.ply`, estimates and orients normals itself, and writes
`samples.npz` next to it, so the released clouds carrying position only is sufficient.

## Scripts that still expect the working tree

`scripts/` and `analysis/` were written against the internal layout and address
`<id>/laser/fruit.ply` directly. They are analysis utilities rather than part of the
training or evaluation path, and have not been ported.
