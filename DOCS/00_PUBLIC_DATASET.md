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
are only meaningful alongside the revision that produced them. About 31 MB:

```bash
curl -L -o score-weights.tar.gz \
    https://github.com/sungjay-kim/SCoRe/releases/download/v1.1/score-weights-v1.1.tar.gz
tar xzf score-weights.tar.gz          # unpacks into weights/
```

```
weights/
├── MANIFEST.md                                   what each file is and what pairs with what
├── score_encoder.pt                              fine-tuned point cloud encoder (PCE-large, D=32)
├── aniso_latent32.npz                            latent anisotropy basis
└── deepsdf_latent32/
    ├── specs.json                                decoder architecture
    ├── ModelParameters/380.pth                   decoder at epoch 380
    └── Reconstructions/380/Codes/complete/       per-fruit latent codes
```

`score_encoder.pt` and `aniso_latent32.npz` belong together — the anisotropy basis
describes one specific encoder's latent space. See `weights/MANIFEST.md`.

## Run

```bash
python temporal/test.py \
    --cfg configs/public/score_public_dataset.json \
    --checkpoint weights/score_encoder.pt \
    --fusion_mode ema_max --ema_tau 3.0 \
    --min_n_points 300 --test_occlusion_levels 0
```

[`configs/public/score_public_dataset.json`](../configs/public/score_public_dataset.json)
is the reference configuration: every path in it is relative, so it works from a fresh
clone once the data and weights are in place.

**The fusion flags are required.** SCoRe is the parameter-free rule that
`--fusion_mode ema_max --ema_tau 3.0` selects: a time-aware EMA over the latent with
a monotone clamp along its first principal direction. `--fusion_mode` defaults to
`kalman`, which runs the trained Kalman module SCoRe was developed against — a different
model that produces different numbers without complaining.

Expected on load: 210 fruits, 2,738 frames across the three splits (1,212 train, 347
validation, 1,179 test), and 2,312 caliper sessions read from `sessions.csv`. Evaluation
uses observations with at least 300 points and available volume references (caliper
proxies at intermediate visits; harvest geometry at the final visit), retaining sequences
with at least three eligible visits (79 test fruits; 79 final and 976 intermediate
observations), and should
report chamfer distance 2.97 mm, F1 at 5 mm 82.7 %, volume RMSE 52.3 mL, MAPE 15.4 %.

The paper reports 2.95 mm, 83.0 %, 50.6 mL and 15.2 % for this same model and checkpoint.
The checkpoint is unchanged. The volume gap is mainly due to the harvest reference: `harvest_volume_ml` in the release
is recomputed as the convex hull of the released reference scan, which sits a median 0.7 %
above the screened Poisson mesh volume the paper evaluated against. Substituting that one
reference accounts for 1.2 mL of the 1.7 mL RMSE difference; the remaining 0.5 mL is the
reference scans and partial clouds being read through the release layout.

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
`<id>/laser/fruit.ply` directly. Many historical utilities still require that layout. The portable
`scripts/reference/` analyses described below use the public benchmark files.


## Paper reference analyses

The dataset additionally supplies `metadata/paper_evaluation.csv` and
`benchmarks/score_reference/`. The former maps public IDs to the exact paper cohort
and original screened-Poisson references. It does not overwrite the released convex-hull
values or change the inference loader. The latter contains archived predictions,
fixed occlusion masks, expected summaries, and a checksum manifest.

To rerun the published numerical analyses on a CPU (Python 3.10+, NumPy, pandas,
SciPy with `scipy.optimize.isotonic_regression`):

```bash
python scripts/reference/verify_bundle.py
python scripts/reference/analyze_existing_predictions.py
python scripts/reference/run_statistical_uncertainty.py
python scripts/reference/summarize_robustness.py
```

Outputs go to `runs/reference_analysis/`. The first analysis recomputes volume-space
EMA/max/isotonic/logistic baselines, validation-only scalar-τ selection and linear
calibration, paired fruit bootstrap intervals, size strata, and volume decreases.
The second recomputes paired ablation and validation-τ intervals. The third recomputes
five random-history occlusion patterns (final frame always occluded), their fruit
bootstrap intervals, and the three fine-tuning-seed summaries. These commands analyze
saved model predictions; they do not rerun encoder training or neural inference.

For new inference with the released checkpoint, a preset wrapper supplies the required
fusion flags and runs from the repository root:

```bash
python scripts/evaluate_public.py
python scripts/evaluate_public.py --single-frame
```

The preset uses the existing public-layout references and the live-inference results
stated above. `--dry-run` shows the command; additional `temporal/test.py` arguments
can be appended. That script's built-in occlusion flag remains final-frame-only;
random-history results and their fixed masks are in the reference bundle.

The reference release is pinned in `scripts/reference/dataset_revision.json`. Use its
`revision` with `hf download --revision` to retrieve the exact benchmark files.
