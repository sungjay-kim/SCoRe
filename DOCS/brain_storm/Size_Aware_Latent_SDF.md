# Size-aware DeepSDF Latent (Brainstorm)

## Goal
Make the DeepSDF **latent code `z`** explicitly represent a **size feature** (and optionally a simple “shape ratio” feature), using **auxiliary supervision** derived from ellipsoid fitting on point clouds.

This is a precursor to modifying `corepp/train_deep_sdf.py` (future work; not implemented here).

---

## Current signals available
From per-sample ellipsoid fitting JSON:

- **Volume**: `volume_ml`
- **Extents**: `extents_mm` (3 values)
- **Ratio feature (optional)**: e.g. `log(extents_mm[0] / extents_mm[1])`

We already produce:
- Per-sample fits under `corepp/size_evaluation/ellipsoid_fitting/<SAMPLE>/<SAMPLE>.json`
- Summary stats + plots under `corepp/size_evaluation/ellipsoid_volume_stats/`
- 4-bin volume grouping (healthy-only quartiles) under:
  - `corepp/size_evaluation/ellipsoid_volume_stats/volume_groups_4.json`
  - `corepp/size_evaluation/ellipsoid_volume_stats/volume_groups_4.csv`

---

## Why “contrastive on `z` with 4 volume bins” is risky (be critical)
Using contrastive learning where **positives = same size bin** and **negatives = different bins** can easily hurt DeepSDF training:

- **Within-bin diversity**: very different shapes can share the same volume bin; forcing them close in latent space can reduce reconstruction capacity.
- **Boundary noise**: near-threshold samples become negatives even if size is nearly identical → unstable gradients.
- **Label bias**: volumes are derived from partial/cropped point clouds + heuristics (e.g., Z compensation), so the contrastive objective can encourage learning scan artifacts.
- **Autodecoder constraint**: in DeepSDF, `z` is the main per-shape free variable; imposing a global metric structure on `z` can conflict with reconstruction.

If contrastive is still desired later, apply it to a **projection head** `p(z)` (small MLP) rather than directly on `z`.

---

## Safer alternative: “size head” supervision (recommended first)
Instead of forcing latent distances to match bins, enforce that **size is decodable from `z`**.

Add a small predictor head:

- **Size regression**: `h_size(z) -> log(volume_ml)` (preferred over raw volume for stability)
  - loss: MSE on `log(volume_ml)`
- **or size classification**: `h_size_cls(z) -> 4-bin label`
  - loss: cross-entropy

Optional additional head:

- **Shape ratio regression**: `h_ratio(z) -> log(extent0/extent1)`
  - loss: MSE

Training objective (conceptually):
\[
L = L_\text{recon} + \lambda_\text{size} L_\text{size} + \lambda_\text{ratio} L_\text{ratio}
\]

Why this is safer:
- It encourages the network to **encode size information** in `z` without forcing unrelated shapes to collapse together.
- Much less sensitive to bin boundaries and within-bin diversity.

---

## If we still want a contrastive component later (safer version)
If a contrastive component is desired:

- Add a projection head `p(z)` (e.g., 2–16D).
- Run contrastive loss on `p(z)` only.
- Use **weaker positives** than “same bin = positive”, e.g.:
  - only near-neighbors by volume, or
  - a continuous weighting by |log(v_i)-log(v_j)|
- Keep contrastive weight small and/or ramp it up after reconstruction stabilizes.

---

## Practical notes / pitfalls
- **Scale normalization**: if any preprocessing normalizes scale away (e.g., unit box), then “absolute size” cannot be learned unless the scale factor is explicitly provided.
- **Label noise**: volumes from partial scans can bias supervision. Prefer:
  - log-volume targets,
  - robust losses (Huber),
  - and sanity checks via plots and outlier inspection.
- **Batching**: contrastive requires multiple positives/negatives per batch; DeepSDF training loops may need dataloader changes if you want strong InfoNCE behavior.

---

## Next steps (future)
- Decide target type: regression vs 4-bin classification.
- Decide whether to supervise ratio feature (and how stable it is across scans).
- Add a lightweight head and losses into `corepp/train_deep_sdf.py` with tunable lambdas and logging.


