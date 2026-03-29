# Disentanglement Ideas for Temporal CoRe++

The first temporal prototype keeps a single latent vector `h_t`. This note collects alternative heads that explicitly model *shape*, *scale*, and *growth rate* so that downstream phenotyping tasks can query each component separately.

## 1. Two-Head Latent (Shape vs. Growth)

Split the latent into two subvectors:

- `h_t^shape` (dimension `d_shape`) feeds the DeepSDF decoder.
- `h_t^growth` (dimension `d_growth`) predicts scalar traits such as major-axis length or daily growth rate.

Training losses:

1. `L_shape = ||h_t^shape - z_sdf||_2^2`
2. `L_growth = MAE(predicted_size, measured_size)` where `predicted_size` is obtained from a small MLP fed by `h_t^growth`.
3. Optional KL regularizer to keep the growth head smooth across time.

## 2. Residual Growth Cell

Instead of gating between `z_t` and `h_{t-1}`, treat growth as a residual increment:

```
Δg_t = MLP([z_t, h_{t-1}, Δt])
h_t = h_{t-1} + Δg_t
```

Regularize `Δg_t` to stay small and penalize sign flips when domain knowledge dictates monotonic growth.

## 3. Factorized Decoder Adaptor

Keep the static decoder untouched but learn a small adaptor that shifts the latent before decoding:

```
shape_latent = h_t[:d]
scale_latent = h_t[d:]
ψ = adaptor(scale_latent)
latent_for_decoder = shape_latent + ψ
```

The adaptor can learn global size biases per cultivar or growth stage. This is useful when the same latent must describe fruits of different absolute sizes.

## 4. Temporal Contrastive Objective

Add a contrastive loss across time so that consecutive frames stay close while distant frames drift proportionally to the elapsed time:

```
L_contrast = Σ_t ||h_t - h_{t-1}||_2^2 / Δt
```

Combine with negative pairs drawn from different fruit IDs to keep representations fruit-specific.

---
These ideas are intentionally lightweight so they can be bolted onto the existing `TemporalFusionModule`. Once the baseline pipeline is stable we can promote any of them into first-class modules.
