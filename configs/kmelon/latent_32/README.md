# Reference configuration (latent dimension 32)

The configuration the paper reports: PCE-large encoder at latent dimension 32 with
max + mean pooling, on the DeepSDF decoder at epoch 380. The `latent_64` directory
alongside this one is an earlier campaign and does **not** correspond to the published
numbers.

| File | Stage |
|---|---|
| `pretrain.json` | 2a, encoder pretrain on the 138 synthetic-occlusion meshes |
| `finetune.json` | 2b, encoder fine-tune on the 102 in-domain training fruits |
| `temporal.json` | 3, sequential inference |
| `ablations/` | one file per row of the auxiliary and loss-weighting ablation |

Paths inside these files are the authors' absolute paths. To run from a fresh clone
against the published dataset, use
[`configs/public/score_public_dataset.json`](../../public/score_public_dataset.json),
which is relocatable, and follow
[`DOCS/00_PUBLIC_DATASET.md`](../../../DOCS/00_PUBLIC_DATASET.md).

## The time constant is a command line flag

No config here carries a `tau` field. The EMA time constant is set by `--ema_tau`, and the
paper uses **3.0 days**.

```bash
python temporal/test.py --cfg configs/kmelon/latent_32/temporal.json \
    --checkpoint <encoder .pt> --fusion_mode ema_max --ema_tau 3.0
```

`--fusion_mode ema_max` selects SCoRe, a time-aware EMA over the full latent followed by a
monotone clamp on the first principal growth direction. The default is `kalman`, a
different, trained module, so both flags are required.
