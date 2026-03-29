# Temporal CoRe++ (Time-Aware CoRe)

This document describes the temporal extension for CoRe++, designed to track fruit growth over time by fusing sequential partial observations into a coherent latent representation before decoding with a frozen DeepSDF decoder.

## Motivation

The static encoder predicts a latent vector `z` per frame. When fruits are partially occluded or measurements are sparse, each prediction may drift, producing jittery growth curves. A temporal encoder adds memory so that:

- historical predictions guide the next estimate,
- irregular sampling cadences (missing days) are handled explicitly via a time-aware gate,
- the final latent sequence remains compatible with the pretrained DeepSDF decoder.

## Architecture Overview

```
partial point cloud/image  ──► Encoder ──► z_t ┐
                                         │
                previous latent h_{t-Δt} ─┼─► Fusion Module ──► h_t ──► Decoder (frozen)
                              Δt (days) ──┘                  │
                                                              └─► L_super (latent supervision)
```

- **Encoder:** any of the existing CoRe encoders (PointCloud/PCE, Minkowski, Pool, etc.). The temporal configs reuse the same constructor logic as `train.py`.
- **Fusion Module:** a lightweight gate that blends the new latent with the previous hidden state, conditioned on the elapsed time `Δt` between measurements. See `networks/temporal_models.py` for details.
- **Decoder:** the DeepSDF decoder stays frozen; temporal training only updates the encoder and fusion weights.

## Data Requirements

1. **Time-series split:** `data/kmelon_centered/` is reused for temporal runs; sequences are created by ordering the per-fruit frames via metadata. A `split.json` selects the sequences for train/val/test just like the static case.
2. **Timestamp metadata:** provide a CSV with at least the columns `fruit_id`, `frame_id`, `timestamp`. Timestamps can be ISO strings or numeric. The helper `TemporalMetadataIndex` parses the CSV and returns per-frame timestamps plus chronological ordering. If the CSV is omitted, frames are ordered lexicographically and `Δt=1` is used.

Example (`data/kmelon_centered/temporal_metadata.csv`):

```
fruit_id,frame_id,timestamp
KMELON_0001,20240501_01,2024-05-01
KMELON_0001,20240505_00,2024-05-05
KMELON_0001,20240508_03,2024-05-08
```

## Training Script

`temporal/train.py` mirrors `train.py` but iterates over full sequences. For every fruit sequence it:

1. Resets the hidden state.
2. Encodes each frame to `z_t`.
3. Computes `Δt` from metadata.
4. Updates the hidden state via the fusion gate.
5. Applies latent supervision (`SuperLoss`) between `h_t` and the reference DeepSDF latent for that fruit.
6. Accumulates the loss across the sequence, backpropagates once per sequence, and logs gate statistics.

### Key CLI Arguments

```
python3 temporal/train.py \
    --cfg configs/kmelon/latent_32/temporal_pce.json \
    --experiment deepsdf/experiments/kmelon_latent32 \
    --checkpoint_decoder 301 \
    --tensorboard_port 6060
```

Use `--tensorboard_port` to automatically launch TensorBoard for the current run (omit the flag if you prefer to start it manually later).

Need metrics only for the final frame of each sequence? Set `"metrics_final_frame_only": true` in your config—`temporal/test.py` will still save meshes/latents for every frame but will report Chamfer/F1 only for the last timestamp per fruit.
Use `--resume path/to/checkpoint.pt` if you want to continue training from an earlier run (the checkpoint must have been produced by `temporal/train.py`, since it needs encoder/fusion/optimizer state and saved `global_step`/`epoch`).

Additional config keys:

| Key | Description |
| --- | --- |
| `timestamp_csv` | Path to the metadata CSV described above. |
| `min_sequence_length` | Skip sequences shorter than this length. |
| `delta_time_unit` | Rescale factor applied to raw deltas (e.g. `86400` to convert seconds→days). |
| `fusion_hidden_size` | Width of the MLP inside the fusion gate. |
| `reset_on_gap` | If `true`, reset the hidden state when `Δt` exceeds `max_gap`. |
| `max_gap` | Maximum allowed delta before forcing a reset (in the same units as `Δt`). |

**Epochs & resume:** `epoch` is interpreted as the number of epochs to train. When using `--resume`, the script treats `epoch` as “additional epochs” (e.g., `epoch=20` + `--resume` will run 20 more epochs beyond the checkpoint’s epoch index). If you want to reach a specific absolute epoch, set `epoch` accordingly (e.g., `epoch = desired_total - checkpoint_epoch`).

Logs and checkpoints follow the same auto-timestamped directory layout introduced in `train.py`.

## Testing / Inference

`temporal/test.py` loads the trained encoder + fusion module and the frozen decoder to produce per-frame meshes and metrics. It mirrors `test.py`, but it walks each sequence sequentially to maintain temporal context. Outputs:

- `inf/meshes/{fruit}_{frame}.ply`
- `inf/latent_codes/{fruit}_{frame}.npy`
- `temporal_metrics.csv` (per-frame Chamfer/F1 + optional size metrics)

## Configuration Templates

See `configs/kmelon/latent_32/temporal_pce*.json` for reference setups:

- `temporal_pce.json`: pretraining on synthetic Phenobox sequences.
- `temporal_pce_finetune_orbbec.json`: finetuning on real Orbbec sequences.

Both reuse the standard fields (`batch_size`, `lr`, `encoder`, etc.) plus the temporal extensions described above.

## Next Steps

1. **Alternative fusion cells:** try GRU/LSTM style gates (`ALTERNATIVE_DISENTANGLEMENT.md` sketches a disentangled growth cell).
2. **Disentangled latent heads:** split `h_t` into shape vs growth-rate components, regularize each with dedicated objectives.
3. **Sequence-level metrics:** add penalties for non-monotonic growth or sudden jumps.

Feedback welcome—this is the baseline implementation meant to unblock experimentation with temporal memory.
