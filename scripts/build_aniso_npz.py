"""Build aniso npz for a given latent dim using a finetuned encoder.

Computes from TRAIN split:
- v_basis_K1, v_basis_K2 (K, D): top-K PCs of within-fruit consecutive z deltas
- mu_pop (D): mean of all train z
- Sigma_shape (D, D), Sigma_shape_inv: covariance of shape-ortho components
- Sigma_shape_inv_K1
- mu_d_size_K2 (K): mean delta in size subspace
- Sigma_size_K2 (K, K), Sigma_size_inv_K2
- q90/q95/q99 mahal thresholds for shape and size
- num_train_fruits, encoder_ckpt provenance

Output keys mirror existing aniso_v25_l64_softvolFT_noAug_with_size.npz so
LatentKalmanFusion can load directly.
"""
from __future__ import annotations
import argparse, json, os, sys, re
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))
if str(ROOT / "corepp") not in sys.path: sys.path.append(str(ROOT / "corepp"))

from corepp.temporal.core import build_encoder, prepare_dataset, run_encoder_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--n_points_cache", default=None)
    ap.add_argument("--min_n_points", type=int, default=300)
    ap.add_argument("--splits", default="train",
                    help="comma-separated splits to include (e.g. 'train', 'train,val').")
    args = ap.parse_args()

    with open(args.cfg) as f: param = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_size = int(param.get("latent_size", 64))

    encoder = build_encoder(param, latent_size, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "encoder_state_dict" in ckpt:
        sd = ckpt["encoder_state_dict"]
    elif "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    # Strip any unexpected (e.g. vol_head when not present)
    enc_keys = set(encoder.state_dict().keys())
    sd_filtered = {k: v for k, v in sd.items() if k in enc_keys}
    encoder.load_state_dict(sd_filtered, strict=False)
    encoder.eval()
    print(f"[build_aniso] loaded encoder: {len(sd_filtered)}/{len(sd)} keys; latent_dim={latent_size}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    items_by_fruit = {}
    pretrain_for_ds = param["pretrain_path"] if "pretrain_path" in param else param.get("pretrained_encoder", "")
    for split in splits:
        dataset = prepare_dataset(param, pretrain=pretrain_for_ds, split=split)

        if args.n_points_cache and os.path.isfile(args.n_points_cache):
            import pandas as pd
            cache = pd.read_csv(args.n_points_cache)
            cache_map = {(r.fruit_id, str(r.date)): int(r.n_points) for r in cache.itertuples()}
            orbbec_suf = param.get("orbbec_pcd_suffix", "_preprocessed")
            def fkey(fpath):
                fid = fpath.split('/')[-4]
                base = os.path.splitext(os.path.basename(fpath))[0].replace(orbbec_suf, '')
                m = re.match(r"^(\d{8})", base)
                return (fid, m.group(1) if m else base)
            kept = [f for f in dataset.files if cache_map.get(fkey(f), args.min_n_points) >= args.min_n_points]
            before = len(dataset.files); dataset.files = kept
            print(f"[build_aniso] [{split}] min_n_points={args.min_n_points}: kept {len(kept)}/{before}")

        print(f"[build_aniso] [{split}] running encoder on {len(dataset.files)} files...")
        with torch.no_grad():
            for ds_idx in range(len(dataset.files)):
                item = dataset[ds_idx]
                fid = item.get("fruit_id", "")
                if isinstance(fid, (list, tuple)): fid = fid[0]
                fr = item.get("frame_id", "")
                if isinstance(fr, (list, tuple)): fr = fr[0]
                ts = float(item.get("timestamp", 0.0))
                latent = run_encoder_forward(encoder, item, param, device)
                z = latent.detach().cpu().numpy().reshape(-1)
                items_by_fruit.setdefault(str(fid), []).append((str(fr), ts, z))
                if (ds_idx + 1) % 200 == 0:
                    print(f"  [{split}] {ds_idx+1}/{len(dataset.files)}")

    # Sort each fruit by frame_id (date) and compute deltas
    deltas = []   # (M, D)
    z_all = []
    for fid, lst in items_by_fruit.items():
        lst.sort(key=lambda x: x[0])
        for i in range(1, len(lst)):
            d = lst[i][2] - lst[i-1][2]
            deltas.append(d)
        for fr, ts, z in lst:
            z_all.append(z)
    deltas = np.array(deltas, dtype=np.float64)  # (M, D)
    z_all = np.array(z_all, dtype=np.float64)
    print(f"[build_aniso] {len(items_by_fruit)} fruits, {len(deltas)} consecutive pairs, {len(z_all)} total z")

    D = z_all.shape[1]
    mu_pop = z_all.mean(axis=0)

    # PCA on delta — center
    delta_centered = deltas - deltas.mean(axis=0)
    # SVD
    U, S, Vt = np.linalg.svd(delta_centered, full_matrices=False)
    var = (S ** 2) / max(1, len(delta_centered) - 1)
    pc_var_explained = var / var.sum()
    print(f"[build_aniso] PC var explained (top 5): {pc_var_explained[:5]}")

    v_basis_K1 = Vt[:1]   # (1, D)
    v_basis_K2 = Vt[:2]   # (2, D)
    print(f"[build_aniso] V basis shapes: K1={v_basis_K1.shape}, K2={v_basis_K2.shape}")

    # Shape covariance: residual after removing top-K size axis
    # K=2 case
    P_shape_K2 = np.eye(D) - v_basis_K2.T @ v_basis_K2
    delta_shape_K2 = deltas @ P_shape_K2.T  # (M, D), but rank D-2
    Sigma_shape_K2 = np.cov(delta_shape_K2, rowvar=False) + 1e-6 * np.eye(D)
    Sigma_shape_inv_K2 = np.linalg.pinv(Sigma_shape_K2, rcond=0.01)

    # K=1
    P_shape_K1 = np.eye(D) - v_basis_K1.T @ v_basis_K1
    delta_shape_K1 = deltas @ P_shape_K1.T
    Sigma_shape_K1 = np.cov(delta_shape_K1, rowvar=False) + 1e-6 * np.eye(D)
    Sigma_shape_inv_K1 = np.linalg.pinv(Sigma_shape_K1, rcond=0.01)

    # Size covariance K=2
    delta_size_K2 = deltas @ v_basis_K2.T  # (M, 2)
    mu_d_size_K2 = delta_size_K2.mean(axis=0)
    delta_size_K2_centered = delta_size_K2 - mu_d_size_K2
    Sigma_size_K2 = np.cov(delta_size_K2_centered, rowvar=False) + 1e-6 * np.eye(2)
    Sigma_size_inv_K2 = np.linalg.inv(Sigma_size_K2)

    # Mahal distributions on train pairs → thresholds
    mahal_shape_K2 = ((delta_shape_K2 @ Sigma_shape_inv_K2) * delta_shape_K2).sum(axis=1)
    mahal_shape_K1 = ((delta_shape_K1 @ Sigma_shape_inv_K1) * delta_shape_K1).sum(axis=1)
    mahal_size_K2 = ((delta_size_K2_centered @ Sigma_size_inv_K2) * delta_size_K2_centered).sum(axis=1)

    q_shape_K2 = np.quantile(mahal_shape_K2, [0.9, 0.99])
    q_shape_K1 = np.quantile(mahal_shape_K1, [0.9, 0.99])
    q_size_K2 = np.quantile(mahal_size_K2, [0.9, 0.95, 0.99])

    print(f"[build_aniso] shape K2 mahal q90/q99 = {q_shape_K2}")
    print(f"[build_aniso] shape K1 mahal q90/q99 = {q_shape_K1}")
    print(f"[build_aniso] size  K2 mahal q90/q95/q99 = {q_size_K2}")

    delta_norm_mean = float(np.linalg.norm(deltas, axis=1).mean())

    out = dict(
        v_basis_K1=v_basis_K1.astype(np.float32),
        v_basis_K2=v_basis_K2.astype(np.float32),
        mu_pop=mu_pop.astype(np.float32),
        Sigma_shape=Sigma_shape_K2.astype(np.float32),       # use K2 for "default" Sigma_shape
        Sigma_shape_inv=Sigma_shape_inv_K2.astype(np.float32),
        Sigma_shape_inv_K1=Sigma_shape_inv_K1.astype(np.float32),
        delta_norm_mean=np.float32(delta_norm_mean),
        encoder_ckpt=np.array(args.checkpoint, dtype=object),
        pc_var_explained=pc_var_explained.astype(np.float32),
        num_train_fruits=np.int32(len(items_by_fruit)),
        q90_mahal_K2=np.float32(q_shape_K2[0]),
        q99_mahal_K2=np.float32(q_shape_K2[1]),
        q90_mahal_K1=np.float32(q_shape_K1[0]),
        q99_mahal_K1=np.float32(q_shape_K1[1]),
        # Size axis stats
        mu_d_size_K2=mu_d_size_K2.astype(np.float32),
        Sigma_size_K2=Sigma_size_K2.astype(np.float32),
        Sigma_size_inv_K2=Sigma_size_inv_K2.astype(np.float32),
        q90_mahal_size_K2=np.float32(q_size_K2[0]),
        q95_mahal_size_K2=np.float32(q_size_K2[1]),
        q99_mahal_size_K2=np.float32(q_size_K2[2]),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f"[build_aniso] saved -> {args.output}")
    print(f"[build_aniso] keys: {list(out.keys())}")


if __name__ == "__main__":
    main()
