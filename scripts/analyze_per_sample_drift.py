"""Per-sample latent drift across occlusion levels.

For each fruit, we have 6 latents (occ = 0, 10, 20, 30, 40, 50). We compare:

  - intra-fruit drift  : how far does the same fruit's latent move
                          across occlusion levels?
  - inter-fruit spread : how far apart are different fruits at the same
                          occlusion level?

If shape/content dominates and point count is irrelevant in the normal
operating range:
  intra-fruit drift << inter-fruit spread

We also flag fruits that DO drift toward the fallback region — those should
be the ones whose post-occlusion point count drops below ~200.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import open3d as o3d

PROJECT_ROOT = Path("/home/mainuser/mount/sungjay/tcore")
RUN_DIR = (PROJECT_ROOT /
           "corepp/experiments/kmelon_latent32_v3/_occlusion_sweep_v5_patch/"
           "static_maxmean_masked_small_with_latents")
LAT_DIR = RUN_DIR / "latents"
DATA = PROJECT_ROOT / "corepp/data/kmelon_centered"
OCC_LEVELS = [0, 10, 20, 30, 40, 50]


def load_latent(fruit_id: str, frame_id: str, occ: int) -> np.ndarray:
    suffix = f"_occ{occ}" if occ > 0 else ""
    p = LAT_DIR / f"{fruit_id}_{frame_id}{suffix}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float64).reshape(-1)


def n_orig(fruit_id: str, frame_id: str) -> int:
    p = DATA / fruit_id / "orbbec" / "pointclouds" / f"{frame_id}_preprocessed.ply"
    if not p.exists():
        return -1
    return int(np.asarray(o3d.io.read_point_cloud(str(p)).points).shape[0])


def main() -> None:
    # Discover (fruit_id, frame_id) pairs from latent files (occ=0 ones)
    pairs = []
    for f in LAT_DIR.glob("*.npy"):
        if "_occ" in f.stem:
            continue
        fid, frame = f.stem.rsplit("_", 1)
        pairs.append((fid, frame))
    print(f"[load] {len(pairs)} fruits")

    # Build per-fruit latent matrix Z_fruit [6 x 32]
    fruit_data = []
    for fid, frame in sorted(pairs):
        zs = []
        ok = True
        for occ in OCC_LEVELS:
            z = load_latent(fid, frame, occ)
            if z is None:
                ok = False
                break
            zs.append(z)
        if not ok:
            continue
        n0 = n_orig(fid, frame)
        Z = np.stack(zs)  # (6, 32)
        # intra-fruit drift: pairwise distance across occ levels
        diffs = Z[:, None, :] - Z[None, :, :]
        pairwise = np.linalg.norm(diffs, axis=2)
        upper = pairwise[np.triu_indices(6, k=1)]  # 15 pairs
        # specifically: occ=0 → occ=50 distance
        d_0_to_50 = float(np.linalg.norm(Z[0] - Z[5]))
        fruit_data.append({
            "fruit_id": fid,
            "frame_id": frame,
            "n_orig": n0,
            "Z": Z,
            "intra_max": float(upper.max()),
            "intra_mean": float(upper.mean()),
            "intra_median": float(np.median(upper)),
            "drift_0_to_50": d_0_to_50,
            "n_post_50": max(int(n0 * 0.5), 64) if n0 > 0 else -1,
        })
    df = pd.DataFrame([{k: v for k, v in d.items() if k != "Z"} for d in fruit_data])
    print(f"[fruits] retained {len(df)}")

    # Inter-fruit spread per occlusion level
    Z_by_occ = {occ: [] for occ in OCC_LEVELS}
    for d in fruit_data:
        for i, occ in enumerate(OCC_LEVELS):
            Z_by_occ[occ].append(d["Z"][i])
    inter_stats = {}
    for occ in OCC_LEVELS:
        Z = np.stack(Z_by_occ[occ])
        # mean pairwise distance
        diff = Z[:, None, :] - Z[None, :, :]
        pdist = np.linalg.norm(diff, axis=2)
        upper = pdist[np.triu_indices(len(Z), k=1)]
        inter_stats[occ] = {
            "n_fruits": len(Z),
            "mean_pdist": float(upper.mean()),
            "median_pdist": float(np.median(upper)),
            "max_pdist": float(upper.max()),
        }

    print("\n=== inter-fruit spread (pairwise distance between different fruits, per occ level) ===")
    print(f"{'occ%':>4}  {'n_fruits':>8}  {'mean':>7}  {'median':>7}  {'max':>7}")
    for occ in OCC_LEVELS:
        s = inter_stats[occ]
        print(f"{occ:>4}  {s['n_fruits']:>8}  {s['mean_pdist']:>7.4f}  {s['median_pdist']:>7.4f}  {s['max_pdist']:>7.4f}")

    print("\n=== intra-fruit drift (same fruit moved across 6 occ levels) ===")
    print(f"  mean across fruits      : mean intra-fruit drift = {df['intra_mean'].mean():.4f}")
    print(f"  mean across fruits      : max  intra-fruit drift = {df['intra_max'].mean():.4f}")
    print(f"  mean drift 0→50         : {df['drift_0_to_50'].mean():.4f}")
    print(f"  median drift 0→50       : {df['drift_0_to_50'].median():.4f}")

    print(f"\n  ratio  (mean inter @ occ=0)  /  (mean intra)  =  "
          f"{inter_stats[0]['mean_pdist'] / df['intra_mean'].mean():.2f}")
    print(f"  if ≫ 1: shape dominates, point count irrelevant")
    print(f"  if  ~ 1: point count moves the latent comparably to fruit identity")

    # Stratify intra-drift by n_orig: do small-PCD fruits drift more under occlusion?
    print("\n=== intra-drift stratified by n_orig (original PCD point count) ===")
    bins = [0, 200, 500, 1000, 2000, 5000, 100000]
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    df["n_orig_bin"] = pd.cut(df["n_orig"], bins=bins, labels=labels, right=False)
    g = df.groupby("n_orig_bin", observed=True).agg(
        n_fruits=("fruit_id", "size"),
        mean_drift_0_to_50=("drift_0_to_50", "mean"),
        median_drift_0_to_50=("drift_0_to_50", "median"),
        max_drift=("intra_max", "mean"),
        n_post_50_min=("n_post_50", "min"),
        n_post_50_max=("n_post_50", "max"),
    ).reset_index()
    print(g.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    # Highlight outliers — fruits with biggest drift
    print("\n=== top 10 fruits by drift_0_to_50 (these should be small-PCD samples) ===")
    print(df.sort_values("drift_0_to_50", ascending=False)
            .head(10)[["fruit_id", "frame_id", "n_orig", "n_post_50",
                       "intra_mean", "drift_0_to_50"]]
            .to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    out = RUN_DIR.parent / "lowvol_analysis/per_sample_drift.csv"
    df.drop(columns=[]).to_csv(out, index=False)
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
