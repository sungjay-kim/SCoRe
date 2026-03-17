"""Investigate the 'fallback latent' hypothesis on a single arch (maxmean_masked_small static).

For each occlusion level we have one .npy file per (fruit_id, frame_id),
storing the encoder's 32-d latent for that input. We test:

  H1  At low post-occlusion point counts, the latents *cluster tightly*
      (pairwise distance shrinks).
  H2  The cluster centroid is roughly the same across low-n bins
      (encoder collapses toward a sample-independent fallback).
  H3  Decoding the centroid yields a mesh whose volume matches the
      empirically observed 'mean prediction' (~200 mL for this arch).

We print:
  - n_post bin × pairwise distance stats
  - centroid-to-centroid distance across bins
  - centroid volume after decoding

Run from project root with corepp env active.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("/home/mainuser/mount/sungjay/tcore")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "corepp"))

from corepp.utils import sdf2mesh_cuda, mesh_volume_ml  # noqa: E402
from corepp.sdfrenderer.grid import Grid3D  # noqa: E402
import corepp.deepsdf.deep_sdf as deep_sdf  # noqa: E402
import corepp.deepsdf.deep_sdf.workspace as ws  # noqa: E402
import corepp.networks.utils as net_utils  # noqa: E402

RUN_DIR = PROJECT_ROOT / (
    "corepp/experiments/kmelon_latent32_v3/_occlusion_sweep_v5_patch/"
    "static_maxmean_masked_small_with_latents"
)
LAT_DIR = RUN_DIR / "latents"
SWEEP_ROOT = RUN_DIR.parent
LOWVOL_CSV = SWEEP_ROOT / "lowvol_analysis/lowvol_long.csv"
DECODER_EXPERIMENT = "/home/mainuser/mount/sungjay/tcore/corepp/deepsdf/experiments/kmelon_latent32/"
DECODER_CHECKPOINT = "460"

OCC_LEVELS = [0, 10, 20, 30, 40, 50]


def load_latents() -> pd.DataFrame:
    """Return a long-format dataframe: (fruit_id, frame_id, occ_pct, latent_vec)."""
    rows = []
    for f in sorted(LAT_DIR.glob("*.npy")):
        stem = f.stem  # e.g., "P10_20260416_occ20"  or  "P10_20260416" (occ=0)
        if "_occ" in stem:
            base, occ_part = stem.rsplit("_occ", 1)
            occ = int(occ_part)
        else:
            base, occ = stem, 0
        # base = "{fruit_id}_{frame_id}"
        fruit_id, frame_id = base.rsplit("_", 1)
        z = np.load(f).astype(np.float64).reshape(-1)
        rows.append({"fruit_id": fruit_id, "frame_id": frame_id,
                     "occ_pct": occ, "z": z})
    return pd.DataFrame(rows)


def join_npost(latents: pd.DataFrame) -> pd.DataFrame:
    """Compute n_orig directly from each sample's orbbec PLY (point count on disk)."""
    import open3d as o3d
    DATA = PROJECT_ROOT / "corepp/data/kmelon_centered"
    pairs = latents[["fruit_id", "frame_id"]].drop_duplicates()
    cache = {}
    for fid, frm in zip(pairs["fruit_id"], pairs["frame_id"]):
        p = DATA / fid / "orbbec" / "pointclouds" / f"{frm}_preprocessed.ply"
        if not p.exists():
            cache[(fid, frm)] = -1
            continue
        cache[(fid, frm)] = int(np.asarray(o3d.io.read_point_cloud(str(p)).points).shape[0])
    out = latents.copy()
    out["n_orig"] = [cache.get((f, fr), -1) for f, fr in zip(out["fruit_id"], out["frame_id"])]
    missing = int((out["n_orig"] < 0).sum())
    if missing:
        print(f"[warn] {missing} samples have no PLY → dropping")
        out = out[out["n_orig"] > 0]
    out["n_post"] = np.maximum(
        (out["n_orig"] * (1 - out["occ_pct"] / 100.0)).astype(int),
        64,
    )
    print(f"[n_orig] computed for {len(out)} (fruit,frame,occ) rows; "
          f"unique fruits = {out['fruit_id'].nunique()}")
    return out


def bin_latents(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 5000, 100000]
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    df = df.copy()
    df["bin"] = pd.cut(df["n_post"], bins=bins, labels=labels, right=False)
    return df


def cluster_stats(group: pd.DataFrame) -> dict:
    Z = np.stack(group["z"].values)  # (n, 32)
    n = Z.shape[0]
    if n < 2:
        return {
            "n": n, "centroid": Z.mean(axis=0) if n else np.zeros(32),
            "centroid_norm": float(np.linalg.norm(Z.mean(axis=0))) if n else float("nan"),
            "mean_dist_to_centroid": float("nan"),
            "median_pairwise_dist": float("nan"),
            "max_pairwise_dist": float("nan"),
        }
    centroid = Z.mean(axis=0)
    dists_to_c = np.linalg.norm(Z - centroid, axis=1)
    diff = Z[:, None, :] - Z[None, :, :]
    pdists = np.linalg.norm(diff, axis=2)
    upper = pdists[np.triu_indices(n, k=1)]
    return {
        "n": n,
        "centroid": centroid,
        "centroid_norm": float(np.linalg.norm(centroid)),
        "mean_dist_to_centroid": float(dists_to_c.mean()),
        "median_pairwise_dist": float(np.median(upper)),
        "max_pairwise_dist": float(upper.max()),
    }


def load_decoder() -> tuple:
    deep_sdf.configure_logging.__defaults__  # ensure import side-effects loaded
    specs = ws.load_experiment_specifications(DECODER_EXPERIMENT)
    latent_size = specs["CodeLength"]
    arch_mod = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])
    decoder = arch_mod.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()
    state = net_utils.load_without_parallel(
        torch.load(f"{DECODER_EXPERIMENT}/ModelParameters/{DECODER_CHECKPOINT}.pth")
    )
    decoder.load_state_dict(state)
    decoder.eval()
    return decoder, latent_size


def decode_volume(decoder: torch.nn.Module, latent_vec: np.ndarray,
                  bbox: dict, grid_density: int = 50, threshold: float = 0.0) -> float:
    device = torch.device("cuda")
    grid = Grid3D(grid_density, device, torch.float32, bbox=bbox)
    z = torch.from_numpy(latent_vec.astype(np.float32)).to(device).unsqueeze(0)
    inp = torch.cat([z.expand(grid.points.size(0), -1), grid.points], dim=1)
    with torch.no_grad():
        sdf = decoder(inp)
    mesh = sdf2mesh_cuda(sdf, grid.points, t=threshold)
    if mesh is None:
        return float("nan")
    return float(mesh_volume_ml(mesh))


def main() -> None:
    latents = load_latents()
    print(f"[load] {len(latents)} latent files")
    latents = join_npost(latents)
    binned = bin_latents(latents)

    rows = []
    centroids_by_bin = {}
    for bin_label, sub in binned.groupby("bin", observed=True):
        if sub.empty:
            continue
        s = cluster_stats(sub)
        centroids_by_bin[bin_label] = s["centroid"]
        rows.append({
            "bin": bin_label,
            "n": s["n"],
            "centroid_norm": round(s["centroid_norm"], 4),
            "mean_dist_to_centroid": round(s["mean_dist_to_centroid"], 4)
                if not np.isnan(s["mean_dist_to_centroid"]) else None,
            "median_pairwise_dist": round(s["median_pairwise_dist"], 4)
                if not np.isnan(s["median_pairwise_dist"]) else None,
        })
    stats_df = pd.DataFrame(rows)
    print("\n=== cluster stats by n_post bin ===")
    print(stats_df.to_string(index=False))

    bins_sorted = list(stats_df["bin"])
    print("\n=== centroid-to-centroid distance (low bin → high bin) ===")
    print(f"   {'bin':>14}", end=" ")
    for b in bins_sorted:
        print(f"{str(b):>14}", end="")
    print()
    for b1 in bins_sorted:
        if b1 not in centroids_by_bin:
            continue
        print(f"   {str(b1):>14}", end=" ")
        for b2 in bins_sorted:
            if b2 not in centroids_by_bin:
                print(f"{'-':>14}", end="")
                continue
            d = float(np.linalg.norm(centroids_by_bin[b1] - centroids_by_bin[b2]))
            print(f"{d:>14.4f}", end="")
        print()

    # ---- decode centroids into volume ---------------------------------------
    print("\n[decode] loading DeepSDF decoder...")
    decoder, _ = load_decoder()
    bbox = {"xmin": -0.06, "xmax": 0.06,
            "ymin": -0.06, "ymax": 0.06,
            "zmin": -0.08, "zmax": 0.08}
    print("\n=== decoded volume of each bin's centroid ===")
    print(f"{'bin':>14}    {'n':>3}    {'centroid_norm':>14}    {'centroid → vol (mL)':>20}")
    for bin_label, centroid in centroids_by_bin.items():
        n = stats_df[stats_df["bin"] == bin_label]["n"].iloc[0]
        try:
            vol = decode_volume(decoder, centroid, bbox)
        except Exception as e:
            vol = float("nan")
            print(f"  decode failed for {bin_label}: {e}")
        print(f"{str(bin_label):>14}    {n:>3}    {float(np.linalg.norm(centroid)):>14.4f}    "
              f"{vol:>20.1f}")

    out_csv = SWEEP_ROOT / "lowvol_analysis/latent_collapse_stats.csv"
    stats_df.to_csv(out_csv, index=False)
    print(f"\n[wrote] {out_csv}")


if __name__ == "__main__":
    main()
