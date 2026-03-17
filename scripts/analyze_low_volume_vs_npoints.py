"""Analyze prediction error vs input point count for low-volume samples (GT ≤ 200 ml).

Question: when occlusion drops the input point count to very low values, does
the model regress to the population mean and *artificially* improve metrics?

For each sample × occlusion-level cell we compute:
  - n_orig:        original point count of the input PCD on disk
  - n_post:        n_orig * (1 - occ_pct/100), clipped at min_points = 64
                   (matches `apply_test_occlusion` semantics for `patch` mode)
  - gt_ref_ml:     laser_gt_ml for last-date frames, corrected_gt_ml otherwise
  - vol_abs_err:   |mesh_volume_ml - gt_ref_ml|
  - vol_pct_err:   100 * vol_abs_err / gt_ref_ml

We restrict to gt_ref_ml ≤ 200 ml (small fruits / early growth stages) and plot:
  abs_err vs n_post   per (arch, mode), with occlusion level color-coded.

Outputs land under:
  _occlusion_sweep_v5_patch/lowvol_analysis/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import open3d as o3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/home/mainuser/mount/sungjay/tcore")
SWEEP_ROOT = PROJECT_ROOT / "corepp/experiments/kmelon_latent32_v3/_occlusion_sweep_v5_patch"
DATA_DIR = PROJECT_ROOT / "corepp/data/kmelon_centered"
FIELD_CSV = DATA_DIR / "field_size_data_v2.csv"
OUT_DIR = SWEEP_ROOT / "lowvol_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Make eval_unified importable
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "corepp"))
from scripts.eval_unified import compute_corrected_gt  # noqa: E402

VOL_THRESHOLD = 200.0
MIN_POINTS = 64  # matches apply_test_occlusion default in test scripts
ARCHES = ["max_small", "max_large", "maxmean_masked_small", "maxmean_masked_large"]
MODES = ["static", "temporal"]


def count_points(fruit_id: str, frame_id: str) -> int:
    """Real (non-padded) point count of the orbbec preprocessed PCD on disk."""
    p = DATA_DIR / fruit_id / "orbbec" / "pointclouds" / f"{frame_id}_preprocessed.ply"
    if not p.exists():
        return -1
    pcd = o3d.io.read_point_cloud(str(p))
    return int(np.asarray(pcd.points).shape[0])


def build_npoints_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute one row per unique (fruit_id, frame_id) sample."""
    pairs = df[["kmelon_id", "frame_id"]].drop_duplicates()
    rows = []
    for fid, frame in zip(pairs["kmelon_id"], pairs["frame_id"]):
        n = count_points(str(fid), str(frame))
        rows.append({"kmelon_id": fid, "frame_id": frame, "n_orig": n})
    out = pd.DataFrame(rows)
    print(f"[npts] computed n_orig for {len(out)} samples "
          f"(missing={int((out['n_orig'] < 0).sum())})")
    return out


def main() -> None:
    long_csv = SWEEP_ROOT / "sweep_long.csv"
    df = pd.read_csv(long_csv)
    print(f"[load] {long_csv} → {len(df)} rows")

    # Normalize frame_id strings (csv may have mixed types)
    df["frame_id"] = df["frame_id"].astype(str)

    # ---- corrected_gt for intermediate frames -------------------------------
    field_df = pd.read_csv(FIELD_CSV)
    # compute_corrected_gt expects a per_frame frame with one row per
    # (fruit_id, frame_id). We compute it once on temporal occ=0 (which has
    # all frames) and join the result back to all rows.
    base = df[(df["mode"] == "temporal") & (df["occ_pct"] == 0)].copy()
    # rename for compute_corrected_gt API
    aug = compute_corrected_gt(base, field_df)
    gt_lookup = aug[["fruit_id", "frame_id", "laser_gt_ml", "corrected_gt_ml",
                     "ellipsoid_ml", "last_frame_date", "date"]].copy()
    gt_lookup["frame_id"] = gt_lookup["frame_id"].astype(str)
    gt_lookup["fruit_id"] = gt_lookup["fruit_id"].astype(str)
    print(f"[gt] corrected_gt available for "
          f"{int(gt_lookup['corrected_gt_ml'].notna().sum())} / {len(gt_lookup)} frames")

    df["fruit_id"] = df["kmelon_id"].astype(str)
    df = df.merge(gt_lookup, on=["fruit_id", "frame_id"], how="left")

    # ---- choose gt_ref_ml ---------------------------------------------------
    is_last = df["frame_idx"] == (df["seq_len"] - 1)
    df["is_last_frame"] = is_last
    df["gt_ref_ml"] = np.where(is_last, df["laser_gt_ml"], df["corrected_gt_ml"])
    df["gt_source"] = np.where(is_last, "laser", "corrected_ellipsoid")
    # static rows (which lack frame_idx/seq_len because they evaluated only
    # last frame) — gt_ref_ml = laser
    static_mask = df["mode"] == "static"
    df.loc[static_mask, "gt_ref_ml"] = df.loc[static_mask, "laser_gt_ml"]
    df.loc[static_mask, "gt_source"] = "laser"
    df.loc[static_mask, "is_last_frame"] = True

    # ---- input point counts -------------------------------------------------
    npts = build_npoints_table(df)
    npts["frame_id"] = npts["frame_id"].astype(str)
    npts["fruit_id"] = npts["kmelon_id"].astype(str)
    df = df.merge(npts.drop(columns=["kmelon_id"]), on=["fruit_id", "frame_id"], how="left")

    # post-occlusion point count, matching apply_test_occlusion semantics
    df["n_post"] = np.maximum(
        (df["n_orig"] * (1.0 - df["occ_pct"] / 100.0)).astype(int),
        MIN_POINTS,
    )

    # ---- error metrics ------------------------------------------------------
    df["mesh_volume_ml"] = pd.to_numeric(df["mesh_volume_ml"], errors="coerce")
    df["vol_abs_err"] = (df["mesh_volume_ml"] - df["gt_ref_ml"]).abs()
    df["vol_pct_err"] = 100.0 * df["vol_abs_err"] / df["gt_ref_ml"]
    df["vol_signed_err"] = df["mesh_volume_ml"] - df["gt_ref_ml"]

    # ---- low-volume filter --------------------------------------------------
    valid = (
        df["gt_ref_ml"].notna()
        & (df["gt_ref_ml"] > 0)
        & (df["gt_ref_ml"] <= VOL_THRESHOLD)
        & df["mesh_volume_ml"].notna()
        & (df["mesh_volume_ml"] > 0)
        & (df["n_orig"] > 0)
    )
    low = df[valid].copy()
    print(f"[filter] gt_ref_ml ≤ {VOL_THRESHOLD} ml & valid mesh: "
          f"{len(low)} / {len(df)} rows ({len(df) - len(low)} dropped)")
    print(f"  by source: {low['gt_source'].value_counts().to_dict()}")
    print(f"  by mode  : {low['mode'].value_counts().to_dict()}")

    out_csv = OUT_DIR / "lowvol_long.csv"
    low.to_csv(out_csv, index=False)
    print(f"[wrote] {out_csv}")

    # ---- summary table: bin n_post and report mean abs err ------------------
    bins = [0, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 5000, 10000, 100000]
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    low["n_post_bin"] = pd.cut(low["n_post"], bins=bins, labels=labels, right=False)

    bin_summary = (
        low.groupby(["arch", "mode", "n_post_bin"], observed=True)
           .agg(n=("vol_abs_err", "size"),
                abs_err_mean=("vol_abs_err", "mean"),
                abs_err_median=("vol_abs_err", "median"),
                pct_err_mean=("vol_pct_err", "mean"),
                signed_err_mean=("vol_signed_err", "mean"),
                gt_mean=("gt_ref_ml", "mean"),
                pred_mean=("mesh_volume_ml", "mean"))
           .reset_index()
    )
    bin_csv = OUT_DIR / "lowvol_npoints_bins.csv"
    bin_summary.to_csv(bin_csv, index=False)
    print(f"[wrote] {bin_csv}")

    # ---- plots --------------------------------------------------------------
    # Plot 1: scatter abs_err vs n_post, faceted by arch, color by mode
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, arch in zip(axes.flat, ARCHES):
        sub = low[low["arch"] == arch]
        for mode, c in zip(MODES, ["#d62728", "#1f77b4"]):
            s = sub[sub["mode"] == mode]
            ax.scatter(s["n_post"], s["vol_abs_err"], s=10, alpha=0.35,
                       color=c, label=mode)
        ax.set_xscale("log")
        ax.set_title(arch)
        ax.set_xlabel("post-occlusion point count (log)")
        ax.set_ylabel("|vol err| (mL)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(f"Volume error vs input point count  (gt_ref ≤ {VOL_THRESHOLD} ml)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_abserr_vs_npoints.png", dpi=150)
    plt.close(fig)

    # Plot 2: median abs_err per n_post bin (line per arch×mode)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, arch in zip(axes.flat, ARCHES):
        sub = bin_summary[bin_summary["arch"] == arch]
        for mode, c in zip(MODES, ["#d62728", "#1f77b4"]):
            s = sub[sub["mode"] == mode].dropna(subset=["abs_err_median"])
            if s.empty:
                continue
            x = np.arange(len(s))
            ax.plot(x, s["abs_err_median"], "o-", color=c, label=f"{mode} median")
            ax.plot(x, s["abs_err_mean"], "x--", color=c, alpha=0.6, label=f"{mode} mean")
            ax.set_xticks(x)
            ax.set_xticklabels(s["n_post_bin"].astype(str), rotation=45, ha="right")
        ax.set_title(arch)
        ax.set_ylabel("|vol err| (mL)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"|vol err| vs post-occlusion point count bin  (gt_ref ≤ {VOL_THRESHOLD} ml)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "binned_abserr_vs_npoints.png", dpi=150)
    plt.close(fig)

    # Plot 3: signed error — does the model regress to the mean (positive bias) at low n?
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, arch in zip(axes.flat, ARCHES):
        sub = bin_summary[bin_summary["arch"] == arch]
        for mode, c in zip(MODES, ["#d62728", "#1f77b4"]):
            s = sub[sub["mode"] == mode].dropna(subset=["signed_err_mean"])
            if s.empty:
                continue
            x = np.arange(len(s))
            ax.plot(x, s["signed_err_mean"], "o-", color=c, label=f"{mode}")
            ax.set_xticks(x)
            ax.set_xticklabels(s["n_post_bin"].astype(str), rotation=45, ha="right")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(arch)
        ax.set_ylabel("mean signed err (pred − gt) [mL]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(
        f"Signed volume bias vs input point count  (gt_ref ≤ {VOL_THRESHOLD} ml)\n"
        "positive bias under low n suggests regression-to-population-mean"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "signed_bias_vs_npoints.png", dpi=150)
    plt.close(fig)

    # ---- print quick-look numbers ------------------------------------------
    print("\n[summary] mean prediction & error per n_post bin (last-frame only, "
          f"gt_ref ≤ {VOL_THRESHOLD} ml):")
    overall = bin_summary[bin_summary["mode"] == "temporal"]
    cols = ["arch", "n_post_bin", "n", "gt_mean", "pred_mean",
            "signed_err_mean", "abs_err_mean", "pct_err_mean"]
    print(overall[cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()
