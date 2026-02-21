#!/usr/bin/env python3
"""Unified eval post-processor for static and temporal test results.

Splits a per-frame inference CSV into:
  - last_date_results.csv + last_date_summary.json   (final frame per fruit, vs laser GT)
  - intermediate_results.csv + intermediate_summary.json
        (non-final frames; GT = ellipsoid(date) + offset, where
         offset = laser_GT - ellipsoid(last_measurement_date_<=_last_frame))

Usage:
    python eval_unified.py --results-csv <path-to-shape_completion_results.csv> \
                            [--output-dir <dir>] \
                            [--field-csv <field_size_data_v2.csv>]

By default, output dir = parent dir of the results CSV.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def ellipsoid_ml(h_mm: float, w_mm: float) -> float:
    """Prolate ellipsoid volume in mL given height/width in mm."""
    if pd.isna(h_mm) or pd.isna(w_mm) or h_mm <= 0 or w_mm <= 0:
        return float("nan")
    return (4.0 / 3.0) * np.pi * (h_mm / 2.0) * (w_mm / 2.0) ** 2 / 1000.0


def parse_date(frame_id) -> int:
    """Extract YYYYMMDD prefix from frame_id."""
    s = str(frame_id).split("_")[0]
    if len(s) == 8 and s.isdigit():
        return int(s)
    return -1


def summarize_volume(df: pd.DataFrame, pred_col: str, gt_col: str) -> dict:
    """Compute RMSE/MAE/Bias/MAPE/slope from pred vs gt columns."""
    d = df[[pred_col, gt_col]].dropna()
    d = d[(d[pred_col] > 0) & (d[gt_col] > 0)]
    if len(d) == 0:
        return {"n": 0}
    pred = d[pred_col].to_numpy(dtype=float)
    gt = d[gt_col].to_numpy(dtype=float)
    err = pred - gt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ape = np.abs(err) / gt
    mape = float(np.mean(ape) * 100)
    medape = float(np.median(ape) * 100)
    if len(d) >= 2:
        slope, intercept = np.polyfit(gt, pred, 1)
        slope, intercept = float(slope), float(intercept)
    else:
        slope, intercept = float("nan"), float("nan")
    return {
        "n": int(len(d)),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "mape_pct": mape,
        "medape_pct": medape,
        "slope": slope,
        "intercept": intercept,
        "pred_mean": float(pred.mean()),
        "gt_mean": float(gt.mean()),
    }


def summarize_shape(df: pd.DataFrame) -> dict:
    """Aggregate shape metrics (chamfer / F1 / precision / recall) when present."""
    out = {}
    for col in ["chamfer_distance", "auc_precision", "auc_recall", "auc_f1",
                "precision_5mm", "recall_5mm", "f1_5mm", "latent_norm"]:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        s = s[s != 0] if col != "latent_norm" else s  # filter sentinel zeros
        if len(s) == 0:
            continue
        entry = {
            "count": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
        }
        out[col] = entry
    return out


def compute_corrected_gt(per_frame: pd.DataFrame, field_df: pd.DataFrame) -> pd.DataFrame:
    """Return per_frame augmented with:
      - date (parsed int)
      - height_mm, width_mm  (joined from field measurements)
      - ellipsoid_ml          (raw ellipsoid for this frame's date)
      - last_meas_date        (latest measurement date <= last frame date)
      - last_ellip_ml         (ellipsoid at last_meas_date)
      - ratio                 (laser_GT_ml / last_ellip_ml) — multiplicative scale factor
      - offset_ml             (laser_GT_ml - last_ellip_ml) — kept for diagnostics
      - corrected_gt_ml       (ellipsoid_ml * ratio) — multiplicative correction
    """
    df = per_frame.copy()
    if "kmelon_id" in df.columns and "fruit_id" not in df.columns:
        df = df.rename(columns={"kmelon_id": "fruit_id"})
    df["date"] = df["frame_id"].apply(parse_date)
    df["laser_gt_ml"] = pd.to_numeric(df.get("gt_volume_ml"), errors="coerce")

    # Per-frame ellipsoid via merge
    field = field_df[["fruit_id", "date", "height", "width"]].copy()
    field["date"] = field["date"].astype(int)
    field["fruit_id"] = field["fruit_id"].astype(str)
    df["fruit_id"] = df["fruit_id"].astype(str)
    df = df.merge(field, on=["fruit_id", "date"], how="left")
    df = df.rename(columns={"height": "height_mm", "width": "width_mm"})
    df["ellipsoid_ml"] = df.apply(
        lambda r: ellipsoid_ml(r["height_mm"], r["width_mm"]), axis=1
    )

    # Per-fruit: last frame date and offset
    last_frame_dates = df.groupby("fruit_id")["date"].max().rename("last_frame_date")
    df = df.merge(last_frame_dates, on="fruit_id", how="left")

    # For each fruit, find latest measurement <= last_frame_date
    field_sorted = field.sort_values(["fruit_id", "date"])
    offsets = []
    for fid, last_d in last_frame_dates.items():
        meas = field_sorted[(field_sorted["fruit_id"] == fid) & (field_sorted["date"] <= last_d)]
        if len(meas) == 0:
            offsets.append({"fruit_id": fid, "last_meas_date": np.nan,
                            "last_ellip_ml": np.nan})
            continue
        row = meas.iloc[-1]
        ell = ellipsoid_ml(row["height"], row["width"])
        offsets.append({"fruit_id": fid, "last_meas_date": int(row["date"]),
                        "last_ellip_ml": ell})
    off_df = pd.DataFrame(offsets)
    df = df.merge(off_df, on="fruit_id", how="left")

    df["offset_ml"] = df["laser_gt_ml"] - df["last_ellip_ml"]  # kept for diagnostics
    df["ratio"] = df["laser_gt_ml"] / df["last_ellip_ml"]
    # Multiplicative correction: corrected_gt = ellipsoid_at_t * (laser_GT / ellipsoid_at_last)
    # This preserves growth proportions across stages, more biologically plausible than additive.
    df["corrected_gt_ml"] = df["ellipsoid_ml"] * df["ratio"]

    return df


def plot_growth_curves(aug: pd.DataFrame, output_path: Path, max_fruits: int = None, cols: int = 6,
                       static_pred_col: str = None) -> bool:
    """Per-fruit growth curves plot.

    For each fruit:
      - red solid + circles: predicted (mesh_volume_ml)
      - red dotted + circles: static FT predicted (when ``static_pred_col`` provided)
      - blue solid + squares: corrected ellipsoid (ellipsoid + offset)
      - blue dotted + squares: raw ellipsoid
      - green star: laser GT (constant horizontal level)
    """
    if not _HAS_MPL:
        print("[plot] matplotlib unavailable, skipping growth curves")
        return False

    df = aug.copy()
    # Convert date to datetime for nicer x-axis
    def date_to_dt(d):
        try:
            s = str(int(d))
            return pd.Timestamp(year=int(s[:4]), month=int(s[4:6]), day=int(s[6:8]))
        except Exception:
            return pd.NaT
    df["dt"] = df["date"].apply(date_to_dt)
    df = df.dropna(subset=["dt"])

    # Preserve test-split order (first appearance in CSV)
    fruit_order = list(dict.fromkeys(df["fruit_id"].tolist()))
    has_ellip = df.dropna(subset=["ellipsoid_ml"]).groupby("fruit_id").size()
    eligible_set = set(has_ellip[has_ellip >= 3].index.tolist())
    eligible = [f for f in fruit_order if f in eligible_set]
    if not eligible:
        print("[plot] No fruits with >=3 ellipsoid points, skipping")
        return False

    # Subset only if explicitly capped
    if max_fruits is not None and len(eligible) > max_fruits:
        idxs = np.linspace(0, len(eligible) - 1, max_fruits).astype(int)
        sel = [eligible[i] for i in idxs]
    else:
        sel = eligible

    rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.6 * rows), squeeze=False)

    for i, fid in enumerate(sel):
        ax = axes[i // cols][i % cols]
        sub = df[df["fruit_id"] == fid].sort_values("dt")
        # predicted (temporal/main model)
        ax.plot(sub["dt"], sub["mesh_volume_ml"], "r-o",
                markersize=3, linewidth=1.2, label="pred")
        # static FT pred overlay (red dotted) when available
        if static_pred_col and static_pred_col in sub.columns:
            s_static = sub.dropna(subset=[static_pred_col])
            if len(s_static) > 0:
                ax.plot(s_static["dt"], s_static[static_pred_col], "r:o",
                        markersize=3, linewidth=1.0, alpha=0.7, label="pred (static)")
        # corrected ellipsoid (solid)
        s_corr = sub.dropna(subset=["corrected_gt_ml"])
        if len(s_corr) > 0:
            ax.plot(s_corr["dt"], s_corr["corrected_gt_ml"], "b-s",
                    markersize=3, linewidth=1.2, label="ellip+offset")
        # raw ellipsoid (dotted)
        s_raw = sub.dropna(subset=["ellipsoid_ml"])
        if len(s_raw) > 0:
            ax.plot(s_raw["dt"], s_raw["ellipsoid_ml"], "b:s",
                    markersize=3, linewidth=1.0, alpha=0.7, label="ellip raw")
        # laser GT — star at last date
        last_lg = sub.dropna(subset=["laser_gt_ml"])
        if len(last_lg) > 0:
            lg_val = last_lg["laser_gt_ml"].iloc[-1]
            lg_dt = sub["dt"].iloc[-1]
            ax.plot([lg_dt], [lg_val], marker="*", color="green",
                    markersize=12, linestyle="None", label=f"laser GT ({lg_val:.0f})")

        ax.set_title(fid, fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        if i // cols == rows - 1:
            ax.set_xlabel("date", fontsize=8)
        if i % cols == 0:
            ax.set_ylabel("vol (ml)", fontsize=8)
        if i == 0:
            ax.legend(fontsize=6, loc="upper left")

    # Hide unused axes
    for j in range(len(sel), rows * cols):
        axes[j // cols][j % cols].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved growth curves: {output_path}  ({len(sel)} fruits)")
    return True


def split_last_vs_intermediate(df: pd.DataFrame) -> tuple:
    """Mark last frame per fruit (using max frame_idx if available, else max date)."""
    df = df.copy()
    if "frame_idx" in df.columns and "seq_len" in df.columns:
        df["is_last_frame"] = (df["frame_idx"] == df["seq_len"] - 1)
    else:
        last_date = df.groupby("fruit_id")["date"].transform("max")
        df["is_last_frame"] = (df["date"] == last_date)
    last = df[df["is_last_frame"]].copy()
    inter = df[~df["is_last_frame"]].copy()
    return last, inter


def _summarize_frame_group(last: pd.DataFrame, inter: pd.DataFrame) -> dict:
    """Build the {last_date, intermediate} summary dict for one frame-group split.

    Locked eval protocol (2026-05-08): no >50ml volume filter.
    All valid GT rows included.
    """
    last_f = last
    inter_f = inter
    inter_corr = inter_f
    inter_raw = inter_f
    return {
        "filter_note": "no fruit-volume filter (all valid GT rows)",
        "last_date": {
            "n_fruits": int(last_f["fruit_id"].nunique()),
            "volume_vs_laser": summarize_volume(last_f, "mesh_volume_ml", "laser_gt_ml"),
            "shape_metrics": summarize_shape(last_f),
        },
        "intermediate": {
            "n_rows_total": int(len(inter_f)),
            "n_rows_with_corrected_gt": int(len(inter_corr)),
            "volume_vs_corrected_gt": summarize_volume(inter_corr, "mesh_volume_ml", "corrected_gt_ml") if "corrected_gt_ml" in inter_corr.columns else {"n": 0},
            "volume_vs_raw_ellipsoid": summarize_volume(inter_raw, "mesh_volume_ml", "ellipsoid_ml") if "ellipsoid_ml" in inter_raw.columns else {"n": 0},
            "shape_metrics": summarize_shape(inter_f),
            "offset_stats": {
                "n_fruits_with_offset": int(inter_f["offset_ml"].dropna().groupby(inter_f["fruit_id"]).first().shape[0])
                    if "offset_ml" in inter_f.columns and len(inter_f) > 0 else 0,
                "offset_mean": float(inter_f.dropna(subset=["offset_ml"]).groupby("fruit_id")["offset_ml"].first().mean())
                    if "offset_ml" in inter_f.columns and int(inter_f["offset_ml"].notna().sum()) > 0 else float("nan"),
                "offset_median": float(inter_f.dropna(subset=["offset_ml"]).groupby("fruit_id")["offset_ml"].first().median())
                    if "offset_ml" in inter_f.columns and int(inter_f["offset_ml"].notna().sum()) > 0 else float("nan"),
            },
        },
    }


def write_combined_eval_sweep(sweep_dir, field_csv, output_dir,
                              occlusion_levels=None, plot_growth: bool = True,
                              static_eval_csv=None) -> dict:
    """Aggregate per-occlusion-level inference CSVs into a single combined eval.

    Reads ``shape_completion_results[_occN].csv`` for each occlusion level under
    ``sweep_dir`` and writes:
      - ``<output_dir>/evaluation_results.csv`` — long-format CSV with
        ``occlusion_pct`` and ``frame_type`` (last_date / intermediate) columns
      - ``<output_dir>/evaluation_summary.json`` — nested dict
        ``{occlusion_levels, by_occlusion: {occ_pct: {last_date, intermediate}}}``
      - ``<output_dir>/growth_curves.png`` (occ=0 only, if ``plot_growth=True``)

    Discovers occlusion levels automatically by globbing if ``occlusion_levels`` is None.
    """
    from pathlib import Path as _Path
    import re as _re
    sweep_dir = _Path(sweep_dir)
    out_dir = _Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    field = pd.read_csv(field_csv)

    if occlusion_levels is None:
        levels = set()
        for f in sweep_dir.glob("shape_completion_results*.csv"):
            m = _re.match(r"shape_completion_results(?:_occ(\d+))?\.csv", f.name)
            if m:
                levels.add(int(m.group(1)) if m.group(1) else 0)
        occlusion_levels = sorted(levels)

    all_rows = []
    by_occlusion = {}
    aug_for_plot = None  # use occ=0 augmented df for growth curves
    for occ in occlusion_levels:
        suf = f"_occ{occ}" if occ > 0 else ""
        csv_path = sweep_dir / f"shape_completion_results{suf}.csv"
        if not csv_path.exists():
            print(f"[unified-eval] Missing {csv_path.name}, skipping occ={occ}")
            continue
        res = pd.read_csv(csv_path)
        aug = compute_corrected_gt(res, field)
        if occ == 0:
            aug_for_plot = aug
        last, inter = split_last_vs_intermediate(aug)
        last = last.copy(); last["frame_type"] = "last_date"
        inter = inter.copy(); inter["frame_type"] = "intermediate"
        combined = pd.concat([last, inter], ignore_index=True)
        combined["occlusion_pct"] = int(occ)
        all_rows.append(combined)
        by_occlusion[str(occ)] = _summarize_frame_group(last, inter)

    if not all_rows:
        raise RuntimeError(f"No shape_completion_results CSVs found in {sweep_dir}")

    full = pd.concat(all_rows, ignore_index=True)
    front = ["occlusion_pct", "frame_type"]
    cols = front + [c for c in full.columns if c not in front]
    full = full[cols]
    full.to_csv(out_dir / "evaluation_results.csv", index=False)

    summary = {
        "source_dir": str(sweep_dir),
        "occlusion_levels": occlusion_levels,
        "by_occlusion": by_occlusion,
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))

    if plot_growth and aug_for_plot is not None:
        # Plot policy (2026-05-08): hybrid + monotonic REMOVED.
        # Solid red = raw temporal prediction (mesh_volume_ml as-is).
        # Dotted red = static FT prediction overlay (when static_eval_csv provided, reference only).
        static_pred_col = None
        if static_eval_csv is not None:
            try:
                static_csv_path = _Path(static_eval_csv)
                if static_csv_path.exists():
                    sdf = pd.read_csv(static_csv_path)
                    if "occlusion_pct" in sdf.columns:
                        sdf = sdf[sdf["occlusion_pct"] == 0]
                    if {"fruit_id", "frame_id", "mesh_volume_ml"}.issubset(sdf.columns):
                        s_map = sdf.set_index(["fruit_id", "frame_id"])["mesh_volume_ml"]
                        aug_for_plot = aug_for_plot.copy()
                        aug_for_plot["static_pred_ml"] = aug_for_plot.apply(
                            lambda r: s_map.get((r["fruit_id"], r["frame_id"]), float("nan")), axis=1
                        )
                        static_pred_col = "static_pred_ml"
                        print(f"[unified-eval] static overlay: loaded {len(s_map)} rows (no hybrid/mono)")
                else:
                    print(f"[unified-eval] static overlay: file not found {static_csv_path}")
            except Exception as e:
                print(f"[unified-eval] static overlay skipped: {e}")
        plot_growth_curves(aug_for_plot, out_dir / "growth_curves.png", static_pred_col=static_pred_col)

    return summary


def write_combined_eval(results_csv, field_csv, output_dir, plot_growth: bool = True) -> dict:
    """Run unified evaluation and write a single combined CSV + JSON.

    Outputs (in ``output_dir``):
      - ``evaluation_results.csv`` — all rows with a ``frame_type`` column
        ('last_date' | 'intermediate')
      - ``evaluation_summary.json`` — single dict with both sub-summaries
      - ``growth_curves.png`` (when ``plot_growth=True``)
    """
    from pathlib import Path as _Path
    res_csv = _Path(results_csv)
    out_dir = _Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = pd.read_csv(res_csv)
    field = pd.read_csv(field_csv)

    aug = compute_corrected_gt(res, field)
    last, inter = split_last_vs_intermediate(aug)

    last = last.copy(); last["frame_type"] = "last_date"
    inter = inter.copy(); inter["frame_type"] = "intermediate"
    combined = pd.concat([last, inter], ignore_index=True)
    if "frame_type" in combined.columns:
        cols = ["frame_type"] + [c for c in combined.columns if c != "frame_type"]
        combined = combined[cols]
    combined.to_csv(out_dir / "evaluation_results.csv", index=False)

    summary = {
        "source_csv": str(res_csv),
        "last_date": {
            "n_fruits": int(last["fruit_id"].nunique()),
            "volume_vs_laser": summarize_volume(last, "mesh_volume_ml", "laser_gt_ml"),
            "shape_metrics": summarize_shape(last),
        },
        "intermediate": {
            "n_rows_total": int(len(inter)),
            "n_rows_with_corrected_gt": int(inter["corrected_gt_ml"].notna().sum()),
            "volume_vs_corrected_gt": summarize_volume(inter, "mesh_volume_ml", "corrected_gt_ml"),
            "volume_vs_raw_ellipsoid": summarize_volume(inter, "mesh_volume_ml", "ellipsoid_ml"),
            "shape_metrics": summarize_shape(inter),
            "offset_stats": {
                "n_fruits_with_offset": int(inter["offset_ml"].dropna().groupby(inter["fruit_id"]).first().shape[0]),
                "offset_mean": float(inter.dropna(subset=["offset_ml"]).groupby("fruit_id")["offset_ml"].first().mean())
                    if int(inter["offset_ml"].notna().sum()) > 0 else float("nan"),
                "offset_median": float(inter.dropna(subset=["offset_ml"]).groupby("fruit_id")["offset_ml"].first().median())
                    if int(inter["offset_ml"].notna().sum()) > 0 else float("nan"),
            },
        },
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))

    if plot_growth:
        plot_growth_curves(aug, out_dir / "growth_curves.png")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-csv", required=True,
                    help="Per-frame inference CSV (shape_completion_results.csv)")
    ap.add_argument("--output-dir", default=None,
                    help="Defaults to parent dir of results CSV")
    ap.add_argument("--field-csv",
                    default="/home/mainuser/mount/sungjay/tcore/corepp/data/kmelon_centered/field_size_data_v2.csv",
                    help="Per-date field measurement CSV (with fruit_id/date/height/width)")
    args = ap.parse_args()

    res_csv = Path(args.results_csv)
    if not res_csv.exists():
        print(f"ERROR: results CSV not found: {res_csv}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else res_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = write_combined_eval(
        results_csv=res_csv,
        field_csv=args.field_csv,
        output_dir=out_dir,
        plot_growth=True,
    )

    # === Print human-readable summary ===
    def fmt(v, key, w=7, prec=2):
        x = v.get(key)
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "  N/A".rjust(w)
        return f"{x:>{w}.{prec}f}"

    print(f"\n=== unified eval: {res_csv.name} ===")
    print(f"Output dir: {out_dir}")
    last_summary = summary["last_date"]
    inter_summary = summary["intermediate"]
    v = last_summary["volume_vs_laser"]
    print(f"\n[last date]  vs laser GT   n={v.get('n', 0):>4}  "
          f"RMSE={fmt(v,'rmse',6,1)}  MAE={fmt(v,'mae',6,1)}  "
          f"Bias={fmt(v,'bias',7,1)}  MAPE={fmt(v,'mape_pct',5,1)}%  medAPE={fmt(v,'medape_pct',5,1)}%  slope={fmt(v,'slope',5,3)}")
    sm = last_summary["shape_metrics"]
    if "f1_5mm" in sm:
        print(f"             F1@5={sm['f1_5mm']['mean']:.1f}  CD_mm={sm['chamfer_distance']['mean']*1000:.2f}  "
              f"P@5={sm.get('precision_5mm',{}).get('mean',float('nan')):.1f}  "
              f"R@5={sm.get('recall_5mm',{}).get('mean',float('nan')):.1f}")

    v = inter_summary["volume_vs_corrected_gt"]
    print(f"\n[intermediate]  vs corrected GT  (multiplicative: ellipsoid * (laser_GT / last_ellipsoid))")
    print(f"             n={v.get('n', 0):>4}  RMSE={fmt(v,'rmse',6,1)}  MAE={fmt(v,'mae',6,1)}  "
          f"Bias={fmt(v,'bias',7,1)}  MAPE={fmt(v,'mape_pct',5,1)}%  medAPE={fmt(v,'medape_pct',5,1)}%  slope={fmt(v,'slope',5,3)}")
    v_raw = inter_summary["volume_vs_raw_ellipsoid"]
    print(f"             (raw ellipsoid for ref) n={v_raw.get('n', 0):>4}  "
          f"RMSE={fmt(v_raw,'rmse',6,1)}  Bias={fmt(v_raw,'bias',7,1)}  "
          f"MAPE={fmt(v_raw,'mape_pct',5,1)}%  medAPE={fmt(v_raw,'medape_pct',5,1)}%")
    offs = inter_summary["offset_stats"]
    print(f"             offset over fruits: n={offs['n_fruits_with_offset']}  "
          f"mean={offs['offset_mean']:.1f}  median={offs['offset_median']:.1f}")

    print(f"\nSaved:")
    print(f"  {out_dir/'evaluation_results.csv'}")
    print(f"  {out_dir/'evaluation_summary.json'}")
    print(f"  {out_dir/'growth_curves.png'}")


if __name__ == "__main__":
    main()
