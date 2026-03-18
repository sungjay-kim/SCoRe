#!/usr/bin/env python3
"""
Summarize ellipsoid-fitted volume statistics from:
  size_evaluation/ellipsoid_fitting/<SAMPLE>/*.json

Excludes unhealthy samples by default (folder name contains '_unhealthy').

It prefers the canonical JSON:
  <SAMPLE>/<SAMPLE>.json
If missing, it falls back to:
  - the only JSON in the folder, or
  - the most recently modified JSON in the folder.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

# headless-friendly backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _is_unhealthy(sample_folder_name: str) -> bool:
    return "_unhealthy" in (sample_folder_name or "").lower()


def _safe_load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _pick_json_for_sample(sample_dir: str, sample_name: str) -> Optional[str]:
    """
    Return a chosen JSON path for a sample folder.
    Preference:
      1) <sample_name>.json
      2) if only one json exists, that one
      3) newest mtime json
    """
    preferred = os.path.join(sample_dir, f"{sample_name}.json")
    if os.path.isfile(preferred):
        return preferred

    jsons = [
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.lower().endswith(".json") and os.path.isfile(os.path.join(sample_dir, f))
    ]
    if not jsons:
        return None
    if len(jsons) == 1:
        return jsons[0]
    jsons.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return jsons[0]


def _quantile(sorted_vals: List[float], q: float) -> float:
    """
    q in [0,1]. Linear interpolation between points (like numpy default).
    Assumes sorted_vals is non-empty and sorted ascending.
    """
    if not sorted_vals:
        raise ValueError("empty list")
    q = float(q)
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    n = len(sorted_vals)
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    t = pos - lo
    return float(sorted_vals[lo] * (1.0 - t) + sorted_vals[hi] * t)


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / max(1, len(vals)))


def _std(vals: List[float], mean: float) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return float(var**0.5)


@dataclass
class VolumeRow:
    sample: str
    volume_ml: float
    json_path: str
    extent0_mm: Optional[float] = None


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize volume_ml stats from ellipsoid fitting JSON outputs.")
    ap.add_argument(
        "--input_dir",
        default="size_evaluation/ellipsoid_fitting",
        help="Directory containing per-sample folders with JSON outputs.",
    )
    ap.add_argument(
        "--out_dir",
        default="size_evaluation/ellipsoid_volume_stats",
        help="Directory to write summary JSON/TXT and plots.",
    )
    ap.add_argument(
        "--out_json",
        default="",
        help="Output JSON path for summary stats.",
    )
    ap.add_argument(
        "--out_txt",
        default="",
        help="Output TXT path for human-readable summary.",
    )
    ap.add_argument(
        "--out_csv",
        default="",
        help="Optional output CSV path (sample,volume_ml,json_path).",
    )
    ap.add_argument(
        "--exclude_unhealthy",
        action="store_true",
        default=True,
        help="Exclude samples whose folder name contains '_unhealthy' (default: true).",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Include top/bottom K samples by volume in outputs.",
    )
    args = ap.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"input_dir does not exist: {input_dir}")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.abspath(args.out_json) if args.out_json else os.path.join(out_dir, "ellipsoid_volume_stats.json")
    out_txt = os.path.abspath(args.out_txt) if args.out_txt else os.path.join(out_dir, "ellipsoid_volume_stats.txt")
    out_csv = os.path.abspath(args.out_csv) if args.out_csv else os.path.join(out_dir, "ellipsoid_volume_table.csv")

    rows: List[VolumeRow] = []
    n_samples_total = 0
    n_unhealthy_excluded = 0
    n_missing_json = 0
    n_missing_volume = 0
    n_missing_extent0 = 0
    n_bad_json = 0

    for sample in sorted(os.listdir(input_dir)):
        sample_dir = os.path.join(input_dir, sample)
        if not os.path.isdir(sample_dir):
            continue
        n_samples_total += 1
        if args.exclude_unhealthy and _is_unhealthy(sample):
            n_unhealthy_excluded += 1
            continue

        json_path = _pick_json_for_sample(sample_dir, sample)
        if not json_path:
            n_missing_json += 1
            continue

        data = _safe_load_json(json_path)
        if not isinstance(data, dict):
            n_bad_json += 1
            continue

        vol = data.get("volume_ml", None)
        try:
            vol_f = float(vol)
        except Exception:
            n_missing_volume += 1
            continue

        # optional extent0 (mm)
        extent0 = None
        ext_mm = data.get("extents_mm", None)
        if isinstance(ext_mm, list) and len(ext_mm) >= 1:
            try:
                extent0 = float(ext_mm[0])
            except Exception:
                extent0 = None
        if extent0 is None:
            n_missing_extent0 += 1

        rows.append(VolumeRow(sample=sample, volume_ml=vol_f, json_path=json_path, extent0_mm=extent0))

    vols = sorted([r.volume_ml for r in rows])
    n = len(vols)
    if n == 0:
        raise SystemExit("No valid volume_ml values found after filtering.")

    mean_v = _mean(vols)
    std_v = _std(vols, mean_v)
    stats = {
        "input_dir": input_dir,
        "exclude_unhealthy": bool(args.exclude_unhealthy),
        "counts": {
            "sample_folders_total": int(n_samples_total),
            "unhealthy_excluded": int(n_unhealthy_excluded),
            "valid_samples": int(n),
            "missing_json": int(n_missing_json),
            "bad_json": int(n_bad_json),
            "missing_volume_ml": int(n_missing_volume),
            "missing_extent0_mm": int(n_missing_extent0),
        },
        "volume_ml": {
            "mean": mean_v,
            "std": std_v,
            "min": float(vols[0]),
            "p05": _quantile(vols, 0.05),
            "p25": _quantile(vols, 0.25),
            "median": _quantile(vols, 0.50),
            "p75": _quantile(vols, 0.75),
            "p95": _quantile(vols, 0.95),
            "max": float(vols[-1]),
        },
    }

    # top/bottom examples
    rows_sorted = sorted(rows, key=lambda r: r.volume_ml)
    k = max(1, int(args.top_k))
    stats["bottom_k"] = [
        {"sample": r.sample, "volume_ml": r.volume_ml, "json": r.json_path}
        for r in rows_sorted[:k]
    ]
    stats["top_k"] = [
        {"sample": r.sample, "volume_ml": r.volume_ml, "json": r.json_path}
        for r in rows_sorted[-k:][::-1]
    ]

    # write JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # write TXT
    lines = []
    lines.append("ELLIPSOID VOLUME STATISTICS (volume_ml)")
    lines.append("=" * 72)
    lines.append(f"Input dir: {input_dir}")
    lines.append(f"Exclude unhealthy: {bool(args.exclude_unhealthy)}")
    lines.append("")
    c = stats["counts"]
    lines.append("Counts:")
    lines.append(f"  sample_folders_total: {c['sample_folders_total']}")
    lines.append(f"  unhealthy_excluded:   {c['unhealthy_excluded']}")
    lines.append(f"  valid_samples:        {c['valid_samples']}")
    lines.append(f"  missing_json:         {c['missing_json']}")
    lines.append(f"  bad_json:             {c['bad_json']}")
    lines.append(f"  missing_volume_ml:    {c['missing_volume_ml']}")
    lines.append("")
    s = stats["volume_ml"]
    lines.append("Volume (mL):")
    lines.append(f"  mean:   {s['mean']:.3f}")
    lines.append(f"  std:    {s['std']:.3f}")
    lines.append(f"  min:    {s['min']:.3f}")
    lines.append(f"  p05:    {s['p05']:.3f}")
    lines.append(f"  p25:    {s['p25']:.3f}")
    lines.append(f"  median: {s['median']:.3f}")
    lines.append(f"  p75:    {s['p75']:.3f}")
    lines.append(f"  p95:    {s['p95']:.3f}")
    lines.append(f"  max:    {s['max']:.3f}")
    lines.append("")
    lines.append(f"Top {k}:")
    for r in stats["top_k"]:
        lines.append(f"  {r['sample']}: {r['volume_ml']:.3f} mL")
    lines.append("")
    lines.append(f"Bottom {k}:")
    for r in stats["bottom_k"]:
        lines.append(f"  {r['sample']}: {r['volume_ml']:.3f} mL")
    lines.append("")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # write CSV
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("sample,volume_ml,extent0_mm,json_path\n")
        for r in sorted(rows, key=lambda rr: rr.sample):
            e0 = "" if r.extent0_mm is None else f"{r.extent0_mm:.9f}"
            f.write(f"{r.sample},{r.volume_ml:.9f},{e0},{r.json_path}\n")

    # plots
    vols_np = np.asarray([r.volume_ml for r in rows], dtype=float)
    title_suffix = " (healthy-only)" if args.exclude_unhealthy else ""

    # Histogram
    hist_path = os.path.join(out_dir, "volume_histogram.png")
    plt.figure(figsize=(9, 5.5))
    plt.hist(vols_np, bins=30, edgecolor="black", alpha=0.85)
    plt.title(f"Ellipsoid volume distribution{title_suffix}")
    plt.xlabel("volume (mL)")
    plt.ylabel("count")
    plt.grid(True, alpha=0.25)
    # annotate key quantiles
    s = stats["volume_ml"]
    for key, color in [("median", "C3"), ("p25", "C1"), ("p75", "C1"), ("p05", "C0"), ("p95", "C0")]:
        plt.axvline(float(s[key]), color=color, linestyle="--", linewidth=1.2, alpha=0.9)
    plt.tight_layout()
    plt.savefig(hist_path, dpi=160)
    plt.close()

    # Boxplot
    box_path = os.path.join(out_dir, "volume_boxplot.png")
    plt.figure(figsize=(7.5, 4.8))
    plt.boxplot(vols_np, vert=False, showfliers=True)
    plt.title(f"Ellipsoid volume boxplot{title_suffix}")
    plt.xlabel("volume (mL)")
    plt.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(box_path, dpi=160)
    plt.close()

    # Scatter: volume vs extent0
    scat_path = os.path.join(out_dir, "volume_vs_extent0_scatter.png")
    x = np.asarray([r.extent0_mm for r in rows if r.extent0_mm is not None], dtype=float)
    y = np.asarray([r.volume_ml for r in rows if r.extent0_mm is not None], dtype=float)
    if x.size > 0:
        plt.figure(figsize=(7.8, 5.6))
        plt.scatter(x, y, s=18, alpha=0.65, edgecolors="none")
        plt.title(f"volume_ml vs extents_mm[0]{title_suffix}")
        plt.xlabel("Height (mm)")
        plt.ylabel("Volume (mL)")
        plt.grid(True, alpha=0.25)
        # simple linear trendline (for reference)
        try:
            m, b = np.polyfit(x, y, deg=1)
            xx = np.linspace(float(x.min()), float(x.max()), 200)
            plt.plot(xx, m * xx + b, color="C3", linewidth=2, alpha=0.9)
        except Exception:
            pass
        plt.tight_layout()
        plt.savefig(scat_path, dpi=160)
        plt.close()

    lines.append(f"Wrote JSON: {out_json}")
    lines.append(f"Wrote TXT:  {out_txt}")
    lines.append(f"Wrote CSV:  {out_csv}")
    lines.append(f"Wrote PNG:  {hist_path}")
    lines.append(f"Wrote PNG:  {box_path}")
    if x.size > 0:
        lines.append(f"Wrote PNG:  {scat_path}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


