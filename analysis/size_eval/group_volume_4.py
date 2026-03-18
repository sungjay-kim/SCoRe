#!/usr/bin/env python3
"""
Group samples into 4 volume-based buckets using volume_ml from:
  size_evaluation/ellipsoid_fitting/<SAMPLE>/<SAMPLE>.json

Default grouping: quartiles (roughly equal counts), healthy-only.

Outputs:
  - JSON mapping: size_evaluation/ellipsoid_volume_stats/volume_groups_4.json
  - CSV table:    size_evaluation/ellipsoid_volume_stats/volume_groups_4.csv
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _is_unhealthy(sample_folder_name: str) -> bool:
    return "_unhealthy" in (sample_folder_name or "").lower()


def _safe_load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _pick_json_for_sample(sample_dir: str, sample_name: str) -> Optional[str]:
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


def _assign_group(v: float, q25: float, q50: float, q75: float) -> int:
    # Group 0: <= q25
    # Group 1: (q25, q50]
    # Group 2: (q50, q75]
    # Group 3: > q75
    if v <= q25:
        return 0
    if v <= q50:
        return 1
    if v <= q75:
        return 2
    return 3


def main() -> int:
    ap = argparse.ArgumentParser(description="Group samples into 4 size buckets from ellipsoid_fitting volume_ml.")
    ap.add_argument(
        "--input_dir",
        default="size_evaluation/ellipsoid_fitting",
        help="Directory containing per-sample folders with JSON outputs.",
    )
    ap.add_argument(
        "--out_dir",
        default="size_evaluation/ellipsoid_volume_stats",
        help="Directory to write grouping outputs.",
    )
    ap.add_argument(
        "--exclude_unhealthy",
        action="store_true",
        default=True,
        help="Exclude samples whose folder name contains '_unhealthy' (default: true).",
    )
    ap.add_argument(
        "--method",
        choices=["quartile"],
        default="quartile",
        help="Grouping method (currently only quartile).",
    )
    args = ap.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"input_dir does not exist: {input_dir}")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "volume_groups_4.json")
    out_csv = os.path.join(out_dir, "volume_groups_4.csv")

    rows: List[Tuple[str, float, str]] = []  # (sample, volume_ml, json_path)
    n_total = 0
    n_excluded = 0
    n_missing = 0

    for sample in sorted(os.listdir(input_dir)):
        sample_dir = os.path.join(input_dir, sample)
        if not os.path.isdir(sample_dir):
            continue
        n_total += 1
        if args.exclude_unhealthy and _is_unhealthy(sample):
            n_excluded += 1
            continue
        jp = _pick_json_for_sample(sample_dir, sample)
        if not jp:
            n_missing += 1
            continue
        data = _safe_load_json(jp)
        if not isinstance(data, dict) or "volume_ml" not in data:
            n_missing += 1
            continue
        try:
            v = float(data["volume_ml"])
        except Exception:
            n_missing += 1
            continue
        rows.append((sample, v, jp))

    if not rows:
        raise SystemExit("No valid samples found.")

    vols = np.asarray([v for _s, v, _jp in rows], dtype=float)
    q25, q50, q75 = [float(np.quantile(vols, q)) for q in (0.25, 0.50, 0.75)]

    mapping: Dict[str, Dict[str, object]] = {}
    counts = [0, 0, 0, 0]
    for s, v, jp in rows:
        g = _assign_group(v, q25, q50, q75)
        counts[g] += 1
        mapping[s] = {"group": int(g), "volume_ml": float(v), "json": os.path.abspath(jp)}

    payload = {
        "method": args.method,
        "exclude_unhealthy": bool(args.exclude_unhealthy),
        "input_dir": input_dir,
        "counts": {
            "sample_folders_total": int(n_total),
            "unhealthy_excluded": int(n_excluded),
            "missing_or_invalid": int(n_missing),
            "group0": int(counts[0]),
            "group1": int(counts[1]),
            "group2": int(counts[2]),
            "group3": int(counts[3]),
        },
        "boundaries_ml": {
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "group0": f"<= {q25:.6g}",
            "group1": f"({q25:.6g}, {q50:.6g}]",
            "group2": f"({q50:.6g}, {q75:.6g}]",
            "group3": f"> {q75:.6g}",
        },
        "samples": mapping,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # CSV table (stable order by sample)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("sample,group,volume_ml,json_path\n")
        for s in sorted(mapping.keys()):
            f.write(f"{s},{mapping[s]['group']},{mapping[s]['volume_ml']:.9f},{mapping[s]['json']}\n")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    print("Boundaries (mL):", f"q25={q25:.3f}", f"q50={q50:.3f}", f"q75={q75:.3f}")
    print("Counts:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


