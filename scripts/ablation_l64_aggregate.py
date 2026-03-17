#!/usr/bin/env python3
"""Aggregate the 5-condition Kalman-vs-AsymSizeGate ablation table.

Reads 5 evaluation_summary.json files (A0, A2, A_iso, A_aniso_sym, A5), pulls the
metrics under by_occlusion["0"], and prints a markdown table plus the 4 difference
rows. CD is converted to mm by multiplying chamfer_distance.mean by 1000.

Usage:
    python ablation_l64_aggregate.py \\
        --A0  <path>/eval/evaluation_summary.json \\
        --A2  <path>/eval/evaluation_summary.json \\
        --Aiso <path>/eval/evaluation_summary.json \\
        --Aaniso_sym <path>/eval/evaluation_summary.json \\
        --A5  <path>/eval/evaluation_summary.json
"""
import argparse
import json
import sys
from pathlib import Path

METRIC_ORDER = [
    "last_MAPE", "last_medAPE", "last_RMSE", "last_bias",
    "CD_mm", "F1_5mm",
    "inter_MAPE", "inter_RMSE",
]

ROW_LABELS = {
    "A0":         "A0 (static, no temporal)",
    "A2":         "A2 (Kalman, no gating)",
    "Aiso":       "A_iso (isotropic Mahal.)",
    "Aaniso_sym": "A_aniso_sym (aniso, sym thr, no DZ)",
    "A5":         "A5 (aniso + asym + DZ — full)",
}


def extract(path):
    with open(path) as f:
        d = json.load(f)
    occ0 = d["by_occlusion"]["0"]
    last = occ0["last_date"]
    inter = occ0["intermediate"]
    vol_l = last["volume_vs_laser"]
    shape = last["shape_metrics"]
    vol_i = inter["volume_vs_corrected_gt"]
    return {
        "last_MAPE":   vol_l["mape_pct"],
        "last_medAPE": vol_l["medape_pct"],
        "last_RMSE":   vol_l["rmse"],
        "last_bias":   vol_l["bias"],
        "CD_mm":       shape["chamfer_distance"]["mean"] * 1000.0,
        "F1_5mm":      shape["f1_5mm"]["mean"],
        "inter_MAPE":  vol_i["mape_pct"],
        "inter_RMSE":  vol_i["rmse"],
        "n_fruits":    last["n_fruits"],
    }


def fmt(v, prec=2):
    return f"{v:.{prec}f}"


def print_table(rows):
    cols = METRIC_ORDER
    pretty = {
        "last_MAPE":   "last MAPE (%)",
        "last_medAPE": "last medAPE (%)",
        "last_RMSE":   "last RMSE (ml)",
        "last_bias":   "last bias (ml)",
        "CD_mm":       "CD (mm)",
        "F1_5mm":      "F1@5mm (%)",
        "inter_MAPE":  "inter MAPE (%)",
        "inter_RMSE":  "inter RMSE (ml)",
    }
    header = "| Condition | " + " | ".join(pretty[c] for c in cols) + " | n |"
    sep = "|---" * (len(cols) + 2) + "|"
    print(header)
    print(sep)
    for key in ["A0", "A2", "Aiso", "Aaniso_sym", "A5"]:
        if key not in rows:
            continue
        r = rows[key]
        cells = " | ".join(fmt(r[c]) for c in cols)
        print(f"| {ROW_LABELS[key]} | {cells} | {r['n_fruits']} |")


def print_diffs(rows):
    diffs = [
        ("Kalman fusion (A2 − A0)",                            "A0", "A2"),
        ("Outlier gating (A_iso − A2)",                        "A2", "Aiso"),
        ("Size/shape decomposition (A_aniso_sym − A_iso)",     "Aiso", "Aaniso_sym"),
        ("Asymmetric + DZ floor (A5 − A_aniso_sym)",           "Aaniso_sym", "A5"),
    ]
    cols = METRIC_ORDER
    pretty = {
        "last_MAPE":   "Δ last MAPE",
        "last_medAPE": "Δ medAPE",
        "last_RMSE":   "Δ RMSE",
        "last_bias":   "Δ bias",
        "CD_mm":       "Δ CD (mm)",
        "F1_5mm":      "Δ F1@5mm",
        "inter_MAPE":  "Δ inter MAPE",
        "inter_RMSE":  "Δ inter RMSE",
    }
    print()
    print("### Difference rows (each step's contribution)")
    print()
    header = "| Step | " + " | ".join(pretty[c] for c in cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    print(header)
    print(sep)
    for name, a, b in diffs:
        if a not in rows or b not in rows:
            print(f"| {name} | (missing) |")
            continue
        ra, rb = rows[a], rows[b]
        cells = " | ".join(fmt(rb[c] - ra[c], prec=2) for c in cols)
        print(f"| {name} | {cells} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A0", required=True)
    ap.add_argument("--A2", required=True)
    ap.add_argument("--Aiso", required=True)
    ap.add_argument("--Aaniso_sym", required=True)
    ap.add_argument("--A5", required=True)
    args = ap.parse_args()
    paths = {
        "A0":         args.A0,
        "A2":         args.A2,
        "Aiso":       args.Aiso,
        "Aaniso_sym": args.Aaniso_sym,
        "A5":         args.A5,
    }
    rows = {}
    for k, p in paths.items():
        if not Path(p).is_file():
            print(f"ERROR: {k} path missing: {p}", file=sys.stderr)
            sys.exit(1)
        rows[k] = extract(p)
    print("## §12 — Ablation: Kalman fusion vs Asymmetric Size-Gate (l64/small/mm)")
    print()
    print_table(rows)
    print_diffs(rows)


if __name__ == "__main__":
    main()
