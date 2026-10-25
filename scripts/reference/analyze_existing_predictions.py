"""Analyses of the archived paper predictions; no model retraining.

Run with the existing corepp Python environment. Results describe the original
caliper-filtered evaluation cohort, not a newly independent validation protocol.
No test reference enters a temporal filter or the linear calibration fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, isotonic_regression
from scipy.special import expit


ROOT = Path("data/3DKoreanMelon/benchmarks/score_reference/predictions")
KEYS = ["fruit_id", "date"]


def load_frame(path):
    df = pd.read_csv(path, dtype={"fruit_id": str, "date": str})
    df = df.loc[df.occlusion_pct == 0].copy()
    if df.duplicated(KEYS).any():
        raise ValueError(f"Duplicate fruit/date rows in {path}")
    df["is_last_frame"] = df.is_last_frame.astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if df.is_last_frame.isna().any():
        raise ValueError("Unrecognized last-frame flag")
    df["date_dt"] = pd.to_datetime(df.date, format="%Y%m%d")
    df["reference_ml"] = np.where(
        df.is_last_frame, df.gt_volume_ml, df.corrected_gt_ml
    )
    for name in ["mesh_volume_ml", "reference_ml"]:
        if not np.isfinite(df[name]).all():
            raise ValueError(f"Nonfinite {name}: do not silently drop rows")
    if (df.reference_ml <= 0).any():
        raise ValueError("Nonpositive reference; MAPE undefined")
    return df.sort_values(KEYS).reset_index(drop=True)


def load_pair(single_path, score_path):
    single, score = load_frame(single_path), load_frame(score_path)
    if not single[KEYS].equals(score[KEYS]):
        raise ValueError("Single-frame and SCoRe cohorts differ")
    if not np.array_equal(single.is_last_frame, score.is_last_frame):
        raise ValueError("Last-frame flags differ")
    if not np.allclose(single.reference_ml, score.reference_ml, rtol=0, atol=1e-6):
        raise ValueError("References differ between methods")
    single["single"] = single.mesh_volume_ml
    single["score"] = score.mesh_volume_ml
    single["score_f1"] = score.f1_5mm
    single["score_cd_mm"] = score.chamfer_distance * 1000
    single["single_f1"] = single.f1_5mm
    single["single_cd_mm"] = single.chamfer_distance * 1000
    return single


def causal_pava(values):
    """Endpoint of equal-weight isotonic regression fitted to each prefix.

    Previously emitted values are not revised. Endpoint outputs need not form a
    monotone sequence because each new prefix solves a different regression.
    """
    blocks = []
    output = []
    for value in values:
        blocks.append([float(value), 1])
        while len(blocks) > 1:
            a, b = blocks[-2], blocks[-1]
            if a[0] / a[1] <= b[0] / b[1]:
                break
            blocks[-2:] = [[a[0] + b[0], a[1] + b[1]]]
        output.append(blocks[-1][0] / blocks[-1][1])
    return np.array(output)


def logistic_prefix(times, values):
    """Causal three-parameter logistic prefix fit with fixed numerical rules.

    First three visits use the current single-frame estimate (insufficient data).
    Later fits use two deterministic initializations, bounded least squares, and
    return only the fitted endpoint. Bounds depend on past predictions/times, not
    reference volume. These numerical settings are not optimized on test errors.
    """
    result = np.asarray(values, dtype=float).copy()
    failed = 0
    for end in range(4, len(values) + 1):
        t = np.asarray(times[:end], dtype=float)
        y = np.asarray(values[:end], dtype=float)
        span = max(float(t[-1]), 1.0)
        vmax = max(float(np.max(y)), 1.0)
        scale = max(vmax, 1.0)
        best = None
        for rate, multiple in [(0.05, 1.1), (0.2, 2.0)]:
            x0 = np.array([multiple * vmax, rate, np.median(t)])
            fit = least_squares(
                lambda x: (x[0] * expit(x[1] * (t - x[2])) - y) / scale,
                x0,
                bounds=([1e-6, 1e-4, -2 * span], [10 * vmax, 5.0, 3 * span]),
                x_scale=[vmax, 0.1, span],
                max_nfev=300,
            )
            if fit.success and np.isfinite(fit.fun).all():
                cost = float(np.sum(fit.fun ** 2))
                if best is None or cost < best[0]:
                    best = (cost, fit.x)
        if best is None:
            failed += 1
        else:
            level, rate, midpoint = best[1]
            result[end - 1] = level * expit(rate * (t[-1] - midpoint))
    return result, failed


def filters(times, values, tau, include_logistic=True):
    values = np.asarray(values, dtype=float)
    ema = values.copy()
    feedback = values.copy()
    for i in range(1, len(values)):
        gap = times[i] - times[i - 1]
        if gap <= 0:
            raise ValueError("Observation dates must increase strictly")
        alpha = -np.expm1(-gap / tau)
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        proposal = alpha * values[i] + (1 - alpha) * feedback[i - 1]
        feedback[i] = max(feedback[i - 1], proposal)
    out = {
        "volume_ema": ema,
        "volume_running_max": np.maximum.accumulate(values),
        "volume_ema_then_max": np.maximum.accumulate(ema),
        "volume_ema_max_feedback": feedback,
        "volume_causal_isotonic": causal_pava(values),
    }
    failures = 0
    if include_logistic:
        out["volume_causal_logistic"], failures = logistic_prefix(times, values)
    return out, failures


def add_temporal(df, tau, include_logistic=True):
    df = df.copy()
    failures = 0
    for _, group in df.groupby("fruit_id", sort=True):
        days = (group.date_dt - group.date_dt.iloc[0]).dt.total_seconds().to_numpy() / 86400
        output, nfailed = filters(days, group.single.to_numpy(), tau, include_logistic)
        failures += nfailed
        for name, values in output.items():
            df.loc[group.index, name] = values
    return df, failures


def metrics(df, method):
    error = df[method].to_numpy() - df.reference_ml.to_numpy()
    return {
        "n_frames": len(df),
        "n_fruits": df.fruit_id.nunique(),
        "rmse_ml": float(np.sqrt(np.mean(error ** 2))),
        "mae_ml": float(np.mean(np.abs(error))),
        "mape_pct": float(100 * np.mean(np.abs(error) / df.reference_ml)),
        "mean_signed_error_ml": float(np.mean(error)),
    }


def subsets(df):
    return [("last", df[df.is_last_frame]), ("intermediate", df[~df.is_last_frame])]


def bootstrap_difference(df, comparator, repeats, seed):
    """Cluster paired resampling; preserves pooled-frame metric within each draw."""
    temp = pd.DataFrame({
        "fruit_id": df.fruit_id,
        "sse_score": (df.score - df.reference_ml) ** 2,
        "sse_other": (df[comparator] - df.reference_ml) ** 2,
        "count": 1,
    }).groupby("fruit_id", sort=True).sum()
    a = temp.to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(repeats, len(a)))
    totals = a[indices].sum(axis=1)
    draws = np.sqrt(totals[:, 0] / totals[:, 2]) - np.sqrt(totals[:, 1] / totals[:, 2])
    return {
        "delta_rmse_score_minus_comparator_ml":
            metrics(df, "score")["rmse_ml"] - metrics(df, comparator)["rmse_ml"],
        "ci95_low_ml": float(np.quantile(draws, .025)),
        "ci95_high_ml": float(np.quantile(draws, .975)),
        "n_fruits": len(a), "n_frames": len(df), "bootstrap_repeats": repeats,
    }


def monotonicity(df, method):
    rows = []
    for fruit, group in df.groupby("fruit_id", sort=True):
        values = group[method].to_numpy()
        changes = np.diff(values)
        if len(changes) == 0:
            continue
        drops = -changes[changes < 0]
        relative = -changes[changes < 0] / np.maximum(values[:-1][changes < 0], 1e-12) * 100
        rows.append({
            "fruit_id": fruit, "method": method, "n_frames": len(values),
            "n_transitions": len(changes), "n_decreases": int((changes < 0).sum()),
            "n_decreases_over_1ml": int((changes < -1).sum()),
            "nondecreasing": bool((changes >= 0).all()),
            "strictly_increasing": bool((changes > 0).all()),
            "max_decrease_ml": float(drops.max()) if len(drops) else 0.0,
            "sum_decrease_ml": float(drops.sum()),
            "max_decrease_pct": float(relative.max()) if len(relative) else 0.0,
        })
    return rows


def monotonicity_summary(df, method):
    per_fruit = pd.DataFrame(monotonicity(df, method))
    change = df.groupby("fruit_id", sort=True)[method].diff()
    decreases = change < 0
    drops = -change[decreases]
    previous = df[method] - change
    relative = 100 * drops / previous[decreases]
    result = {
        "method": method, "n_fruits": len(per_fruit),
        "n_transitions": int(per_fruit.n_transitions.sum()),
        "n_fruits_with_decrease": int((per_fruit.n_decreases > 0).sum()),
        "n_nondecreasing_fruits": int(per_fruit.nondecreasing.sum()),
        "n_strictly_increasing_fruits": int(per_fruit.strictly_increasing.sum()),
        "n_decreases": int(decreases.sum()),
        "n_fruits_with_decrease_over_1ml": int((per_fruit.n_decreases_over_1ml > 0).sum()),
        "n_decreases_over_1ml": int(per_fruit.n_decreases_over_1ml.sum()),
    }
    for unit, values in [("ml", drops), ("pct", relative)]:
        for name, value in [("mean", values.mean()), ("median", values.median()),
                            ("p95", values.quantile(.95)), ("max", values.max())]:
            # No decrease has no conditional magnitude distribution.
            result[f"decrease_{name}_{unit}"] = float(value) if len(values) else None
    return result


def self_check():
    # Reference implementation check, plus append-future invariance.
    y = np.array([12., 20., 10., 23., 22., 31.])
    t = np.array([0., 2., 5., 8., 10., 15.])
    ours = causal_pava(y)
    expected = np.array([isotonic_regression(y[:i]).x[-1] for i in range(1, len(y) + 1)])
    np.testing.assert_allclose(ours, expected)
    a, _ = filters(t, y, 3.)
    b, _ = filters(np.append(t, 20.), np.append(y, 9999.), 3.)
    for name in a:
        np.testing.assert_allclose(a[name], b[name][:-1], atol=1e-10)
    const, _ = filters(t, np.full(len(t), 10.), 3., include_logistic=False)
    for values in const.values():
        np.testing.assert_allclose(values, 10.)
    return {"pava_matches_scipy": True, "append_future_invariance": True,
            "constant_sequence": True}


def file_info(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("runs/reference_analysis/scalar"))
    parser.add_argument("--tau", type=float, default=3.0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    if args.tau <= 0 or args.bootstrap < 100:
        raise ValueError("Positive tau and at least 100 bootstrap draws required")
    args.output.mkdir(parents=True, exist_ok=True)
    checks = self_check()
    paths = {
        "test_single": args.root / "final_seed11/mesh_single_allframes/eval/evaluation_results.csv",
        "test_score": args.root / "final_seed11/mesh_score/eval/evaluation_results.csv",
        "validation_single": args.root / "gridval_seq/l32_large_maxmean_single/eval/evaluation_results.csv",
        "validation_score": args.root / "tau_sweep_seed11/tauL32VAL_tau3.0/eval/evaluation_results.csv",
    }
    cohorts = {
        "test": load_pair(paths["test_single"], paths["test_score"]),
        "validation": load_pair(paths["validation_single"], paths["validation_score"]),
    }
    val_last = cohorts["validation"].loc[cohorts["validation"].is_last_frame]
    design = np.column_stack([val_last.single, np.ones(len(val_last))])
    coefficient, _, rank, _ = np.linalg.lstsq(design, val_last.reference_ml, rcond=None)
    if rank < 2:
        raise ValueError("Linear calibration is not identifiable")
    # Supplement the shared tau=3 comparison with selection on validation only.
    # Use the same candidate grid as the original manuscript's tau sweep.
    tau_candidates = [1., 2., 3., 3.5, 5., 7., 10.]
    tau_methods = ["volume_ema", "volume_ema_then_max", "volume_ema_max_feedback"]
    sweep = []
    for candidate in tau_candidates:
        current, _ = add_temporal(cohorts["validation"], candidate, include_logistic=False)
        for method in tau_methods:
            sweep.append({"method": method, "tau_days": candidate,
                          **metrics(current[current.is_last_frame], method)})
    sweep_df = pd.DataFrame(sweep)
    selected_taus = {}
    for method in tau_methods:
        ranked = sweep_df[sweep_df.method == method].sort_values(["rmse_ml", "tau_days"])
        selected_taus[method] = float(ranked.iloc[0].tau_days)
    sweep_df.to_csv(args.output / "validation_tau_sweep.csv", index=False)
    # Save the validation selection before evaluating the selected variants on test.
    (args.output / "selected_scalar_taus.json").write_text(
        json.dumps(selected_taus, indent=2) + "\n")
    summaries, intervals, bins, mono, mono_summary = [], [], [], [], []
    fit_failures = {}
    methods = ["single", "score", "volume_ema", "volume_running_max",
               "volume_ema_then_max", "volume_ema_max_feedback",
               "volume_causal_isotonic", "volume_causal_logistic",
               "single_linear_calibrated"]
    methods += [method + "_val_tau" for method in tau_methods]
    for split, df in cohorts.items():
        df, failed = add_temporal(df, args.tau)
        fit_failures[split] = failed
        df["single_linear_calibrated"] = coefficient[0] * df.single + coefficient[1]
        for method, tau in selected_taus.items():
            selected, _ = add_temporal(df, tau, include_logistic=False)
            df[method + "_val_tau"] = selected[method]
        df["split"] = split
        df.drop(columns=["date_dt"]).to_csv(args.output / f"{split}_predictions.csv", index=False)
        for subset, part in subsets(df):
            for method in methods:
                summaries.append({"split": split, "subset": subset, "method": method,
                                  **metrics(part, method)})
            # Validation is used in calibration; inferential comparison is test only.
            if split == "test":
                for method in methods:
                    if method != "score":
                        intervals.append({"subset": subset, "comparator": method,
                                          **bootstrap_difference(part, method, args.bootstrap, args.seed)})
                for lo, hi in [(0, 100), (100, 200), (200, 300), (300, float("inf"))]:
                    group = part[(part.reference_ml >= lo) & (part.reference_ml < hi)]
                    if len(group):
                        for method in ["single", "score", "single_linear_calibrated"]:
                            bins.append({"subset": subset, "volume_bin_ml": f"[{lo},{hi})",
                                         "method": method, **metrics(group, method)})
        if split == "test":
            for method in methods:
                mono.extend(monotonicity(df, method))
                mono_summary.append(monotonicity_summary(df, method))
            errors = df.copy()
            errors["absolute_error_single"] = (df.single - df.reference_ml).abs()
            errors["absolute_error_score"] = (df.score - df.reference_ml).abs()
            failure_rows = []
            for subset, part in subsets(errors):
                group = part.groupby("fruit_id")[["absolute_error_single", "absolute_error_score"]].mean()
                for fruit, row in group.iterrows():
                    failure_rows.append({
                        "subset": subset, "fruit_id": fruit,
                        "mae_single_ml": row.absolute_error_single,
                        "mae_score_ml": row.absolute_error_score,
                        "score_minus_single_mae_ml": row.absolute_error_score - row.absolute_error_single,
                        "score_worse": bool(row.absolute_error_score > row.absolute_error_single),
                    })
            pd.DataFrame(failure_rows).to_csv(args.output / "per_fruit_error_comparison.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(intervals).to_csv(args.output / "paired_bootstrap.csv", index=False)
    pd.DataFrame(bins).to_csv(args.output / "volume_strata.csv", index=False)
    pd.DataFrame(mono).to_csv(args.output / "monotonicity_per_fruit.csv", index=False)
    pd.DataFrame(mono_summary).to_csv(args.output / "monotonicity_summary.csv", index=False)
    # Shape gain is measured for the meshes, which scalar filtering leaves unchanged.
    last = cohorts["test"][cohorts["test"].is_last_frame]
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(last), size=(args.bootstrap, len(last)))
    shape_rows = []
    for name in ["f1", "cd_mm"]:
        differences = last["score_" + name].to_numpy() - last["single_" + name].to_numpy()
        draws = differences[indices].mean(axis=1)
        shape_rows.append({"metric": name, "n_fruits": len(last),
                           "single_mean": float(last["single_" + name].mean()),
                           "score_mean": float(last["score_" + name].mean()),
                           "score_minus_single": float(differences.mean()),
                           "ci95_low": float(np.quantile(draws, .025)),
                           "ci95_high": float(np.quantile(draws, .975))})
    pd.DataFrame(shape_rows).to_csv(args.output / "paired_shape_metrics.csv", index=False)
    manifest = {
        "analysis_date": "2026-10-23", "command": sys.argv,
        "inputs": {name: file_info(path) for name, path in paths.items()},
        "script": file_info(__file__), "self_checks": checks,
        "tau_days": args.tau, "bootstrap_repeats": args.bootstrap, "bootstrap_seed": args.seed,
        "validation_selected_scalar_taus": selected_taus,
        "scalar_tau_candidates_days": tau_candidates,
        "linear_calibration": {
            "fit_split": "encoder validation last frames", "n": len(val_last),
            "slope": float(coefficient[0]), "intercept_ml": float(coefficient[1]),
            "applied_to": "all test frames without clipping",
            "validation_last_prediction_range_ml": [float(val_last.single.min()), float(val_last.single.max())],
            "test_prediction_range_ml": [float(cohorts["test"].single.min()), float(cohorts["test"].single.max())],
            "negative_test_calibrated_predictions": int((coefficient[0] * cohorts["test"].single + coefficient[1] < 0).sum()),
        },
        "logistic_prefix_fit_failures_falling_back_to_current_single": fit_failures,
        "limitations": [
            "Archived caliper-filtered submission cohort; not a new holdout evaluation.",
            "Encoder validation fruits were included in decoder pretraining.",
            "Logistic numerical rules are fixed exploratory choices, not validation-optimized.",
            "Causal prefix isotonic and logistic emitted endpoints need not be monotone.",
            "Scalar methods do not modify meshes; no improved CD/F1 is claimed for them.",
            "Intervals describe fruit sampling for one trained model, not seed uncertainty.",
            "Calibration/tau-selection uncertainty is not propagated into test bootstrap intervals.",
            "Last-frame calibration applied to intermediate frames is a transfer diagnostic.",
            "CSV volumes are rounded to 0.1 mL.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    table = pd.DataFrame(summaries)
    print(table[(table.split == "test") & (table.subset == "last")].to_string(index=False))
    print(json.dumps({"self_checks": checks, "linear_calibration": manifest["linear_calibration"],
                      "logistic_failures": fit_failures}, indent=2))


if __name__ == "__main__":
    main()

