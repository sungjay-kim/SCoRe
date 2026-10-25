"""Recompute occlusion and fine-tuning-seed summaries from released predictions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def metrics(pred, ref):
    error = np.asarray(pred) - np.asarray(ref)
    if not np.isfinite(error).all() or (np.asarray(ref) <= 0).any():
        raise ValueError('Invalid predictions or references; no silent row dropping')
    return dict(n=len(error), rmse_ml=float(np.sqrt(np.mean(error ** 2))),
                mae_ml=float(np.mean(np.abs(error))),
                mape_pct=float(np.mean(np.abs(error) / ref) * 100),
                signed_error_ml=float(np.mean(error)))


def summarize_occlusion(root, out):
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(root / 'analyses/intermittent_fixed_final/predictions.csv', low_memory=False)
    keys = ['occlusion_pct', 'pattern', 'method', 'fruit_id', 'date']
    assert not frame.duplicated(keys).any()
    assert frame[(frame.occlusion_pct > 0) & frame.is_last_frame].is_occluded.all()
    summaries = []
    for (pct, pattern, method, last), g in frame.groupby(['occlusion_pct', 'pattern', 'method', 'is_last_frame']):
        row = dict(occlusion_pct=pct, pattern=pattern, method=method,
                   subset='last' if last else 'intermediate', n_fruits=g.fruit_id.nunique(),
                   n_frames=len(g), n_occluded=int(g.is_occluded.sum()),
                   failed_n=int((~g.success).sum()),
                   **metrics(g.volume_ml.to_numpy(), g.reference_ml.to_numpy()))
        if last:
            row.update(f1_pct=g.f1_pct.mean(), cd_mm=g.cd_mm.mean(), cd_n=int(g.cd_mm.notna().sum()))
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / 'metrics_by_pattern.csv', index=False)
    means = []
    for (pct, method, subset), g in summary.groupby(['occlusion_pct', 'method', 'subset']):
        row = dict(occlusion_pct=pct, method=method, subset=subset, n_patterns=len(g),
                   failed_total=int(g.failed_n.sum()))
        for metric in ['rmse_ml', 'mape_pct', 'mae_ml', 'f1_pct', 'cd_mm']:
            row[metric + '_mean'] = g[metric].mean()
            row[metric + '_sd'] = g[metric].std(ddof=1) if len(g) > 1 else np.nan
        means.append(row)
    pd.DataFrame(means).to_csv(out / 'mean_sd.csv', index=False)
    intervals = []
    for (pct, last), g in frame.groupby(['occlusion_pct', 'is_last_frame']):
        fruits = sorted(g.fruit_id.unique())
        patterns = sorted(g.pattern.unique())
        ids = np.random.default_rng(20261024).integers(0, len(fruits), (5000, len(fruits)))
        estimates = {}; draws = {}
        metric_names = ['rmse_ml', 'mape_pct'] + (['f1_pct', 'cd_mm'] if last else [])
        for method in ['single', 'score']:
            per_est = {m: [] for m in metric_names}; per_draw = {m: [] for m in metric_names}
            for pattern in patterns:
                h = g[(g.method == method) & (g.pattern == pattern)].copy()
                error = h.volume_ml - h.reference_ml
                h['sq_error'] = error ** 2; h['ape'] = error.abs() / h.reference_ml * 100
                agg = h.groupby('fruit_id').agg(count=('volume_ml', 'size'), sq=('sq_error', 'sum'), ape=('ape', 'sum')).loc[fruits]
                counts = agg['count'].to_numpy()
                n = counts[ids].sum(1)
                per_est['rmse_ml'].append(np.sqrt(agg.sq.sum() / counts.sum()))
                per_draw['rmse_ml'].append(np.sqrt(agg.sq.to_numpy()[ids].sum(1) / n))
                per_est['mape_pct'].append(agg.ape.sum() / counts.sum())
                per_draw['mape_pct'].append(agg.ape.to_numpy()[ids].sum(1) / n)
                if last:
                    assert len(h) == 79 and h.cd_mm.notna().all(), 'CD denominator requires explicit failure handling'
                    for metric in ['f1_pct', 'cd_mm']:
                        values = h.set_index('fruit_id').loc[fruits, metric].to_numpy()
                        per_est[metric].append(values.mean())
                        per_draw[metric].append(values[ids].mean(1))
            for metric in metric_names:
                estimates[(method, metric)] = float(np.mean(per_est[metric]))
                draws[(method, metric)] = np.mean(per_draw[metric], axis=0)
        for metric in metric_names:
            for method in ['single', 'score', 'score_minus_single']:
                if method == 'score_minus_single':
                    estimate = estimates[('score', metric)] - estimates[('single', metric)]
                    samples = draws[('score', metric)] - draws[('single', metric)]
                else:
                    estimate = estimates[(method, metric)]; samples = draws[(method, metric)]
                intervals.append(dict(occlusion_pct=pct, subset='last' if last else 'intermediate',
                                      metric=metric, method=method, n_fruits=79,
                                      n_patterns=len(patterns), estimate=estimate,
                                      lower=float(np.quantile(samples, .025)), upper=float(np.quantile(samples, .975)),
                                      resamples=5000, seed=20261024))
    pd.DataFrame(intervals).to_csv(out / 'bootstrap.csv', index=False)


def summarize_seeds(root, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in (11, 22, 33):
        frame = pd.read_csv(root / f'analyses/seed_variability/predictions_seed{seed}.csv')
        assert len(frame) == 2110 and not frame.duplicated(['fruit_id', 'date', 'method']).any()
        for (method, last), group in frame.groupby(['method', 'is_last_frame']):
            rows.append(dict(seed=seed, method=method, subset='last' if last else 'intermediate',
                             n_frames=len(group), n_fruits=group.fruit_id.nunique(),
                             **metrics(group.volume_ml.to_numpy(), group.reference_ml.to_numpy()),
                             f1_pct=group.f1_pct.mean(), cd_mm=group.cd_mm.mean(),
                             failed_n=int((~group.success).sum()), cd_n=int(group.cd_mm.notna().sum())))
    table = pd.DataFrame(rows)
    table.to_csv(out / 'metrics_by_seed.csv', index=False)
    result = []
    for (method, subset), group in table.groupby(['method', 'subset']):
        for metric in ['rmse_ml', 'mape_pct', 'f1_pct', 'cd_mm']:
            values = group[metric].dropna()
            if len(values):
                result.append(dict(method=method, subset=subset, metric=metric,
                                   n_seeds=len(values), mean=values.mean(), sample_sd=values.std(ddof=1)))
    pd.DataFrame(result).to_csv(out / 'seed_mean_sd.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/3DKoreanMelon/benchmarks/score_reference'))
    parser.add_argument('--output', type=Path, default=Path('runs/reference_analysis'))
    args = parser.parse_args()
    summarize_occlusion(args.root, args.output / 'occlusion')
    summarize_seeds(args.root, args.output / 'seeds')
    print('Recomputed five occlusion patterns and three fine-tuning seeds.')
