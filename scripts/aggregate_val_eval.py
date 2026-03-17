#!/usr/bin/env python3
"""Aggregate val-set evaluation_summary.json across 32 runs and emit
Shape-priority Top-10 tables (static + temporal) matching the test-set format:
  # | Cell | CD | F1@5mm | last MAPE | medAPE | last RMSE | l-bias | inter MAPE | inter RMSE | inter bias

Cell label: l{latent}/{cell_size}/{norm}   where norm: mm=maxmean, max=max
Sort: by (CD + (100-F1@5))/2 ascending => shape-priority composite.
"""
import json, os, glob, re, sys
from pathlib import Path

ROOT = Path('/home/mainuser/mount/sungjay/tcore/corepp/experiments')

CELL_RE_FT = re.compile(r"pce_finetune_v25_calFix_weakSoftvol_l(\d+)_(small|large)(?:_(max|maxmean))?_noAug_e100$")
CELL_RE_TP = re.compile(r"temporal_kalman_v25_l(\d+)_(small|large)(?:_(max|maxmean))?_softvolFT_noAug_aniso2_sizegate_asym$")

def norm_short(n):
    return 'mm' if n == 'maxmean' else 'max'

def load_summary(eval_dir: Path):
    p = eval_dir / 'evaluation_summary.json'
    if not p.is_file():
        return None
    return json.loads(p.read_text())

def extract_row(summary):
    if summary is None: return None
    occ0 = summary.get('by_occlusion', {}).get('0')
    if occ0 is None: return None
    ld = occ0.get('last_date', {})
    inter = occ0.get('intermediate', {})
    v = ld.get('volume_vs_laser', {})
    sh = ld.get('shape_metrics', {})
    iv = inter.get('volume_vs_corrected_gt', {})
    def g(d, k, default=float('nan')):
        v = d.get(k, default)
        return v if v is not None else float('nan')
    cd_mm = sh.get('chamfer_distance', {}).get('mean', float('nan'))
    if cd_mm and cd_mm == cd_mm:  # not NaN
        cd_mm = cd_mm * 1000.0  # m -> mm
    f1 = sh.get('f1_5mm', {}).get('mean', float('nan'))
    return dict(
        n_last=ld.get('n_fruits', 0),
        cd=cd_mm,
        f1=f1,
        last_mape=g(v,'mape_pct'),
        med_ape=g(v,'medape_pct'),
        last_rmse=g(v,'rmse'),
        l_bias=g(v,'bias'),
        inter_mape=g(iv,'mape_pct'),
        inter_medape=g(iv,'medape_pct'),
        inter_rmse=g(iv,'rmse'),
        inter_bias=g(iv,'bias'),
        n_inter=g(iv,'n'),
    )

def collect(kind: str):
    """kind in {'static','temporal'}"""
    rows = []
    for latent in (16, 32, 64, 128):
        if kind == 'static':
            base = ROOT / f'kmelon_latent{latent}' / 'finetune'
            cell_re = CELL_RE_FT
        else:
            base = ROOT / f'kmelon_latent{latent}' / 'temporal'
            cell_re = CELL_RE_TP
        for run_dir in sorted(base.glob('*' )):
            m = cell_re.search(run_dir.name)
            if not m: continue
            latent_n, cell_size, norm = m.groups()
            if norm is None:
                norm = 'maxmean'  # legacy default (encoder=point_cloud_maxmean_masked)
            cell_label = f'l{latent_n}/{cell_size}/{norm_short(norm)}'
            eval_dir = run_dir / 'val_pass' / 'eval'
            summary = load_summary(eval_dir)
            r = extract_row(summary)
            if r is None:
                print(f'[skip] no val summary: {run_dir.name}', file=sys.stderr)
                continue
            r['cell'] = cell_label
            r['run'] = run_dir.name
            rows.append(r)
    return rows

def composite(r):
    # lower is better: CD (mm) + (100-F1) / 2  — matching "by CD + F1" spirit
    cd = r['cd'] if r['cd']==r['cd'] else 100
    f1 = r['f1'] if r['f1']==r['f1'] else 0
    return cd + (100 - f1)*0.05  # weight: CD ~ 3-4 mm, F1 70-80% — keep CD dominant

def fmt(r, rank):
    return (f"{rank:>2}  {r['cell']:<14}  {r['cd']:>4.2f}  {r['f1']:>5.1f}%  "
            f"{r['last_mape']:>5.2f}%  {r['med_ape']:>5.2f}%  {r['last_rmse']:>6.2f}  "
            f"{r['l_bias']:+6.1f}  {r['inter_mape']:>5.2f}%  {r['inter_rmse']:>6.2f}  {r['inter_bias']:+6.1f}")

def print_table(title, rows, k=10):
    rows = sorted(rows, key=composite)
    print(f'\n=== {title} — Top {k} (by shape-priority: CD + (100-F1)*0.05) ===')
    hdr = " #  Cell            CD    F1@5    lastMAPE  medAPE  lastRMSE  l-bias  intMAPE  intRMSE  intBias"
    print(hdr)
    print('-' * len(hdr))
    for i, r in enumerate(rows[:k], 1):
        print(fmt(r, i))
    # also dump n_last counts for context
    print('\n  (n_last per row):')
    for i, r in enumerate(rows[:k], 1):
        print(f'  #{i:<2} {r["cell"]:<14} n_last={r["n_last"]}  n_inter={int(r.get("n_inter",0))}')

if __name__ == '__main__':
    statics = collect('static')
    temporals = collect('temporal')
    print(f'\nstatic runs collected: {len(statics)}  temporal: {len(temporals)}')
    if statics:
        print_table('STATIC (val)', statics, k=10)
    if temporals:
        print_table('TEMPORAL (val)', temporals, k=10)

    # last_RMSE best alone
    if statics:
        s_best_rmse = min(statics, key=lambda r: r['last_rmse'])
        print(f'\nlast_RMSE best STATIC: {s_best_rmse["cell"]} -> {s_best_rmse["last_rmse"]:.2f} ml')
    if temporals:
        t_best_rmse = min(temporals, key=lambda r: r['last_rmse'])
        print(f'last_RMSE best TEMPORAL: {t_best_rmse["cell"]} -> {t_best_rmse["last_rmse"]:.2f} ml')
