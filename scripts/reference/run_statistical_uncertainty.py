"""Paired fruit bootstrap for comparisons and the validation tau sweep."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from analyze_existing_predictions import load_frame,bootstrap_difference,metrics

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path('data/3DKoreanMelon/benchmarks/score_reference/predictions'))
parser.add_argument('--output', type=Path, default=Path('runs/reference_analysis/uncertainty'))
args = parser.parse_args()
FINAL = args.root / 'final_seed11'
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
COMPARISONS=[
 ('tbl6_noema__noMono','Single-frame (no EMA, no clamp)','fusion'),
 ('tbl6_noema__volMono','No EMA, volume-gated','fusion'),
 ('tbl6_noema__pc1Mono','No EMA, PC1 clamp','fusion'),
 ('tbl6_noema__pc12Mono','No EMA, PC1+PC2 clamp','fusion'),
 ('tbl6_z__noMono','EMA, no clamp','fusion'),
 ('tbl6_z__volMonoLatent','EMA, volume-gated latent','fusion'),
 ('tbl6_z__pc12Mono','EMA, PC1+PC2 clamp','fusion'),
 ('tbl7/abl_pretrainOFF','Pretraining disabled','component'),
 ('tbl7/abl_binBalOFF','Bin balancing disabled','component'),
 ('tbl7/abl_weakVolOFF_true','Weak volume loss disabled','component'),
 ('tbl7/pool_max','Max pooling','component'),
 ('tbl7/abl_lam_super_20p0','Supervised weight 20','component'),
 ('tbl7/abl_lam_super_1p0','Supervised weight 1','component'),
 ('tbl7/abl_lam_vol_weak_4p0','Weak volume weight 4','component'),
 ('tbl7/abl_lam_vol_weak_0p5','Weak volume weight 0.5','component'),
 ('tbl7/abl_super_l2','L2 supervised loss','component'),
]
score_path=FINAL/'mesh_score/eval/evaluation_results.csv';score=load_frame(score_path)
paths=[score_path,Path(__file__)];records=[];shape=[]
for run,label,group in COMPARISONS:
    path=FINAL/run/'eval/evaluation_results.csv';other=load_frame(path);paths.append(path)
    assert score[['fruit_id','date','is_last_frame']].equals(other[['fruit_id','date','is_last_frame']]),run
    assert np.allclose(score.reference_ml,other.reference_ml,atol=1e-6,rtol=0),run
    d=score.copy();d['score']=d.mesh_volume_ml;d['other']=other.mesh_volume_ml
    for last,subset in [(True,'last'),(False,'intermediate')]:
        g=d[d.is_last_frame==last]
        ci=bootstrap_difference(g,'other',5000,20261023)
        records.append(dict(run=run,label=label,group=group,subset=subset,
            score_rmse_ml=metrics(g,'score')['rmse_ml'],other_rmse_ml=metrics(g,'other')['rmse_ml'],
            **ci,bootstrap_seed=20261023))
    if run=='tbl6_noema__noMono':
        x=score[score.is_last_frame];y=other[other.is_last_frame]
        idx=np.random.default_rng(20261023).integers(0,len(x),(5000,len(x)))
        for col,unit,factor in [('f1_5mm','percentage_points',1),('chamfer_distance','mm',1000)]:
            diff=(x[col].to_numpy()-y[col].to_numpy())*factor;assert np.isfinite(diff).all()
            draw=diff[idx].mean(1)
            shape.append(dict(metric=col,unit=unit,score_minus_single=diff.mean(),lower=np.quantile(draw,.025),upper=np.quantile(draw,.975),n_fruits=len(x)))
result=pd.DataFrame(records);result.to_csv(OUT/'paired_rmse.csv',index=False)
pd.DataFrame(shape).to_csv(OUT/'primary_shape_ci.csv',index=False)
# Table 8 uses the development validation fruits, not the 79 test fruits.
# Treat these intervals as exploratory, conditional on the selected tau=3.0;
# they do not adjust for selection using this same validation set.
tau_root=FINAL.parent/'tau_sweep_seed11'
reference_tau_path=tau_root/'tauL32VAL_tau3.0/eval/evaluation_results.csv'
reference_tau=load_frame(reference_tau_path)
reference_tau=reference_tau[reference_tau.is_last_frame].reset_index(drop=True)
assert len(reference_tau)==18 and reference_tau.fruit_id.nunique()==18
paths.append(reference_tau_path);tau_rows=[]
for tau in [1.,2.,3.5,5.,7.,10.]:
    path=tau_root/f'tauL32VAL_tau{tau:.1f}/eval/evaluation_results.csv'
    other=load_frame(path);other=other[other.is_last_frame].reset_index(drop=True)
    assert reference_tau[['fruit_id','date']].equals(other[['fruit_id','date']])
    assert np.allclose(reference_tau.reference_ml,other.reference_ml,atol=1e-6,rtol=0)
    d=reference_tau.copy();d['score']=d.mesh_volume_ml;d['other']=other.mesh_volume_ml
    interval=bootstrap_difference(d,'other',5000,20261023)
    tau_rows.append(dict(tau_days=tau,reference_tau_days=3.,split='validation',
        n_fruits=18,rmse_ml=metrics(d,'other')['rmse_ml'],
        delta_rmse_variant_minus_reference_ml=-interval['delta_rmse_score_minus_comparator_ml'],
        ci95_low_ml=-interval['ci95_high_ml'],ci95_high_ml=-interval['ci95_low_ml'],
        bootstrap_seed=20261023,resamples=5000))
    paths.append(path)
pd.DataFrame(tau_rows).to_csv(OUT/'validation_tau_paired.csv',index=False)
manifest=dict(date='2026-10-24',status='complete',n_comparisons=16,n_subsets=2,bootstrap_seed=20261023,resamples=5000,
    unit='fruit cluster; all intermediate observations from a selected fruit kept together',
    interval='Marginal percentile 95%; no multiplicity adjustment; exploratory ablation comparisons, not simultaneous rankings.',
    direction='Stored difference is RMSE SCoRe minus comparator. Display tables may reverse it, reversing both interval bounds.',
    cohort='79 fruits, 79 last observations, 976 intermediate observations; exact keys and references checked for every comparator.',
    training_scope='Intervals condition on archived trained models and do not measure training-seed or selection variability.',
    additional_validation_tau=dict(n_comparisons=6,n_fruits=18,reference_tau_days=3.,
        direction='Variant minus tau3 RMSE',
        caveat='Exploratory validation-set intervals conditional on selected tau3; not adjusted for model/time-constant selection and not independent generalization evidence.'),
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Calculated 16 paired comparisons x 2 subsets.')
print(result[result.run=='tbl6_noema__noMono'].to_string(index=False))
print(pd.DataFrame(shape).to_string(index=False))
print(pd.DataFrame(tau_rows).to_string(index=False))
