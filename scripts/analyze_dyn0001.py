"""Hash-check and summarize DYN-0001 fixed one-shot branches."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from neurotopo.equal_logit_future import ARMS,CELLS,CONDITIONS,LAGS,SEEDS
from neurotopo.order_observation import sha
from neurotopo.run import source_fingerprints


def read(path:Path)->dict:return json.loads(path.read_text())
def save(path:Path,value:dict)->None:
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def analyze(out:Path,parent:Path)->dict:
    started=time.monotonic()
    if (out/'failure.json').exists():
        result={'study':'DYN-0001','status':'INCOMPLETE','failure':read(out/'failure.json')}
        save(out/'analysis.json',result);return result
    manifest,complete=read(out/'manifest.json'),read(out/'completion.json')
    if manifest['source_sha256']!=source_fingerprints():raise RuntimeError('Source changed')
    if manifest['version']!='v3':raise RuntimeError('Wrong DYN version')
    if manifest['analysis_script_sha256']!=sha(Path('scripts/analyze_dyn0001.py')):raise RuntimeError('Analysis script changed')
    if manifest['design_sha256']!=sha(Path('research/design/dyn0001_equal_logit_future_observability_v3.md')):raise RuntimeError('Design changed')
    if manifest['suffix_inventory_sha256']!=sha(Path('research/design/dyn0001_v2_suffix_inventory.json')):raise RuntimeError('Suffix inventory changed')
    if sha(parent/'manifest.json')!=manifest['parent_manifest_sha256'] or sha(parent/'completion.json')!=manifest['parent_completion_sha256']:raise RuntimeError('Parent changed')
    if len(complete['rows'])!=52 or complete['cells']!=52 or complete['models']!=26 or complete['suffix_transitions']!=135_249_920 or complete['new_suffix_streams']!=26:raise RuntimeError('Matrix/resource incomplete')
    if len(complete['data_files'])!=26:raise RuntimeError('Extended data matrix incomplete')
    for name,record in complete['data_files'].items():
        if sha(out/name)!=record['sha256']:raise RuntimeError('Extended data changed')
        seed=Path(name).parent.name;cell=Path(name).stem
        if sha(parent/'data'/seed/f'test_{cell}.npz')!=record['original_prefix_sha256']:
            raise RuntimeError('Original prefix input changed')
    if complete['float64_tangent_next_step_evaluations']!=sum(r['float64_tangent_next_step_evaluations'] for r in complete['rows']):
        raise RuntimeError('Tangent compute accounting mismatch')
    rows=[]
    for row in complete['rows']:
        seed,arm,cell=(row[k] for k in ('seed','arm','cell'))
        if seed not in SEEDS or arm not in ARMS or cell not in CELLS:raise RuntimeError('Unexpected matrix row')
        path=out/str(seed)/arm/f'{cell}.npz';dp=out/'data'/str(seed)/f'{cell}.npz'
        cp=parent/str(seed)/arm/'checkpoint-1000.pt';pp=parent/str(seed)/arm/'prototypes.npz'
        for actual,digest in ((path,row['raw_sha256']),(dp,row['input_sha256']),(cp,row['checkpoint_sha256']),(pp,row['prototype_sha256'])):
            if sha(actual)!=digest:raise RuntimeError('Artifact changed: '+str(actual))
        with np.load(path) as f,np.load(dp) as d:
            z=f['logits'];mask=f['valid_event_at_8'];pre=f['logit_at_8_pre'];t=d['targets'][:,8:]
            if z.shape!=(5,512,1017) or mask.shape!=(5,512) or not np.array_equal(z[:,:,0],np.broadcast_to(pre,z[:,:,0].shape)):
                raise RuntimeError('Raw branch shape/time mismatch')
            if not np.isfinite(z).all():raise RuntimeError('Nonfinite branch')
            recomputed={}
            for j,c in enumerate(CONDITIONS):
                diff=z[j].astype(float)-z[0].astype(float);sign=(z[j]>=0)!=(z[0]>=0)
                expected=row['conditions'][c]
                if int(mask[j].sum())!=expected['valid_count']:raise RuntimeError('Validity summary mismatch')
                if abs(float(((z[j,:,-1]>=0)==t[:,-1]).mean())-expected['final_accuracy'])>1e-12:raise RuntimeError('Final accuracy mismatch')
                if abs(float(((z[j,:,1:]>=0)==t[:,1:]).mean())-expected['suffix_accuracy'])>1e-12:raise RuntimeError('Suffix accuracy mismatch')
                for horizon in (256,512,1024):
                    measured=float(((z[j,:,horizon-8]>=0)==t[:,horizon-8]).mean())
                    if abs(measured-expected['accuracy_by_horizon'][str(horizon)])>1e-12:
                        raise RuntimeError('Horizon accuracy mismatch')
                for lag in LAGS:
                    v=expected['lags'][str(lag)]
                    if abs(float(np.abs(diff[:,lag]).mean())-v['mean_abs_logit_diff'])>1e-12 or abs(float(sign[:,lag].mean())-v['sign_diff_fraction'])>1e-12:
                        raise RuntimeError('Lag mismatch')
                recomputed[c]={'valid_count':expected['valid_count'],'attempts':expected['attempts'],
                    'invalid_reasons':expected['invalid_reasons'],'immediate_logit_abs_max':expected['immediate_logit_abs_max'],
                    'gram_displacement_mean':expected['gram_displacement_mean'],
                    'final_accuracy':expected['final_accuracy'],'accuracy_by_horizon':expected['accuracy_by_horizon'],
                    'suffix_accuracy':expected['suffix_accuracy'],
                    'first_sign_divergence_count':expected['first_sign_divergence_count'],
                    'first_sign_divergence_mean_lag':expected['first_sign_divergence_mean_lag'],
                    'lags':expected['lags'],'by_true_initial_label':expected['by_true_initial_label'],
                    'by_pre_sign':expected['by_pre_sign']}
            available=f['tangent_available'];pred=f['next_tangent'];exact=z[3,:,1].astype(float)-z[0,:,1].astype(float)
            if int(available.sum())!=row['tangent']['available_count']:raise RuntimeError('Tangent denominator mismatch')
            if np.any(np.isfinite(pred[~available])):raise RuntimeError('Unavailable tangent was imputed')
            if available.any():
                error=np.max(np.abs(f['double_exact_next'][available]-exact[available]))
                if abs(float(error)-row['tangent']['float_double_delta_error_max'])>1e-12:raise RuntimeError('Double/float error summary mismatch')
        rows.append({'seed':seed,'arm':arm,'cell':cell,'conditions':recomputed,
                     'tangent':row['tangent'],'baseline_max_abs_logit_error':row['baseline_max_abs_logit_error'],
                     'wall_seconds':row['wall_seconds'],'cpu_seconds':row['cpu_seconds'],
                     'raw_sha256':row['raw_sha256']})
    if {(r['seed'],r['arm'],r['cell']) for r in rows}!={(s,a,c) for s in SEEDS for a in ARMS for c in CELLS}:raise RuntimeError('Missing pair')
    lookup={(r['seed'],r['arm'],r['cell']):r for r in rows}
    specific={str(seed):{cell:lookup[seed,'REL',cell] for cell in CELLS} for seed in (25002,25003,23005)}
    effects={}
    for arm in ARMS:
        effects[arm]={}
        for c in CONDITIONS[1:]:
            x=np.array([(lookup[s,arm,'LONG_H']['conditions'][c]['final_accuracy']+lookup[s,arm,'LONG_U']['conditions'][c]['final_accuracy']
                        -lookup[s,arm,'LONG_H']['conditions']['NONE']['final_accuracy']-lookup[s,arm,'LONG_U']['conditions']['NONE']['final_accuracy'])/2 for s in SEEDS[:-1]],float)
            effects[arm][c]={'values':x.tolist(),'mean':float(x.mean()),'sample_sd':float(x.std(ddof=1)),
                             'min':float(x.min()),'max':float(x.max())}
    analysis_seconds=time.monotonic()-started
    if complete['wall_seconds']+analysis_seconds>1800:raise TimeoutError('Shared branch+analysis 1800-second cap exceeded')
    result={'study':'DYN-0001','version':'v3','status':'DESCRIPTIVE_ONLY','scope':'Exposed frozen models/T256 prefixes; v2-frozen longer suffixes; t8 one-shot only; no new training or independent seed confirmation.',
            'source_sha256':manifest['source_sha256'],'manifest_sha256':sha(out/'manifest.json'),
            'completion_sha256':sha(out/'completion.json'),'rows':rows,'specific_cases':specific,'new_seed_effects':effects,
            'resource':{'models':26,'cells':52,'branches':260,'suffix_transitions':135_249_920,'training_updates':0,'prototype_fits':0,
                        'new_suffix_streams':26,'float64_tangent_next_step_evaluations':complete['float64_tangent_next_step_evaluations'],
                        'wall_seconds':complete['wall_seconds'],'analysis_seconds':analysis_seconds}}
    save(out/'analysis.json',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--parent',type=Path,default=Path('artifacts/EXP-0025/v2'))
    args=parser.parse_args();r=analyze(args.run,args.parent);print(json.dumps({k:r[k] for k in ('study','status')}))
