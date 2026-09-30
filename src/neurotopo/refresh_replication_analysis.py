"""EXP0025 fixed independent-qualification and conditional intervention analysis."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .refresh_factor_core import CONDITIONS, shape, weighted
from .state_refresh_analysis import scores, _same_numbers, _read, _save_compact
from .order_observation import sha
from .run import source_fingerprints

SEEDS=tuple(range(25001,25013))
CELLS=('IID_H','IID_U','LONG_H','LONG_U')
METRIC_COLUMNS=('bce','accuracy_all','final_accuracy','hold_accuracy','flip_accuracy','first_error_count')
DIAGNOSTIC_COLUMNS=('valid_fraction','valid_target_logit_error_max','valid_shape_error_max','valid_sign_changes','valid_nonzero_gram_fraction','local_logit_abs_max','local_gram_abs_max')


def refresh_diagnostics(raw: dict,condition: str) -> dict:
    z,g,pz,pg=(raw[k] for k in ('logits','gram','post_logits','post_gram'))
    mask=np.asarray(raw['refresh_mask'],bool);expected=np.zeros(z.shape[1],bool)
    if condition not in CONDITIONS:raise ValueError(condition)
    if condition!='NONE':expected[8::8]=True
    if not np.array_equal(mask,expected) or g.shape!=(*z.shape,3) or pz.shape!=z.shape or pg.shape!=g.shape:raise RuntimeError('Wrong schedule or shapes')
    if not np.array_equal(z[:,~mask],pz[:,~mask]) or not np.array_equal(g[:,~mask],pg[:,~mask]):raise RuntimeError('Unscheduled mutation')
    for k,v in raw.items():
        if not np.isfinite(v).all():raise RuntimeError('Nonfinite '+k)
        if k not in ('gram','post_gram','logits','post_logits','refresh_mask') and v.shape!=z.shape:raise RuntimeError('Audit shape '+k)
    selected=raw['selected_labels'];valid=raw['valid_event'][:,mask];reason=raw['invalid_reason'][:,mask]
    if np.any(raw['valid_event'][:,~mask]) or np.any(selected[:,~mask]!=-1):raise RuntimeError('Event outside schedule')
    if condition in ('NONE','GAUGE8'):
        if np.any(selected!=-1):raise RuntimeError('Control selected prototype')
    elif not np.array_equal(selected[:,mask],(z[:,mask]>=0).astype(int)):raise RuntimeError('Non-self prototype choice')
    if not np.array_equal(valid,reason==0):raise RuntimeError('Validity/reason mismatch')
    zg=z[:,mask].astype(float);zp=pz[:,mask].astype(float);gg=g[:,mask].astype(float);gp=pg[:,mask].astype(float)
    dg=np.linalg.norm(weighted(gp-gg),axis=-1)
    if not np.allclose(dg,raw['gram_delta_norm'][:,mask],atol=1e-12,rtol=0):raise RuntimeError('Gram change mismatch')
    if condition in ('EQUAL8','SCALE8') and (not np.array_equal(zg[~valid],zp[~valid]) or not np.array_equal(gg[~valid],gp[~valid]) or np.any(raw['state_delta_norm'][:,mask][~valid]!=0)):raise RuntimeError('Invalid event did not fall back')
    if condition=='EQUAL8' and not np.allclose(np.abs(zp-zg),raw['logit_target_error'][:,mask],rtol=0,atol=1e-12):raise RuntimeError('Equal-logit audit mismatch')
    if condition=='SCALE8':
        measured=np.linalg.norm(shape(gp)[0]-shape(gg)[0],axis=-1)
        if not np.allclose(measured,raw['shape_error'][:,mask],atol=1e-12,rtol=0):raise RuntimeError('Scale shape audit mismatch')
    count=int(valid.size);nv=int(valid.sum())
    def maximum(x):return float(np.max(x)) if np.size(x) else None
    def mean(x):return float(np.mean(x)) if np.size(x) else None
    result={'refresh_calls':count,'valid_count':nv,'valid_fraction':nv/count if count else None,
            'invalid_reasons':{str(i):int((reason==i).sum()) for i in range(1,8)},
            'local_logit_abs_max':maximum(np.abs(zp-zg)), 'local_gram_abs_max':maximum(np.abs(gp-gg)),
            'valid_target_logit_error_max':maximum(raw['logit_target_error'][:,mask][valid]),
            'valid_shape_error_max':maximum(raw['shape_error'][:,mask][valid]),
            'valid_sign_changes':int(((zg>=0)!=(zp>=0))[valid].sum()),
            'valid_nonzero_gram_fraction':mean(dg[valid]>1e-6),
            'gram_change_mean':mean(dg),'gram_change_max':maximum(dg),
            'state_change_mean':mean(raw['state_delta_norm'][:,mask]),
            'state_change_max':maximum(raw['state_delta_norm'][:,mask]),
            'prototype_distance_before_mean':mean(raw['prototype_distance_before'][:,mask]),
            'prototype_distance_after_mean':mean(raw['prototype_distance_after'][:,mask])}
    available=raw['ratio_available'][:,mask]
    vals=raw['proposed_scale'][:,mask][available]
    result['available_ratio_count']=int(available.sum());result['ratio_range']=[float(vals.min()),float(vals.max())] if len(vals) else None
    return result


def rate(k: int,n: int) -> dict:
    if not 0<=k<=n:raise ValueError('Invalid rate')
    if not n:return {'count':k,'denominator':n,'rate':None,'wilson95':None}
    z=1.959963985;p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {'count':k,'denominator':n,'rate':p,'wilson95':[max(0.,center-half),min(1.,center+half)]}


def long_a(r: dict,c: str,phase: str='evaluation') -> float:
    return float(np.mean([r[phase][c][cell]['metrics']['final_accuracy'] for cell in ('LONG_H','LONG_U')]))


def gauge_pass(r: dict) -> bool:
    return all(r['evaluation']['GAUGE8'][cell]['diagnostics'][k]<=1e-5 for cell in CELLS for k in ('local_logit_abs_max','local_gram_abs_max')) and abs(long_a(r,'GAUGE8')-long_a(r,'NONE'))<=.005


def decide(rows: list[dict]) -> dict:
    lookup={(r['seed'],r['arm']):r for r in rows}
    if len(rows)!=26 or set(lookup)!={(s,a) for s in (*SEEDS,23005) for a in ('REL','INV')}:raise ValueError('Incomplete model matrix')
    info=[]
    for s in SEEDS:
        r=lookup[s,'REL'];cal=r['calibration'];q=r['qualification']['NONE']
        qualified=bool(cal['calibration_accuracy']>=.95 and cal['prototype_labels_correct'] and q['IID_H']['metrics']['accuracy_all']>=.95 and q['IID_U']['metrics']['accuracy_all']>=.95 and q['IID_U']['metrics']['flip_accuracy']>=.95)
        aq=long_a(r,'NONE','qualification');a={c:long_a(r,c) for c in CONDITIONS};gp=gauge_pass(r)
        stratum='FAILURE' if qualified and aq<.95 else 'SUCCESS' if qualified else 'UNQUALIFIED'
        diags=[r['evaluation']['EQUAL8'][cell]['diagnostics'] for cell in ('LONG_H','LONG_U')]
        feasible=all(d['valid_fraction']>=.99 and d['valid_target_logit_error_max'] is not None and d['valid_target_logit_error_max']<=1e-5 and d['valid_sign_changes']==0 for d in diags)
        changed=all(d['valid_nonzero_gram_fraction'] is not None and d['valid_nonzero_gram_fraction']>=.5 for d in diags)
        exclusion='NOT_QUALIFICATION_FAILURE' if stratum!='FAILURE' else 'GAUGE' if not gp else 'EQUAL_FEASIBILITY' if not feasible else 'INSUFFICIENT_CHANGE' if not changed else None
        info.append({'seed':s,'qualified':qualified,'stratum':stratum,'qualification_A':aq,'final_A':a,'gauge_pass':gp,'M_exclusion':exclusion,
                     'baseline_failure_reproduced':a['NONE']<.95,
                     'repair':{c:bool(a['NONE']<.95 and a[c]>=.95 and a[c]-a['NONE']>=.10) for c in CONDITIONS[2:]},
                     'harm':{c:bool(a[c]<.95 and a[c]-a['NONE']<-.02) for c in CONDITIONS[2:]},
                     'delta':{c:a[c]-a['NONE'] for c in CONDITIONS[1:]}})
    failures=[i for i in info if i['stratum']=='FAILURE'];success=[i for i in info if i['stratum']=='SUCCESS'];eligible=[i for i in failures if i['M_exclusion'] is None]
    methods={c:{'repair_original_failures':rate(sum(i['repair'][c] for i in failures),len(failures)),
                'harm_qualified_successes':rate(sum(i['harm'][c] for i in success),len(success)),
                'all_negative_deltas':{str(i['seed']):i['delta'][c] for i in info if i['delta'][c]<0}} for c in CONDITIONS[2:]}
    allg=all(i['gauge_pass'] for i in info if i['qualified']);rr=methods['FULL8']['repair_original_failures']
    R='R_INCONCLUSIVE_FAILURE_COUNT' if len(failures)<2 else 'R_SUPPORTED_IN_NEW_BATCH' if rr['rate']>=.8 and methods['FULL8']['harm_qualified_successes']['count']==0 and allg else 'R_NOT_SUPPORTED'
    mr=rate(sum(i['repair']['EQUAL8'] for i in eligible),len(eligible))
    M='M_INCONCLUSIVE_ELIGIBILITY' if len(eligible)<2 else 'M_SUPPORTED_WITHIN_EQUAL_LOGIT_FAMILY' if mr['rate']>=.8 else 'M_NOT_SUPPORTED'
    rs=R=='R_SUPPORTED_IN_NEW_BATCH';ms=M=='M_SUPPORTED_WITHIN_EQUAL_LOGIT_FAMILY'
    effects={}
    for arm in ('REL','INV'):
        effects[arm]={}
        for c in CONDITIONS[1:]:
            x=np.array([long_a(lookup[s,arm],c)-long_a(lookup[s,arm],'NONE') for s in SEEDS]);mean=float(x.mean());sd=float(x.std(ddof=1));half=2.200985160*sd/np.sqrt(12)
            effects[arm][c]={'values':x.tolist(),'mean':mean,'sample_sd':sd,'min':float(x.min()),'max':float(x.max()),'descriptive_t95':[mean-half,mean+half]}
    return {'decision':'SUPPORTED' if rs and ms else 'PARTIALLY_SUPPORTED' if rs or ms else 'INCONCLUSIVE',
            'substatus':'NEW_SEED_REFRESH_AND_EQUAL_LOGIT_RESCUE' if rs and ms else 'R_ONLY_SUPPORTED' if rs else 'M_ONLY_SUPPORTED' if ms else 'NEITHER_ENDPOINT_SUPPORTED',
            'R':R,'M':M,'qualified_failure_seeds':[i['seed'] for i in failures], 'qualified_success_seeds':[i['seed'] for i in success],
            'M_eligible_seeds':[i['seed'] for i in eligible],'M_conditional_repair':mr,'all_qualified_gauge_pass':allg,
            'seed_decisions':info,'methods':methods,'new_seed_effects':effects}


def analyze(out: Path) -> dict:
    start=time.monotonic()
    if any((out/name).exists() for name in ('run_failure.json','worker_failure.json')):
        failures={name:_read(out/name) for name in ('run_failure.json','worker_failure.json') if (out/name).exists()}
        result={'experiment_id':'EXP-0025','version':'v2','decision':'INCONCLUSIVE','substatus':'INCOMPLETE_BUDGET' if 'TimeoutError' in str(failures) else 'ENGINEERING_ABORT','failures':failures,'gate1_claim':False}
        _save_compact(out/'analysis.json',result);return result
    manifest=_read(out/'manifest.json');complete=_read(out/'completion.json')
    if manifest['source_sha256']!=source_fingerprints() or complete['status']!='COMPLETED':raise RuntimeError('Source/completion mismatch')
    for field in ('frozen_hashes','parent_artifact_sha256'):
        for path,digest in manifest[field].items():
            if sha(Path(path))!=digest:raise RuntimeError('Frozen source changed '+path)
    for path,digest in complete['data_files'].items():
        if sha(out/path)!=digest:raise RuntimeError('Input changed '+path)
    for key,expected in {'research_updates':24000,'prototype_fits':26,'qualification_count':104,'evaluation_count':520,'monitor_checkpoints':120,'training_transition_tokens':8256000,'evaluation_transition_tokens':42172416,'calibration_transition_tokens':106496}.items():
        if complete[key]!=expected:raise RuntimeError('Resource count '+key)
    compact=[];resources=[]
    for r in complete['records']:
        path=out/str(r['seed'])/r['arm']
        if _read(path/'result.json')!=r:raise RuntimeError('Result/completion mismatch')
        for name,digest in r['artifact_sha256'].items():
            if sha(path/name)!=digest:raise RuntimeError('Model artifact changed')
        with np.load(path/'prototypes.npz') as p,np.load(out/'data'/str(r['seed'])/'calibration.npz') as data:
            labels=data['targets'][:,-1];states=p['calibration_states'].astype(float);grams=states.transpose(0,2,1)@states
            means=np.stack([grams[labels==b].mean(0) for b in (0,1)])
            if not np.array_equal(labels,p['calibration_labels']) or not np.allclose(means,p['prototype_grams'],atol=1e-12,rtol=0):raise RuntimeError('Calibration mismatch')
            if float(((p['calibration_logits'][:,-1]>=0)==labels).mean())!=r['calibration']['calibration_accuracy']:raise RuntimeError('Calibration accuracy mismatch')
            if bool(np.array_equal(p['prototype_logits']>=0,[False,True]))!=r['calibration']['prototype_labels_correct']:raise RuntimeError('Prototype class mismatch')
        cr={k:r[k] for k in ('seed','arm','origin','calibration','resources')}
        for phase in ('qualification','evaluation'):
            cr[phase]={}
            for c,cells in r[phase].items():
                cr[phase][c]={}
                for cell,item in cells.items():
                    rawpath=path/f'qual_{cell}.npz' if phase=='qualification' else path/'test'/c/f'{cell}.npz'
                    dp=out/'data'/str(r['seed'])/f'{"qual" if phase=="qualification" else "test"}_{cell}.npz'
                    if sha(rawpath)!=item['artifact_sha256']:raise RuntimeError('Raw file hash mismatch')
                    with np.load(rawpath) as f,np.load(dp) as d:
                        raw={k:f[k] for k in f.files};metrics=scores(raw['logits'],d['targets'],d['commands']);diag=refresh_diagnostics(raw,c)
                    _same_numbers(metrics,item['metrics']);_same_numbers(diag,item['diagnostics'])
                    cr[phase][c][cell]={'metrics':[metrics[k] for k in METRIC_COLUMNS],
                                      'diagnostics':[diag[k] for k in DIAGNOSTIC_COLUMNS]}
                    resources.append([r['seed'],r['arm'],phase,c,cell,item['wall_seconds'],item['cpu_seconds'],item['transition_tokens'],item['refresh_calls']])
        compact.append(cr)
    result={'experiment_id':'EXP-0025','version':'v2',**decide(complete['records']),'models':compact,'metric_columns':METRIC_COLUMNS,'diagnostic_columns':DIAGNOSTIC_COLUMNS,
            'gate1_claim':False,'matching_criterion_version':'G1-MATCH-v3','source_sha256':manifest['source_sha256'],
            'manifest_sha256':sha(out/'manifest.json'),'completion_sha256':sha(out/'completion.json'),
            'scope':'12 prospective paired training seeds; frozen 23005 reference excluded from endpoints; supervised two-prototype codebook; conditional intervention family, no unique causal decomposition.',
            'resources':{k:complete[k] for k in ('research_updates','prototype_fits','qualification_count','evaluation_count','monitor_checkpoints','training_transition_tokens','calibration_transition_tokens','evaluation_transition_tokens')},
            'run_seconds':complete['wall_seconds'],'analysis_seconds':time.monotonic()-start}
    if result['run_seconds']+result['analysis_seconds']>1800:raise TimeoutError('Shared run+analysis budget exceeded')
    _save_compact(out/'resource_disclosure.json',{'columns':['seed','arm','phase','condition','cell','wall_seconds','cpu_seconds','transition_tokens','refresh_calls'],'rows':resources,'model_resources':[{k:r[k] for k in ('seed','arm','origin','resources')} for r in complete['records']],'run_wall_seconds':complete['wall_seconds'],'run_cpu_seconds':complete['cpu_seconds'],'executed_flops':'NOT_MEASURED','peak_memory':'NOT_MEASURED'})
    _save_compact(out/'analysis.json',result)
    if (out/'analysis.json').stat().st_size>250000:raise RuntimeError('Analysis packet exceeds frozen size budget')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=analyze(p.parse_args().run);print(json.dumps({k:a[k] for k in ('decision','substatus')}))
