"""Frozen EXP0026 endpoint reduction and artifact audit."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .schedule_core import MAIN,EXTRA,PERIODS,attempt_mask,count_schedule,lower_tau
from .refresh_factor_core import weighted
from .order_observation import sha
from .run import source_fingerprints

QUALIFIED=(25001,25002,25003,25004,25008,25009,25010,25011,25012)
HORIZONS=(256,512,1024)
METRIC_COLUMNS=('final_256','final_512','final_1024','attempts','valid','valid_fraction','valid_logit_error_max','gauge_logit_max','gauge_gram_max')


def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,separators=(',',':'),allow_nan=False);f.write('\n')


def window(bits:list[bool])->tuple[str,int|None]:
    if len(bits)!=6:raise ValueError('Six k required')
    where=np.flatnonzero(bits)
    if not len(where):return 'NO_WINDOW',None
    if len(where)==6:return 'RIGHT_CENSORED_64',64
    contiguous=np.array_equal(where,np.arange(where[0],where[-1]+1))
    if not contiguous:return 'NONMONOTONIC_WINDOW',None
    if where[0]==0:return 'PREFIX_WINDOW',PERIODS[where[-1]]
    if where[-1]==5:return 'SUFFIX_WINDOW_RIGHT_CENSORED',None
    return 'BOUNDED_WINDOW',None


def summarize(raw:dict,data:dict,condition:str,theta:float)->dict:
    z=raw['logits'];target=data['targets'];n,length=data['commands'].shape
    if z.shape!=(n,length+1) or not np.isfinite(z).all():raise RuntimeError('Logit shape/nonfinite')
    correct=(z>=0)==target
    horizons=tuple(t for t in HORIZONS if t<=length)
    ids,ts=raw['event_i'],raw['event_t'];valid=raw['event_valid'];reason=raw['event_reason']
    if not np.array_equal(valid,reason==0):raise RuntimeError('Event validity mismatch')
    mask=np.zeros((n,length+1),bool);mask[ids,ts]=True
    if mask.sum()!=len(ids) or mask[:,0].any():raise RuntimeError('Duplicate/t0 event')
    counts=np.bincount(ids,minlength=n)
    if not np.array_equal(counts,raw['attempt_counts']):raise RuntimeError('Event count mismatch')
    cm=count_schedule(raw['source_adapt_counts'],length) if condition=='COUNT90' else None
    last=np.full(n,-100000,np.int64)
    for t in range(1,length+1):
        expected=attempt_mask(condition,t,raw['distance_pre'][:,t],theta,last,cm)
        if not np.array_equal(mask[:,t],expected):raise RuntimeError('Schedule mismatch '+condition)
        last[expected]=t
    if not np.array_equal(raw['event_pre_z'],z[ids,ts]):raise RuntimeError('Pre-refresh score mismatch')
    if condition not in ('NONE','GAUGE8','FULL8','SCALE8'):
        if valid.any() and (np.max(np.abs(raw['event_post_z'][valid]-raw['event_pre_z'][valid]))>1e-5 or np.any((raw['event_post_z'][valid]>=0)!=(raw['event_pre_z'][valid]>=0))):
            raise RuntimeError('Equal-logit/sign constraint failed')
    if condition not in ('NONE','GAUGE8','FULL8'):
        bad=~valid
        if not np.array_equal(raw['event_post_g'][bad],raw['event_pre_g'][bad]) or not np.array_equal(raw['event_post_z'][bad],raw['event_pre_z'][bad]) or np.any(raw['event_dh'][bad]!=0):raise RuntimeError('Invalid fallback differs')
    tau,censored=lower_tau(raw['distance_pre'],theta)
    if not np.array_equal(tau,raw['tau_lag']) or not np.array_equal(censored,raw['tau_censored']):raise RuntimeError('Tau mismatch')
    lower=int(np.sort(tau)[n//2-1]);sentinel=length-7
    loss=np.logaddexp(0,z.astype(float))-z.astype(float)*target
    metrics={f'final_{t}':float(correct[:,t].mean()) for t in horizons}
    metrics.update(accuracy_all=float(correct.mean()),bce=float(loss.mean()),attempts=len(ids),valid=int(valid.sum()),
        valid_fraction=float(valid.mean()) if len(ids) else None,
        valid_logit_error_max=float(raw['event_target_error'][valid].max()) if valid.any() else None,
        gauge_logit_max=float(np.max(np.abs(raw['event_post_z']-raw['event_pre_z']))) if condition=='GAUGE8' and len(ids) else None,
        gauge_gram_max=float(np.max(np.abs(raw['event_post_g']-raw['event_pre_g']))) if condition=='GAUGE8' and len(ids) else None,
        first_error_count=int((~correct).any(1).sum()))
    return {'metrics':metrics,'terminal_correct_counts':[int(correct[:,t].sum()) for t in horizons],
        'denominator':n,'invalid_reason_counts':{str(i):int((reason==i).sum()) for i in range(1,8)},
        'tau':None if lower==sentinel else lower,'tau_status':'TAU_NOT_REACHED' if lower==sentinel else 'REACHED',
        'tau_crossings':int((~censored).sum()),'tau_censored':int(censored.sum()),
        'prediction_class_switch_rate':float(np.mean((z[:,1:]>=0)!=(z[:,:-1]>=0))),
        'event_state_delta_mean':float(raw['event_dh'].mean()) if len(ids) else None,
        'event_gram_delta_mean':float(raw['event_dg'].mean()) if len(ids) else None}


def decisions(rows:list[dict])->dict:
    lookup={(r['seed'],r['arm'],r['cell'],r['rotation'],r['condition']):r['summary']['metrics'] for r in rows if r['phase']=='main'}
    expected={(s,a,c,r,k) for s in (*range(25001,25013),23005) for a in ('REL','INV') for c in ('LONG_H','LONG_U') for r in ('ORIGINAL','IDENTITY') for k in (MAIN+EXTRA if r=='ORIGINAL' else MAIN)}
    if set(lookup)!=expected:raise RuntimeError('Incomplete decision matrix')
    info=[]
    for seed in range(25001,25013):
        for rotation in ('ORIGINAL','IDENTITY'):
            def get(c,task):return lookup[seed,'REL',task,rotation,c]
            failure=seed in QUALIFIED and all(get('NONE','LONG_H')[f'final_{t}']<.95 for t in HORIZONS)
            gp=None
            if rotation=='ORIGINAL':
                gp=all(get('GAUGE8',cell)['attempts']>0 and get('GAUGE8',cell)['gauge_logit_max']<=1e-5 and get('GAUGE8',cell)['gauge_gram_max']<=1e-5 and all(abs(get('GAUGE8',cell)[f'final_{t}']-get('NONE',cell)[f'final_{t}'])<=.005 for t in HORIZONS) for cell in ('LONG_H','LONG_U'))
            passes=[]
            for k in PERIODS:
                h,u=get('K'+str(k),'LONG_H'),get('K'+str(k),'LONG_U')
                passes.append(all(h[f'final_{t}']>=.95 and h[f'final_{t}']-get('NONE','LONG_H')[f'final_{t}']>=.10 and u[f'final_{t}']>=.95 and u[f'final_{t}']-get('NONE','LONG_U')[f'final_{t}']>=-.02 for t in HORIZONS) and all(m['valid_fraction']>=.99 and m['valid_logit_error_max'] is not None and m['valid_logit_error_max']<=1e-5 for m in (h,u)))
            label,upper=window(passes)
            if not failure:label,upper='BASELINE_NOT_FAILURE',None
            elif rotation=='ORIGINAL' and not gp:label,upper='CONTROL_INCONCLUSIVE',None
            info.append({'seed':seed,'rotation':rotation,'previously_qualified':seed in QUALIFIED,'conditional_failure':failure,'gauge_pass':gp,'k_success':passes,'window':label,'k_upper_observed':upper})
    supported={'PREFIX_WINDOW','BOUNDED_WINDOW','RIGHT_CENSORED_64','SUFFIX_WINDOW_RIGHT_CENSORED'}
    found=[r['seed'] for r in info if r['rotation']=='ORIGINAL' and r['window'] in supported]
    controls=[r['seed'] for r in info if r['rotation']=='ORIGINAL' and r['window']=='CONTROL_INCONCLUSIVE']
    sub='MULTIPLE_EXPOSED_SCHEDULE_WINDOWS' if len(found)>=2 else 'SINGLE_EXPOSED_SCHEDULE_WINDOW' if found else 'CONTROL_INCONCLUSIVE' if controls else 'NO_REGISTERED_SCHEDULE_WINDOW'
    return {'decision':'PARTIALLY_SUPPORTED' if found else 'INCONCLUSIVE','substatus':sub,'window_seeds':found,'control_inconclusive_seeds':controls,'seed_windows':info}


def diagnostic_outputs(out:Path,rows:list[dict],parent:Path)->tuple[list,list]:
    lookup={(r['seed'],r['arm'],r['cell'],r['rotation'],r['condition']):r for r in rows if r['phase']=='main'}
    pairs=[];geometry=[]
    for seed in (*range(25001,25013),23005):
        for arm in ('REL','INV'):
            state=torch.load(parent/str(seed)/arm/'checkpoint-1000.pt',map_location='cpu',weights_only=True)
            normal=state['model']['readout.weight'].numpy()[0].astype(float);normal[1]/=np.sqrt(2.)
            nn=float(np.linalg.norm(normal))
            for cell in ('LONG_H','LONG_U'):
                records=[lookup[seed,arm,cell,r,'NONE'] for r in ('ORIGINAL','IDENTITY')]
                with np.load(out/records[0]['raw_path']) as a,np.load(out/records[1]['raw_path']) as b:
                    delta=weighted(a['gram'].astype(float)-b['gram'].astype(float));norm=np.linalg.norm(delta,axis=-1)
                    zdelta=np.abs(a['logits'].astype(float)-b['logits'].astype(float))
                    pairs.append([seed,arm,cell,float(norm.max()),float(norm.mean()),float(np.quantile(norm,.9)),float(zdelta.max()),float(zdelta.mean()),
                                  *[records[0]['summary']['metrics'][f'final_{t}']-records[1]['summary']['metrics'][f'final_{t}'] for t in HORIZONS]])
                    folder=out/'paired_curves'/str(seed)/arm;folder.mkdir(parents=True,exist_ok=True)
                    with (folder/(cell+'.npz')).open('xb') as f:
                        np.savez_compressed(f,gram_distance_mean=norm.mean(0),gram_distance_q90=np.quantile(norm,.9,axis=0),logit_distance_mean=zdelta.mean(0))
                for rotation,row in zip(('ORIGINAL','IDENTITY'),records):
                    with np.load(out/row['raw_path']) as f:
                        command=weighted(f['command_flip_gram'].astype(float)-f['command_hold_gram'].astype(float))
                        removed=-weighted(f['command_reset_delta'].astype(float));cn=np.linalg.norm(command,axis=-1);rn=np.linalg.norm(removed,axis=-1)
                        valid=f['command_reset_valid'] & (cn>=1e-6)&(rn>=1e-6)&(nn>=1e-6)
                        cosine=np.sum(command*removed,axis=-1)[valid]/(cn[valid]*rn[valid])
                        projection=removed-np.sum(removed*normal,axis=-1)[...,None]*normal/(nn*nn) if nn>=1e-6 else None
                        geometry.append([seed,arm,cell,rotation,int(valid.sum()),int(valid.size),float(np.mean(cosine)) if len(cosine) else None,
                                         float(np.median(cosine)) if len(cosine) else None,float(cn.mean()),float(rn.mean()),
                                         float(np.linalg.norm(projection[valid],axis=-1).mean()) if valid.any() and projection is not None else None])
    return pairs,geometry


def group_effects(rows:list[dict])->dict:
    lookup={(r['seed'],r['arm'],r['cell'],r['rotation'],r['condition']):r['summary']['metrics'] for r in rows if r['phase']=='main'}
    result={}
    for arm in ('REL','INV'):
        result[arm]={}
        for rotation in ('ORIGINAL','IDENTITY'):
            result[arm][rotation]={}
            for condition in (MAIN+EXTRA if rotation=='ORIGINAL' else MAIN):
                result[arm][rotation][condition]={}
                for horizon in HORIZONS:
                    x=np.array([np.mean([lookup[s,arm,cell,rotation,condition][f'final_{horizon}']-lookup[s,arm,cell,rotation,'NONE'][f'final_{horizon}'] for cell in ('LONG_H','LONG_U')]) for s in range(25001,25013)])
                    mean=float(x.mean());sd=float(x.std(ddof=1));half=2.200985160*sd/np.sqrt(12)
                    result[arm][rotation][condition][str(horizon)]={'values':x.tolist(),'mean':mean,'sample_sd':sd,'min':float(x.min()),'max':float(x.max()),'descriptive_t95':[mean-half,mean+half]}
    return result


def analyze(out:Path)->dict:
    start=time.monotonic()
    if (out/'failure.json').exists() or (out/'worker_failure.json').exists():
        failures={p.name:read(p) for p in (out/'failure.json',out/'worker_failure.json') if p.exists()}
        result={'experiment_id':'EXP-0026','version':'v2','decision':'INCONCLUSIVE','substatus':'INCOMPLETE_BUDGET' if 'TimeoutError' in str(failures) else 'ENGINEERING_ABORT','failures':failures,'gate1_claim':False}
        save(out/'analysis.json',result);return result
    m,c=read(out/'manifest.json'),read(out/'completion.json')
    if m['source_sha256']!=source_fingerprints():raise RuntimeError('Source changed')
    for path,h in m['frozen_hashes'].items():
        if sha(Path(path))!=h:raise RuntimeError('Frozen artifact changed '+path)
    if c['main_rollouts']!=1196 or c['tau_rollouts']!=104 or c['base_transitions']!=681574400 or c['command_transitions']!=851968:raise RuntimeError('Wrong resource counts')
    compact=[];resources=[];tau=[];checked_rows=[]
    for row in c['rows']:
        path=out/row['raw_path'];dp=Path(row['input_path'])
        if sha(path)!=row['raw_sha256'] or sha(dp)!=row['input_sha256']:raise RuntimeError('Raw/input hash mismatch')
        with np.load(path) as f,np.load(dp) as d:
            raw={k:f[k] for k in f.files};data={k:d[k] for k in d.files}
            checked=summarize(raw,data,row['condition'],row['theta'])
        if checked!=row['summary']:raise RuntimeError('Raw summary mismatch')
        checked_rows.append(row)
        ident=[row[k] for k in ('seed','arm','cell','rotation','condition')]
        compact_ident=[row['seed'],('REL','INV').index(row['arm']),('LONG_H','LONG_U').index(row['cell']),('ORIGINAL','IDENTITY').index(row['rotation']),(MAIN+EXTRA).index(row['condition'])]
        resources.append([row['phase'],*ident,row['wall_seconds'],row['cpu_seconds'],row['summary']['metrics']['attempts'],row['summary']['metrics']['valid']])
        if row['phase']=='tau':tau.append([*ident[:4],checked['tau'],checked['tau_censored'],checked['prediction_class_switch_rate']])
        else:compact.append([*compact_ident,*[checked['metrics'][k] for k in METRIC_COLUMNS]])
    verdict=decisions(checked_rows)
    pairs,geometry=diagnostic_outputs(out,checked_rows,Path(m['config']['parent']))
    effects=group_effects(checked_rows)
    save(out/'descriptive_effects.json',effects)
    save(out/'geometry_summary.json',{'pair_columns':['seed','arm','cell','gram_max','gram_mean','gram_q90','logit_max','logit_mean','accuracy_delta256','accuracy_delta512','accuracy_delta1024'],'paired_R_I':pairs,'geometry_columns':['seed','arm','cell','rotation','valid_overlap','total_overlap','cosine_mean','cosine_median','command_norm_mean','removed_norm_mean','removed_kernel_norm_mean'],'command_reset_geometry':geometry})
    result={'experiment_id':'EXP-0026','version':'v2',**verdict,'gate1_claim':False,'scope':'Exposed frozen models; conditional same-data failure strata; no independent seed/slow-mode claim.',
            'category_order':{'arm':['REL','INV'],'cell':['LONG_H','LONG_U'],'rotation':['ORIGINAL','IDENTITY'],'condition':list(MAIN+EXTRA)},
            'metric_columns':['seed','arm','cell','rotation','condition',*METRIC_COLUMNS],'rows':compact,
            'tau_columns':['seed','arm','cell','rotation','tau_lower_median','censored_count','class_switch_rate'],'tau_rows':tau,
            'resource':{k:c[k] for k in ('main_rollouts','tau_rollouts','base_transitions','command_transitions','diagnostic_refresh_proposals','refresh_attempts','refresh_valid','wall_seconds')},
            'analysis_seconds':time.monotonic()-start,'source_sha256':m['source_sha256'],'manifest_sha256':sha(out/'manifest.json'),'completion_sha256':sha(out/'completion.json'),'effects_sha256':sha(out/'descriptive_effects.json'),'geometry_sha256':sha(out/'geometry_summary.json')}
    if result['resource']['wall_seconds']+result['analysis_seconds']>3600:raise TimeoutError('Shared budget exceeded')
    save(out/'resource_disclosure.json',{'columns':['phase','seed','arm','cell','rotation','condition','wall_seconds','cpu_seconds','attempts','valid'],'rows':resources,'model_resources':c['model_resources'],'flops':'NOT_MEASURED','peak_memory':'NOT_MEASURED'})
    save(out/'analysis.json',result)
    if (out/'analysis.json').stat().st_size>250000:raise RuntimeError('Compact packet exceeds250000')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);r=analyze(p.parse_args().run);print(json.dumps({k:r[k] for k in ('decision','substatus')}))
