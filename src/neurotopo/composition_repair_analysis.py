"""Prespecified six-seed prediction, consistency and repair decisions."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from .run import write_json,source_fingerprints
from .order_observation import sha

SEEDS=tuple(range(23001,23007))
ARMS=('REL_BASE','REL_CYCLE','REL_EXTRA','INV_BASE')
CELLS={'IID_H':(8,'HOLD',.4),'IID_U':(8,'UPDATE',.4),**{f'LONG{t}_{s}':(t,m,np.pi) for t in (64,128,256) for s,m in [('H','HOLD'),('U','UPDATE')]}}
LONG=tuple(k for k in CELLS if k.startswith('LONG'))


def scores(z: np.ndarray, target: np.ndarray, commands: np.ndarray) -> dict:
    z=z.astype(float);loss=np.logaddexp(0,z)-z*target;correct=(z>=0)==target
    if not np.isfinite(z).all():raise FloatingPointError('Nonfinite evaluation')
    return {'bce':float(loss.mean()),'future_bce':float(loss[:,1:].mean()),
            'accuracy_all':float(correct.mean()),'future_accuracy':float(correct[:,1:].mean()),
            'final_accuracy':float(correct[:,-1].mean()),'initial_accuracy':float(correct[:,0].mean()),
            'hold_accuracy':float(correct[:,1:][commands==0].mean()) if (commands==0).any() else None,
            'flip_accuracy':float(correct[:,1:][commands==1].mean()) if (commands==1).any() else None}


def summarize(values: list[float]) -> dict:
    x=np.asarray(values,float)
    if x.shape!=(6,) or not np.isfinite(x).all():raise ValueError('Six finite seed values required')
    mean=float(x.mean());sd=float(x.std(ddof=1));half=2.570581836*sd/np.sqrt(6)
    return {'values_by_seed':{str(s):float(v) for s,v in zip(SEEDS,x)},'mean':mean,'sample_sd':sd,'min':float(x.min()),'max':float(x.max()),'t95':[mean-half,mean+half]}


def average_ranks(values: list[float]) -> np.ndarray:
    a=np.asarray(values,float);order=np.argsort(a,kind='stable');ranks=np.empty(len(a),float);start=0
    while start<len(a):
        end=start+1
        while end<len(a) and a[order[end]]==a[order[start]]:end+=1
        ranks[order[start:end]]=(start+end-1)/2+1;start=end
    return ranks


def spearman(x: list[float], y: list[float]) -> float|None:
    a,b=average_ranks(x),average_ranks(y)
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def iid_pass(row: dict) -> bool:
    e=row['evaluation']
    return e['IID_H']['accuracy_all']>=.95 and e['IID_U']['accuracy_all']>=.95 and e['IID_U']['flip_accuracy']>=.95


def decide(rows: list[dict]) -> dict:
    by={(r['seed'],r['arm']):r for r in rows}
    if set(by)!={(s,a) for s in SEEDS for a in ARMS} or len(rows)!=24:raise ValueError('Incomplete model matrix')
    for r in rows:
        r['L']=float(np.mean([r['evaluation'][k]['future_bce'] for k in LONG]))
        r['A']=float(np.mean([r['evaluation'][k]['final_accuracy'] for k in LONG]))
        r['iid_pass']=iid_pass(r)
    base=[by[s,'REL_BASE'] for s in SEEDS];cycle=[by[s,'REL_CYCLE'] for s in SEEDS];extra=[by[s,'REL_EXTRA'] for s in SEEDS]
    rho=spearman([r['diagnostic']['D_rel'] for r in base],[r['L'] for r in base])
    p_learning=all(r['iid_pass'] and r['diagnostic']['current_accuracy']>=.95 for r in base)
    failed=sum(r['A']<.95 for r in base);success=6-failed
    p_status=('P_INCONCLUSIVE_LEARNING' if not p_learning else 'P_INCONCLUSIVE_RANGE' if min(failed,success)<2 else 'P_SUPPORTED_IN_SIX_SEEDS' if rho is not None and rho>=.6 else 'P_NOT_SUPPORTED')
    eligible=[r['diagnostic']['D']>1e-12 and r['diagnostic']['D_rel']>1e-12 for r in extra]
    improved=[ok and c['diagnostic']['D']<=.5*e['diagnostic']['D'] and c['diagnostic']['D_rel']<=.5*e['diagnostic']['D_rel'] for c,e,ok in zip(cycle,extra,eligible)]
    means_lower=all(np.mean([r['diagnostic'][k] for r in cycle])<np.mean([r['diagnostic'][k] for r in extra]) for k in ('D','D_rel'))
    m_status='M_INCONCLUSIVE_FLOOR' if sum(eligible)<4 else 'M_SUPPORTED' if sum(improved)>=4 and means_lower else 'M_NOT_SUPPORTED'
    d_extra=summarize([e['L']-c['L'] for e,c in zip(extra,cycle)])
    d_base=summarize([b['L']-c['L'] for b,c in zip(base,cycle)])
    ordinary=summarize([b['L']-e['L'] for b,e in zip(base,extra)])
    guard=sum(c['iid_pass'] and e['iid_pass'] for c,e in zip(cycle,extra))>=5
    repair=guard and d_extra['mean']>=.05 and d_extra['t95'][0]>0 and d_base['mean']>0
    p_pass=p_status=='P_SUPPORTED_IN_SIX_SEEDS';m_pass=m_status=='M_SUPPORTED'
    if not guard:decision,sub='INCONCLUSIVE','LEARNING_GUARD_FAILURE'
    elif repair and p_pass and m_pass:decision,sub='SUPPORTED','COMPOSITION_LINK_AND_REPAIR'
    elif repair:decision,sub='PARTIALLY_SUPPORTED','TRAINING_REPAIR_MECHANISM_UNRESOLVED'
    elif p_pass:decision,sub='PARTIALLY_SUPPORTED','PREDICTIVE_DIAGNOSTIC_ONLY'
    else:decision,sub='INCONCLUSIVE','NO_REGISTERED_REPAIR_OR_LINK'
    return {'decision':decision,'substatus':sub,'P':{'status':p_status,'rho':rho,'base_iid_and_prefix_qualified':p_learning,'long_failure_seeds':failed,'long_success_seeds':success},
            'M':{'status':m_status,'eligible_by_seed':dict(zip(map(str,SEEDS),eligible)),'improved_by_seed':dict(zip(map(str,SEEDS),improved)),'all_seed_means_lower':bool(means_lower)},
            'R':{'supported':bool(repair),'paired_learning_guard':guard,'d_EXTRA':d_extra,'d_BASE':d_base,'BASE_minus_EXTRA':ordinary},
            'control_error_headroom_bound':float(np.mean([r['L'] for r in extra]))}


def analyze(out: Path) -> dict:
    started=time.monotonic()
    fail=out/'worker_failure.json'
    if fail.exists() or (out/'run_failure.json').exists():
        detail=json.loads((out/'run_failure.json').read_text()) if (out/'run_failure.json').exists() else {}
        worker=json.loads(fail.read_text()) if fail.exists() else {}
        result={'experiment_id':'EXP-0023','version':'v2','decision':'INCONCLUSIVE','substatus':'INCOMPLETE_BUDGET' if 'TimeoutError' in detail.get('error','') else 'ENGINEERING_ABORT','detail':detail,'worker':worker,'gate1_claim':False}
        write_json(out/'analysis.json',result);return result
    manifest=json.loads((out/'manifest.json').read_text());complete=json.loads((out/'completion.json').read_text())
    assert manifest['source_sha256']==source_fingerprints()
    for field in ('frozen_hashes','parent_artifact_sha256'):
        for name,digest in manifest[field].items():assert sha(Path(name))==digest
    for name,digest in complete['data_files'].items():assert sha(out/name)==digest
    rows=[]
    for rec in complete['records']:
        seed,arm=rec['seed'],rec['variant'];path=out/str(seed)/arm
        for name,digest in rec['artifact_sha256'].items():assert sha(path/name)==digest
        full=json.loads((path/'result.json').read_text())
        assert {k:v for k,v in full.items() if k!='losses'}==rec
        assert full['updates']==1000 and len(full['losses'])==1000
        assert [c['step'] for c in full['curves']]==[0,125,250,500,1000]
        for c in full['curves']:
            assert sha(path/f"checkpoint-{c['step']}.pt")==c['checkpoint_sha256']
            assert sha(path/f"monitor-{c['step']}.npz")==c['prediction_sha256']
        ev={}
        with np.load(path/'predictions.npz') as pred:
            for cell in CELLS:
                with np.load(out/'data'/str(seed)/f'{cell}.npz') as d:
                    ev[cell]=scores(pred[cell],d['targets'],d['commands'])
        from .composition_core import diagnostic_summary
        with np.load(path/'diagnostic.npz') as d:diag=diagnostic_summary({k:d[k] for k in d.files})
        for k,v in diag.items():assert abs(v-full['diagnostic'][k])<1e-10
        rows.append({'seed':seed,'arm':arm,'evaluation':ev,'diagnostic':diag,'resources':full['resources'],'training_seconds':full['training_monitor_seconds']})
    for seed in SEEDS:
        group=[r for r in complete['records'] if r['seed']==seed]
        assert len({r['training_data_sha256'] for r in group})==1
        assert len({r['initial_hashes']['encoder'] for r in group})==len({r['initial_hashes']['readout'] for r in group})==1
        assert len({r['initial_hashes']['all'] for r in group if r['variant']!='INV_BASE'})==1
    old=[]
    from .composition_core import diagnostic_summary
    assert len(complete['old_records'])==15
    for r in complete['old_records']:
        path=out/'old'/str(r['seed'])/r['arm']
        for name,digest in r['artifact_sha256'].items():assert sha(path/name)==digest
        assert sha(path.parent/'data'/'diagnostic.npz')==r['diagnostic_data_sha256']
        assert json.loads((path/'result.json').read_text())==r
        with np.load(path/'diagnostic.npz') as d:diag=diagnostic_summary({k:d[k] for k in d.files})
        for k,v in diag.items():assert abs(v-r['diagnostic'][k])<1e-10
        old.append({'seed':r['seed'],'arm':r['arm'],'diagnostic':diag,'parent_long_final_accuracy':{k:r['parent_evaluation'][k]['final_accuracy'] for k in ('LONG_H','LONG_U')},'scope':'Descriptive exposed-parent checkpoints only'})
    assert complete['research_updates']==24000 and len(rows)==24
    verdict=decide(rows)
    result={'experiment_id':'EXP-0023','version':'v2',**verdict,'models':rows,'old_diagnostics':old,'research_updates':24000,'gate1_claim':False,'matching_criterion_version':'G1-MATCH-v3','source_sha256':manifest['source_sha256'],'manifest_sha256':sha(out/'manifest.json'),'completion_sha256':sha(out/'completion.json'),'scope':'One coefficient and six new paired training seeds; old checkpoints descriptive only; prediction, marker change and repair distinct.'}
    result['run_seconds']=complete['wall_seconds']
    result['analysis_seconds']=time.monotonic()-started
    assert result['run_seconds']+result['analysis_seconds']<=1200
    write_json(out/'analysis.json',result)
    if (out/'analysis.json').stat().st_size>250000:raise ValueError('Compact analysis exceeds reflection input cap')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=analyze(a.run);print(json.dumps({k:r[k] for k in ('experiment_id','decision','substatus')}))
