"""Single bounded all-checkpoint replay; independent reductions, no training."""
from __future__ import annotations
import argparse,json,signal,time,math,statistics
from pathlib import Path
import numpy as np
import torch
from .composition_core import predict_full,diagnostic
from .composition_repair import SEEDS,ARMS,CELLS,OLD_SEEDS,OLD_ARMS
from .learned_protected_pilot import LearnedMemory,generate,state_hash
from .run import source_fingerprints,write_json
from .order_observation import sha


def score(z: np.ndarray,d: dict) -> dict:
    target=np.column_stack((d['b0'],(d['b0'][:,None]+np.cumsum(d['commands'],1))%2))
    np.testing.assert_array_equal(target,d['targets'])
    p=z.astype(float);loss=np.maximum(p,0)-p*target+np.log1p(np.exp(-np.abs(p)));ok=(p>=0)==target;u=d['commands']
    return {'bce':float(loss.mean()),'future_bce':float(loss[:,1:].mean()),'accuracy_all':float(ok.mean()),'future_accuracy':float(ok[:,1:].mean()),'final_accuracy':float(ok[:,-1].mean()),'initial_accuracy':float(ok[:,0].mean()),'hold_accuracy':float(ok[:,1:][u==0].mean()) if (u==0).any() else None,'flip_accuracy':float(ok[:,1:][u==1].mean()) if (u==1).any() else None}


def check_score(actual: dict, saved: dict) -> None:
    for key,v in actual.items():
        if key not in saved:continue
        if v is None:assert saved[key] is None
        else:assert abs(v-saved[key])<1e-10,(key,v,saved[key])


def checked_data(path: Path, regenerated: dict) -> dict:
    with np.load(path) as saved:
        for key,v in regenerated.items():np.testing.assert_array_equal(saved[key],v)
    return regenerated


def diagnose(model: LearnedMemory,data: dict,path: Path,summary: dict) -> tuple[dict,float]:
    raw,_=diagnostic(model,data);error=0.
    with np.load(path) as saved:
        for key,value in raw.items():
            difference=float(np.max(np.abs(value-saved[key])));assert difference<=1e-6
            error=max(error,difference)
    z=raw['z0'].astype(float);dh=raw['z_hold'].astype(float)-z;df=raw['z_flip2'].astype(float)-z
    d=float(np.sum(dh*dh+df*df)/(2*len(z)));relative=d/(float(np.dot(z,z)/len(z))+1e-6)
    assert abs(d-summary['D'])<1e-10 and abs(relative-summary['D_rel'])<1e-10
    correct=float(np.mean((z>=0)==data['targets'][:,-1]));assert correct==summary['current_accuracy']
    return {'D':d,'D_rel':relative,'current_accuracy':correct},error


def rank_correlation(x: list[float],y: list[float]) -> float|None:
    def rank(a):return np.array([1+sum(v<t for v in a)+(sum(v==t for v in a)-1)/2 for t in a],float)
    a,b=rank(x),rank(y);a-=a.mean();b-=b.mean();den=float(np.sqrt(np.dot(a,a)*np.dot(b,b)))
    return None if den==0 else float(np.dot(a,b)/den)


def replay(run: Path,output: Path) -> dict:
    start=time.monotonic()
    def abort(*_):raise TimeoutError('300-second single replay cap')
    signal.signal(signal.SIGALRM,abort);signal.setitimer(signal.ITIMER_REAL,300)
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        manifest=json.loads((run/'manifest.json').read_text());complete=json.loads((run/'completion.json').read_text());analysis=json.loads((run/'analysis.json').read_text())
        assert manifest['source_sha256']==source_fingerprints()
        for key in ('frozen_hashes','parent_artifact_sha256'):
            for name,digest in manifest[key].items():assert sha(Path(name))==digest
        for name,digest in complete['data_files'].items():assert sha(run/name)==digest
        max_error=0.;old_count=0;eval_count=0;cp_count=0;new=[]
        for seed in OLD_SEEDS:
            data=checked_data(run/'old'/str(seed)/'data'/'diagnostic.npz',generate(128,8,[234001,seed,8,0],'MIX'))
            for arm in OLD_ARMS:
                path=run/'old'/str(seed)/arm;record=json.loads((path/'result.json').read_text())
                for name,digest in record['artifact_sha256'].items():assert sha(path/name)==digest
                m=LearnedMemory(arm,seed);m.load_state_dict(torch.load(path/'checkpoint-1000.pt',weights_only=True,map_location='cpu')['model']);before=state_hash(m)
                _,e=diagnose(m,data,path/'diagnostic.npz',record['diagnostic']);max_error=max(max_error,e);old_count+=1;assert state_hash(m)==before
        for seed in SEEDS:
            base=run/'data'/str(seed)
            monitor=checked_data(base/'monitor.npz',generate(128,8,[236001,seed,8,0],'MIX'))
            diagdata=checked_data(base/'diagnostic.npz',generate(128,8,[234001,seed,8,0],'MIX'))
            datasets={k:checked_data(base/f'{k}.npz',generate(512,t,[233001,seed,t,1 if mode=='HOLD' else 2],mode,amp,0.)) for k,(t,mode,amp) in CELLS.items()}
            for variant in ARMS:
                path=run/str(seed)/variant;record=json.loads((path/'result.json').read_text());arm='INV' if variant=='INV_BASE' else 'REL';m=LearnedMemory(arm,seed)
                assert record['updates']==len(record['losses'])==1000
                for name,digest in record['artifact_sha256'].items():assert sha(path/name)==digest
                for curve in record['curves']:
                    cp=path/f"checkpoint-{curve['step']}.pt";assert sha(cp)==curve['checkpoint_sha256']
                    m.load_state_dict(torch.load(cp,weights_only=True,map_location='cpu')['model']);z,g,_=predict_full(m,monitor)
                    pred=path/f"monitor-{curve['step']}.npz";assert sha(pred)==curve['prediction_sha256']
                    with np.load(pred) as f:
                        e=max(float(np.max(np.abs(z-f['logits']))),float(np.max(np.abs(g-f['gram']))));assert e<=1e-6;max_error=max(max_error,e)
                    check_score(score(z,monitor),curve['metrics']);cp_count+=1
                before=state_hash(m);evaluations={}
                with np.load(path/'predictions.npz') as f:
                    for cell,data in datasets.items():
                        z,g,_=predict_full(m,data);e=max(float(np.abs(z-f[cell]).max()),float(np.abs(g-f[cell+'_gram']).max()));assert e<=1e-6;max_error=max(max_error,e)
                        evaluations[cell]=score(z,data);check_score(evaluations[cell],record['evaluation'][cell]);eval_count+=1
                d,e=diagnose(m,diagdata,path/'diagnostic.npz',record['diagnostic']);max_error=max(max_error,e);assert state_hash(m)==before
                long=[k for k in CELLS if k.startswith('LONG')]
                passed=evaluations['IID_H']['accuracy_all']>=.95 and evaluations['IID_U']['accuracy_all']>=.95 and evaluations['IID_U']['flip_accuracy']>=.95
                new.append({'seed':seed,'arm':variant,'L':statistics.mean(evaluations[k]['future_bce'] for k in long),'A':statistics.mean(evaluations[k]['final_accuracy'] for k in long),'iid_pass':passed,'diagnostic':d})
        by={(r['seed'],r['arm']):r for r in new};b=[by[s,'REL_BASE'] for s in SEEDS];c=[by[s,'REL_CYCLE'] for s in SEEDS];e=[by[s,'REL_EXTRA'] for s in SEEDS]
        rho=rank_correlation([r['diagnostic']['D_rel'] for r in b],[r['L'] for r in b])
        p_learning=all(r['iid_pass'] and r['diagnostic']['current_accuracy']>=.95 for r in b)
        failures=sum(r['A']<.95 for r in b)
        ps='P_INCONCLUSIVE_LEARNING' if not p_learning else 'P_INCONCLUSIVE_RANGE' if min(failures,6-failures)<2 else 'P_SUPPORTED_IN_SIX_SEEDS' if rho is not None and rho>=.6 else 'P_NOT_SUPPORTED'
        eligible=[r['diagnostic']['D']>1e-12 and r['diagnostic']['D_rel']>1e-12 for r in e]
        improved=[ok and all(x['diagnostic'][k]<=.5*y['diagnostic'][k] for k in ('D','D_rel')) for x,y,ok in zip(c,e,eligible)]
        lower=all(statistics.mean(r['diagnostic'][k] for r in c)<statistics.mean(r['diagnostic'][k] for r in e) for k in ('D','D_rel'))
        ms='M_INCONCLUSIVE_FLOOR' if sum(eligible)<4 else 'M_SUPPORTED' if sum(improved)>=4 and lower else 'M_NOT_SUPPORTED'
        delta=[y['L']-x['L'] for x,y in zip(c,e)];delta_base=[y['L']-x['L'] for x,y in zip(c,b)];mean=statistics.mean(delta);sd=statistics.stdev(delta);ci=[mean-2.570581836*sd/math.sqrt(6),mean+2.570581836*sd/math.sqrt(6)]
        guard=sum(x['iid_pass'] and y['iid_pass'] for x,y in zip(c,e))>=5
        repair=guard and mean>=.05 and ci[0]>0 and statistics.mean(delta_base)>0
        decision='INCONCLUSIVE' if not guard else 'SUPPORTED' if repair and ps=='P_SUPPORTED_IN_SIX_SEEDS' and ms=='M_SUPPORTED' else 'PARTIALLY_SUPPORTED' if repair or ps=='P_SUPPORTED_IN_SIX_SEEDS' else 'INCONCLUSIVE'
        assert decision==analysis['decision'] and ps==analysis['P']['status'] and ms==analysis['M']['status'] and repair==analysis['R']['supported']
        np.testing.assert_allclose(ci,analysis['R']['d_EXTRA']['t95'],atol=1e-10,rtol=0)
        if rho is None:assert analysis['P']['rho'] is None
        else:assert abs(rho-analysis['P']['rho'])<1e-12
        assert complete['research_updates']==24000 and old_count==15 and eval_count==192 and cp_count==120
        result={'pass':True,'decision':decision,'P_status':ps,'M_status':ms,'R_supported':repair,'independent_rho':rho,'d_EXTRA_mean':mean,'d_EXTRA_t95':ci,'parent_models':old_count,'new_models':24,'monitor_checkpoints':cp_count,'evaluation_cells':eval_count,'new_diagnostics':24,'max_error':max_error,'replay_training_updates':0,'seconds':time.monotonic()-start,'scope':'Frozen shared model/core replay plus independently reduced targets/losses/ranks/paired statistics/P-M-R; no retraining.'}
        write_json(output,result);write_json(output.with_name(output.stem+'_seed_metrics.json'),{'rows':new});return result
    except BaseException as exc:
        write_json(output.with_name(output.stem+'_failure.json'),{'error':repr(exc),'seconds':time.monotonic()-start});raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(replay(a.run,a.output)))
