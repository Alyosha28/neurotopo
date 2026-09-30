"""Replay frozen EXP-0016 checkpoints; independently recompute oracle, scores and decision."""
from __future__ import annotations
import argparse,hashlib,json,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .orbit_control_pilot import build_model,sha,tensor_hash
from .orbit_information_reproduce import filter_matrix


def require(condition:bool,message:str)->None:
    if not condition:raise ValueError(message)


def audit_source(m:dict,source:Path)->None:
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    require(commit==m['git_commit'],'Source commit mismatch')
    require(not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip(),'Dirty source')
    actual={str(p.relative_to(source)):sha(p) for p in sorted((source/'src').rglob('*.py')) if not p.name.startswith('._')}
    require(actual==m['source_sha256'],'Source hash mismatch')
    for relative,expected in {**actual,**m['frozen_hashes']}.items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'Unsafe path')
        require(sha(source/relative)==expected,'Frozen file mismatch')
        blob=subprocess.check_output(['git','-C',str(source),'show',f'{commit}:{relative}'])
        require(hashlib.sha256(blob).hexdigest()==expected,'Source differs from committed bytes')
    validation=source/'artifacts/validation/suite.json';v=json.loads(validation.read_text())
    require(sha(validation)==m['validation_sha256'] and v['pass'] and v['source_sha256']==actual,'Validation mismatch')
    preflight=source/'artifacts/validation/exp0016_preflight.json'
    require(sha(preflight)==m['preflight_sha256'] and json.loads(preflight.read_text())['pass_'],'Preflight mismatch')
    require(sha(source/'configs/exp0016_v1.json')==m['config_sha256'],'Config hash mismatch')
    require(json.loads((source/'configs/exp0016_v1.json').read_text())==m['config'],'Config content mismatch')
    require(sys.version==m['python'] and np.__version__==m['numpy'] and torch.__version__==m['torch'],'Runtime mismatch')
    require(sha(Path(sys.executable).resolve())==m['executable_sha256'],'Interpreter mismatch')
    require(actual['src/neurotopo/orbit_control_reproduce.py']==sha(Path(__file__)),'Auditor differs from frozen source')


def decision(rows:list[dict])->tuple[str,str]:
    require(len(rows)==len({(r['arm'],r['seed']) for r in rows}),'Duplicate arm/seed')
    hs=[r for r in rows if r['arm']=='H']
    require(sorted(r['seed'] for r in hs)==[16001,16002,16003],'Missing H seeds')
    ceiling=all(r['metrics']['32']['kl_mean']<.1 and r['metrics']['64']['ci'][1]<.02 for r in hs)
    if ceiling:
        require(len(rows)==3,'GRU training after ceiling')
        return 'INCONCLUSIVE','SIMPLE_ALGORITHM_CEILING'
    require(len(rows)==12 and {(r['arm'],r['seed']) for r in rows}=={(a,s) for a in ['H','B','U','D'] for s in [16001,16002,16003]},'Incomplete GRU phase')
    for arm in ['B','U','D']:
        iid=[r['metrics']['32']['kl_mean'] for r in rows if r['arm']==arm]
        if max(iid)>=.15 or np.mean(iid)>=.1:return 'INCONCLUSIVE','LEARNING_FAILURE'
    for arm in ['B','U','D']:
        ood=[r['metrics']['64']['kl_mean'] for r in rows if r['arm']==arm]
        if min(ood)<.04 or np.mean(ood)<.08:return 'INCONCLUSIVE','CONTROL_CEILING'
    for arm in ['B','U','D']:
        ood=[r['metrics']['64']['kl_mean'] for r in rows if r['arm']==arm]
        if max(ood)>1 or np.mean(ood)>.8:return 'INCONCLUSIVE','CONTROL_FLOOR'
    return 'SUPPORTED','CONTROL_PILOT_WINDOW'


def evaluate(run:Path)->dict:
    m=json.loads((run/'manifest.json').read_text());analysis=json.loads((run/'analysis.json').read_text());source=Path(m['source_repo'])
    require(m['status']=='COMPLETED' and m['candidate_A_updates']==0,'Incomplete or forbidden candidate work')
    audit_source(m,source);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    errors={};rows=[];samples=0
    for record in m['records']:
        arm,seed=record['arm'],record['seed'];path=run/f'seed-{seed}'/arm
        result=json.loads((path/'result.json').read_text())
        require({k:v for k,v in result.items() if k!='train_losses'}==record,'Record differs from manifest')
        require(len(result['train_losses'])==result['updates']==3000 and np.isfinite(result['train_losses']).all(),'Training history invalid')
        require(sha(path/'final.pt')==result['checkpoint_sha256'] and sha(path/'predictions.npz')==result['predictions_sha256'],'Raw artifact changed')
        saved=torch.load(path/'final.pt',map_location='cpu',weights_only=True);model=build_model(arm,seed);model.load_state_dict(saved['model']);model.eval()
        require(tensor_hash(model)==result['final_state_sha256'],'Tensor hash mismatch')
        values=[];metrics={};prefix=f'{arm}/{seed}'
        with np.load(path/'predictions.npz',allow_pickle=False) as data:
            for length in [32,64]:
                y=data[f'y_{length}'];require(y.shape==(4096,length),'Unexpected evaluation sample dimensions')
                target=filter_matrix(y);replayed=[]
                with torch.no_grad():
                    for begin in range(0,len(y),128):
                        x=F.one_hot(torch.from_numpy(y[begin:begin+128].astype(np.int64)),16).float();replayed.append(model(x).numpy())
                logits=np.concatenate(replayed).astype(np.float64);stored=data[f'logp_{length}'].astype(np.float64)
                errors[f'{prefix}/{length}/target']=float(np.max(np.abs(target-data[f'target_{length}'])))
                errors[f'{prefix}/{length}/checkpoint_replay']=float(np.max(np.abs(np.exp(logits)-np.exp(stored))))
                shifted=logits-logits.max(1,keepdims=True);logp=shifted-np.log(np.exp(shifted).sum(1,keepdims=True));kl=np.sum(target*(np.log(target)-logp),axis=1)
                require(np.isfinite(kl).all(),'Nonfinite KL');values.append(kl);metrics[str(length)]={'kl_mean':float(kl.mean())}
                errors[f'{prefix}/{length}/kl']=float(np.max(np.abs(kl-data[f'kl_{length}'])))
                samples+=len(y)
            rng=np.random.Generator(np.random.PCG64(1800000000+seed));boot=np.empty((10000,2))
            for begin in range(0,10000,25):
                end=min(begin+25,10000);idx=rng.integers(0,4096,size=(end-begin,4096))
                for j in range(2):boot[begin:end,j]=values[j][idx].mean(1)
            errors[f'{prefix}/bootstrap']=float(np.max(np.abs(boot-data['bootstrap_means'])))
        for j,length in enumerate([32,64]):
            metrics[str(length)]['ci']=np.quantile(boot[:,j],[.025,.975],method='linear').tolist()
            original=result['metrics'][str(length)]
            expected=np.array([metrics[str(length)]['kl_mean'],*metrics[str(length)]['ci']])
            stored_metrics=np.array([original['kl_mean'],*original['ci']])
            require(stored_metrics.shape==(3,) and np.isfinite(stored_metrics).all(),'Invalid recorded interval')
            errors[f'{prefix}/{length}/metrics']=float(np.max(np.abs(expected-stored_metrics)))
        rows.append(dict(arm=arm,seed=seed,metrics=metrics))
    require(all(np.isfinite(v) and v<=(2e-6 if k.endswith(('/target','/checkpoint_replay')) else 1e-6) for k,v in errors.items()),'Recomputation mismatch')
    verdict,substatus=decision(rows)
    require((verdict,substatus)==(analysis['decision'],analysis['substatus']),'Decision mismatch')
    require(m['research_updates']==analysis['research_updates']==3000*len(rows),'Update accounting mismatch')
    for r in analysis['records']:
        original=next(t for t in m['records'] if t['arm']==r['arm'] and t['seed']==r['seed'])
        require(all(original[k]==r[k] for k in r),'Analysis record differs')
    return dict(pass_=True,recomputed_decision=verdict,recomputed_substatus=substatus,errors=errors,maximum_error=max(errors.values()),evaluation_sequences_replayed=samples,source_commit=m['git_commit'],manifest_sha256=sha(run/'manifest.json'),analysis_sha256=sha(run/'analysis.json'),scope='CHECKPOINT_REPLAY_AND_INDEPENDENT_ORACLE_NOT_RETRAINING_REPLICATION')


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('No overwrite')
    started=time.monotonic()
    def timeout(signum,frame):raise TimeoutError('Reproduction budget300s exceeded')
    prior=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,300)
    try:report=evaluate(a.run)
    except Exception as exc:report=dict(pass_=False,decision='INCONCLUSIVE',substatus='REPRODUCTION_UNRESOLVED',error=repr(exc))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior)
    report.update(seconds=time.monotonic()-started,training_updates=0,new_samples=0,auditor_sha256=sha(Path(__file__)))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='errors'},indent=2),flush=True)
    if not report['pass_']:raise SystemExit(1)

if __name__=='__main__':main()
