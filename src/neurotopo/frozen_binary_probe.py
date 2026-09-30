"""EXP0018: convex linear probes of frozen CO states, never GRU training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import signal
import sys
import time
from typing import Any
import numpy as np
import torch
from .order_observation import ControlGRU, generate, input_tensor, sha
from .run import git_output, source_fingerprints


def save_json(path: Path, data: Any) -> None:
    encoded = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(encoded)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0., -z))


def objective(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    z = x @ w
    p = sigmoid(z)
    penalty = np.full(len(w), .001); penalty[-1] = 0.
    value = float(np.mean(np.logaddexp(0., z) - y*z) + .5*np.dot(penalty*w, w))
    gradient = x.T @ (p-y) / len(y) + penalty*w
    hessian = (x.T * (p*(1-p))) @ x / len(y) + np.diag(penalty)
    return value, gradient, hessian


def fit(x: np.ndarray, y: np.ndarray) -> dict:
    w = np.zeros(x.shape[1], dtype=np.float64)
    trace = []; updates = 0
    for _ in range(50):
        value, grad, hess = objective(x,y,w)
        norm = float(np.max(np.abs(grad)))
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError('Invalid convex objective')
        if norm <= 1e-7:
            break
        direction = np.linalg.solve(hess,grad)
        descent = float(grad @ direction)
        accepted = False
        for k in range(25):
            step = 2.**(-k)
            candidate = w-step*direction
            next_value = objective(x,y,candidate)[0]
            if next_value <= value-1e-4*step*descent:
                trace.append(dict(objective_before=value,objective_after=next_value,step=step,gradient_inf=norm))
                w = candidate; updates += 1; accepted = True
                break
        if not accepted:
            break
    value, grad, _ = objective(x,y,w)
    norm = float(np.max(np.abs(grad)))
    return dict(weights=w.tolist(),updates=updates,converged=norm<=1e-7,gradient_inf=norm,objective=value,trace=trace)


def scaling(h: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    h = h.astype(np.float64)
    mean, scale = h.mean(0), h.std(0,ddof=0)
    return mean, np.where(scale<1e-8,1.,scale)


def transform(h: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (h.astype(np.float64)-mean)/scale
    return np.column_stack((z,np.ones(len(z))))


def metrics(z: np.ndarray, y: np.ndarray) -> dict:
    pred = z>=0
    class_acc = [float((pred[y==b]==b).mean()) if np.any(y==b) else None for b in (0,1)]
    return dict(accuracy=float((pred==y).mean()),bce=float(np.mean(np.logaddexp(0,z)-y*z)),
                balanced_accuracy=float(sum(class_acc)/2) if all(v is not None for v in class_acc) else None,
                positive_fraction=float(y.mean()))


def standard_state_hash(model: ControlGRU) -> str:
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def extract(model: ControlGRU, tokens: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    hidden=[]; logits=[]
    with torch.no_grad():
        for start in range(0,len(tokens),128):
            x=input_tensor(tokens[start:start+128])
            state,_=model.gru(x)
            logits.append(model.readout(state)[:,-1].numpy())
            hidden.append(state[:,-1].numpy())
    return np.concatenate(hidden),np.concatenate(logits)


def parent_metrics(logits: np.ndarray, truth: np.ndarray) -> dict:
    logp=logits.astype(np.float64)-np.logaddexp.reduce(logits.astype(np.float64),axis=1)[:,None]
    cube=logp.reshape(-1,2,3)
    b=np.logaddexp.reduce(cube,axis=2)
    j=np.logaddexp.reduce(cube,axis=1)
    return dict(b_accuracy=float((b.argmax(1)==truth//3).mean()),j_accuracy=float((j.argmax(1)==truth%3).mean()))


def summarize(values: list[float]) -> dict:
    a=np.asarray(values,dtype=float);mean=float(a.mean());sd=float(a.std(ddof=1));half=4.302652729911275*sd/np.sqrt(3)
    return dict(values=values,mean=mean,sd=sd,t95_df2=[float(mean-half),float(mean+half)])


def classify(records: list[dict]) -> tuple[str,str]:
    expected={(s,a) for s in (17001,17002,17003) for a in ('D64','D16')}
    if len(records)!=6 or {(r['seed'],r['arm']) for r in records}!=expected:
        return 'INCONCLUSIVE','ENGINEERING_ABORT'
    if any(not r['fits'][name]['converged'] for r in records for name in ('true','shuffle')):
        return 'INCONCLUSIVE','PROBE_OPTIMIZATION_FAILURE'
    primary=[r for r in records if r['arm']=='D64']
    for r in primary:
        for n in ('32','128'):
            m=r['test'][n]
            if not (.47<=m['true']['positive_fraction']<=.53 and .47<=m['shuffle']['accuracy']<=.53
                    and m['parent']['b_accuracy']<=.55 and m['parent']['j_accuracy']>=.95):
                return 'INCONCLUSIVE','CONTROL_FAILURE'
    def strong(r:dict,n:str)->bool:
        m=r['test'][n];return m['true']['accuracy']>=.8 and m['gap']>=.2
    def weak(r:dict)->bool:
        m=r['test']['32'];return m['true']['accuracy']>=.55 and m['gap']>=.05
    if all(strong(r,n) for r in primary for n in ('32','128')):
        return 'SUPPORTED','LINEAR_INFORMATION_GENERALIZES'
    if all(strong(r,'32') for r in primary):
        return 'PARTIALLY_SUPPORTED','LENGTH_LIMITED_LINEAR_READOUT'
    if any(strong(r,'32') for r in primary) or all(weak(r) for r in primary):
        return 'PARTIALLY_SUPPORTED','LIMITED_LINEAR_SIGNAL'
    if all(r['test']['32']['true']['accuracy']<=.55 and r['test']['32']['gap']<=.05 for r in primary):
        return 'INCONCLUSIVE','NO_USEFUL_LINEAR_READOUT'
    return 'INCONCLUSIVE','MIXED_WEAK_READOUT'


def validate_config(c:dict)->None:
    assert c['experiment_id']=='EXP-0018' and c['version']=='v1' and c['workflow']=='frozen_binary_probe' and c['gate1_claim'] is False
    assert c['data']==dict(fit_examples=8192,test_examples=4096,fit_operations=32,test_operations=[32,128],namespace=181001,shuffle_namespace=184001)
    assert c['probe']==dict(l2=.001,max_updates=50,gradient_tolerance=1e-7,armijo=1e-4,backtracks=25,scale_floor=1e-8)
    assert c['resource']==dict(worker_wall_seconds_cap=120,reproduction_wall_seconds_cap=120,research_update_cap=600)
    assert c['thresholds']==dict(strong_accuracy=.8,strong_gap=.2,weak_accuracy=.55,weak_gap=.05,control_low=.47,control_high=.53,original_b_max=.55,original_j_min=.95)
    assert len(c['sources'])==6 and {(s['seed'],s['arm']) for s in c['sources']}=={(s,a) for s in (17001,17002,17003) for a in ('D64','D16')}
    assert c['matching_criterion_version']=='G1-MATCH-v3'
    assert np.__version__=='2.3.3' and torch.__version__.split('+')[0]=='2.8.0'


def run(config_path:Path,output_override:Path|None=None)->Path:
    started=time.monotonic(); c=json.loads(config_path.read_text());validate_config(c)
    assert not git_output('status','--porcelain'),'Clean frozen checkout required'
    suite=Path('artifacts/validation/suite.json');v=json.loads(suite.read_text())
    assert v['pass'] and v['source_sha256']==source_fingerprints()
    for source in c['sources']:
        assert sha(Path(source['path']))==source['sha256']
    output=output_override or Path(c['output']);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    manifest=dict(experiment_id='EXP-0018',version='v1',config=c,source_sha256=source_fingerprints(),git_commit=git_output('rev-parse','HEAD'),git_status='',
                  frozen_hashes={str(p):sha(p) for p in (config_path,Path(c['protocol']),Path(c['entry_contract']),Path('uv.lock'),suite)},
                  python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),original_gru_updates=0,gate1_claim=False)
    save_json(output/'manifest.json',manifest)
    records=[];updates=0
    try:
        for source in c['sources']:
            if time.monotonic()-started>=120:raise TimeoutError('Shared run budget')
            seed,arm=source['seed'],source['arm'];path=output/f'seed-{seed}'/arm;path.mkdir(parents=True)
            model=ControlGRU(arm,seed);model.load_state_dict(torch.load(source['path'],weights_only=True,map_location='cpu')['model'])
            model.eval();model.requires_grad_(False);before=standard_state_hash(model)
            arrays={};features={};labels={};parent={};tokens_by_split={}
            extraction_started=time.monotonic()
            for split,n,count,ident in [('fit',32,8192,0),('test32',32,4096,1),('test128',128,4096,2)]:
                data=generate(count,n,'CO',[181001,seed,ident,n]);tokens=data['tokens'];truth=data['latent_final']
                h,z=extract(model,tokens);y=(truth//3).astype(np.float64)
                features[split]=h;labels[split]=y;tokens_by_split[split]=tokens
                parent[split]=parent_metrics(z,truth)
                for name,value in (('tokens',tokens),('latent',truth),('hidden',h),('original_logits',z)):
                    arrays[f'{split}_{name}']=value
            overlap=len({r.tobytes() for r in tokens_by_split['fit']} & {r.tobytes() for r in tokens_by_split['test32']})
            assert overlap==0,'Probe fit/test overlap'
            assert standard_state_hash(model)==before and sha(Path(source['path']))==source['sha256']
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            extraction_seconds=time.monotonic()-extraction_started
            mean,scale=scaling(features['fit']);xfit=transform(features['fit'],mean,scale)
            shuffled=np.random.default_rng(np.random.SeedSequence([184001,seed,0 if arm=='D64' else 1])).permutation(labels['fit'])
            arrays.update(fit_mean=mean,fit_scale=scale,fit_true_labels=labels['fit'],fit_shuffled_labels=shuffled)
            fits={};tests={'32':{},'128':{}};fitting_started=time.monotonic()
            for name,yfit in [('true',labels['fit']),('shuffle',shuffled)]:
                fitted=fit(xfit,yfit);updates+=fitted['updates'];fits[name]=fitted
                w=np.asarray(fitted['weights']);arrays[f'{name}_weights']=w
                fitted['fit_metrics']=metrics(xfit@w,yfit)
                for n in (32,128):
                    split=f'test{n}';z=transform(features[split],mean,scale)@w
                    arrays[f'{split}_{name}_logits']=z
                    tests[str(n)][name]=metrics(z,labels[split])
            for n in (32,128):
                tests[str(n)]['gap']=tests[str(n)]['true']['accuracy']-tests[str(n)]['shuffle']['accuracy']
                tests[str(n)]['parent']=parent[f'test{n}']
            with (path/'features.npz').open('xb') as f:np.savez_compressed(f,**arrays)
            row=dict(seed=seed,arm=arm,fits=fits,test=tests,state_before_after_sha256=before,parent_checkpoint_sha256=source['sha256'],
                     features_sha256=sha(path/'features.npz'),fit_test_overlap=overlap,extraction_seconds=extraction_seconds,
                     probe_fit_and_save_seconds=time.monotonic()-fitting_started,
                     resources=dict(parent_parameters=sum(p.numel() for p in model.parameters()),parent_state_values=model.hidden,probe_coefficients=model.hidden+1,
                                    fit_examples=8192,test_examples_each=4096,newton_updates=sum(f['updates'] for f in fits.values()),gru_updates=0,executed_flops='NOT_MEASURED',peak_memory='NOT_MEASURED'))
            save_json(path/'result.json',row);records.append(row)
            print(json.dumps(dict(seed=seed,arm=arm,test=tests,updates=row['resources']['newton_updates'])),flush=True)
        assert updates<=600
        save_json(output/'completion.json',dict(status='COMPLETED',seconds=time.monotonic()-started,probe_updates=updates,records=records,original_gru_updates=0))
    except BaseException as error:
        save_json(output/'run_failure.json',dict(error=repr(error),seconds=time.monotonic()-started,probe_updates=updates,completed_records=records))
        raise
    return output


def analyze(output:Path)->dict:
    if (output/'worker_failure.json').exists():
        failure=json.loads((output/'worker_failure.json').read_text());sub=failure['substatus']
        p=output/'run_failure.json';detail=json.loads(p.read_text()) if p.exists() else {}
        if detail.get('error','').startswith('TimeoutError('):sub='INCOMPLETE_BUDGET'
        result=dict(experiment_id='EXP-0018',decision='INCONCLUSIVE',substatus=sub,failure=failure,detail=detail,gate1_claim=False)
    else:
        completion=json.loads((output/'completion.json').read_text());records=completion['records']
        for r in records:
            assert sha(output/f"seed-{r['seed']}"/r['arm']/'features.npz')==r['features_sha256']
        decision,sub=classify(records);summaries=[]
        for arm in ('D64','D16'):
            rows=[r for r in records if r['arm']==arm]
            for n in ('32','128'):
                summaries.append(dict(arm=arm,operations=int(n),true_accuracy=summarize([r['test'][n]['true']['accuracy'] for r in rows]),
                                      shuffle_accuracy=summarize([r['test'][n]['shuffle']['accuracy'] for r in rows]),
                                      gap=summarize([r['test'][n]['gap'] for r in rows]),
                                      true_bce=summarize([r['test'][n]['true']['bce'] for r in rows]),
                                      original_b_accuracy=summarize([r['test'][n]['parent']['b_accuracy'] for r in rows])))
        result=dict(experiment_id='EXP-0018',decision=decision,substatus=sub,records=records,summaries=summaries,
                    original_gru_updates=0,probe_updates=completion['probe_updates'],gate1_claim=False,
                    interpretation='Fixed-regularization linear accessibility only; failure is not absence of information.')
    save_json(output/'analysis.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('records','summaries')}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    def terminate(*_:Any)->None:raise TimeoutError('Supervisor terminated at shared wall cap')
    signal.signal(signal.SIGTERM,terminate)
    run(a.config,a.output)
