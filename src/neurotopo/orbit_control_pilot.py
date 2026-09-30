"""Registered, staged real-control training for EXP-0016; never constructs candidate A."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import sys
import time
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .orbit_information import generate, posterior, bootstrap_means, sha, array_sha
from .run import git_output, source_fingerprints, write_json

PERMUTATION = [0,5,11,2,14,7,3,12,8,1,15,6,10,4,13,9]
ARMS = ('H','B','U','D')


class LearnedHMM(nn.Module):
    def __init__(self):
        super().__init__(); self.transition_logits=nn.Parameter(torch.zeros(16,16))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        transition=torch.softmax(self.transition_logits,dim=0)
        q=x.new_full((len(x),16),1/16)
        for t in range(x.shape[1]):
            q=(q@transition.T)*(.4/15 + x[:,t]*(.6-.4/15))
            q=q/q.sum(dim=1,keepdim=True)
        return q.log()


def project_cyclic(w:torch.Tensor)->torch.Tensor:
    return torch.stack([torch.roll(w,(k,k),(-2,-1)) for k in range(16)]).mean(0)


def project_conjugate(w:torch.Tensor)->torch.Tensor:
    p=torch.tensor(PERMUTATION,device=w.device);inv=torch.argsort(p)
    canonical=w[...,inv,:][...,inv]
    projected=project_cyclic(canonical)
    return projected[...,p,:][...,p]


class ControlGRU(nn.Module):
    def __init__(self,arm:str,seed:int):
        super().__init__()
        if arm not in ('B','U','D'):raise ValueError('No candidate model is supported')
        self.arm=arm;self.hidden=64 if arm=='D' else 16
        self.gru=nn.GRU(16,self.hidden,batch_first=True)
        self.readout=nn.Linear(self.hidden,16)
        # Reset after constructors: discarded default initialization does not alter paired explicit draws.
        torch.manual_seed(seed);scale=1/math.sqrt(self.hidden)
        with torch.no_grad():
            for parameter in [self.gru.weight_ih_l0,self.gru.weight_hh_l0]:parameter.uniform_(-scale,scale)
            self.gru.bias_ih_l0.zero_();self.gru.bias_hh_l0.zero_()
            self.readout.weight.uniform_(-scale,scale);self.readout.bias.zero_()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        _,h=self.gru(x)
        return F.log_softmax(self.readout(h[-1]),dim=-1)

    def penalty(self)->torch.Tensor:
        if self.arm!='B':return self.readout.weight.new_zeros(())
        matrices=torch.cat([self.gru.weight_ih_l0.reshape(3,16,16),self.gru.weight_hh_l0.reshape(3,16,16),self.readout.weight[None]],dim=0)
        biases=torch.cat([self.gru.bias_ih_l0.reshape(3,16),self.gru.bias_hh_l0.reshape(3,16),self.readout.bias[None]],dim=0)
        return ((matrices-project_conjugate(matrices)).square().sum()+(biases-biases.mean(-1,keepdim=True)).square().sum())/1904


def build_model(arm:str,seed:int)->nn.Module:
    if arm=='H':return LearnedHMM()
    return ControlGRU(arm,seed)


def tensor_hash(model:nn.Module)->str:
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def scores(target:np.ndarray,logits:np.ndarray)->np.ndarray:
    logits=np.asarray(logits,dtype=np.float64)
    shifted=logits-logits.max(axis=1,keepdims=True)
    logp=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
    value=(target*(np.log(target)-logp)).sum(axis=1)
    if not np.isfinite(value).all() or value.min() < -1e-10:raise FloatingPointError('Invalid predictive KL')
    return value


def validate_config(c:dict)->None:
    expected=dict(experiment_id='EXP-0016',version='v1',workflow='orbit_control_pilot',gate1_claim=False,
                  task=dict(states=16,train_length=32,eval_lengths=[32,64],train_root_base=1600000000,eval_root_base=1700000000,eval_examples=4096),
                  training=dict(seeds=[16001,16002,16003],batch_size=64,updates=3000,lr=.001,betas=[.9,.999],eps=1e-8,clip=1.,threads=1,monitor_every=500),
                  qualification=dict(h_iid_max=.1,h_ood_upper_max=.02,iid_each_max=.15,iid_mean_max=.1,ood_each_min=.04,ood_each_max=1.,ood_mean_min=.08,ood_mean_max=.8,bootstrap_resamples=10000,bootstrap_root_base=1800000000),
                  resource=dict(worker_wall_seconds_cap=1200,reproduction_wall_seconds_cap=300,research_update_cap=36000))
    for key,value in expected.items():
        if c.get(key)!=value:raise ValueError(f'Frozen pilot config differs: {key}')
    if np.__version__!='2.3.3' or torch.__version__.split('+')[0]!='2.8.0':raise ValueError('Pinned runtime required')


def hmm_ceiling(records:list[dict])->bool:
    return sorted(r['seed'] for r in records)==[16001,16002,16003] and all(r['arm']=='H' and r['status']=='COMPLETED' and r['metrics']['32']['kl_mean']<.1 and r['metrics']['64']['ci'][1]<.02 for r in records)


def classify(records:list[dict])->tuple[str,str]:
    if any(r.get('status')!='COMPLETED' for r in records):return 'INCONCLUSIVE','ENGINEERING_ABORT'
    for row in records:
        for metrics in row['metrics'].values():
            if (not np.isfinite([metrics['kl_mean'],*metrics['ci']]).all()
                    or len(metrics['ci'])!=2 or metrics['ci'][0]>metrics['ci'][1]):
                return 'INCONCLUSIVE','ENGINEERING_ABORT'
    hs=[r for r in records if r['arm']=='H']
    if hmm_ceiling(hs):
        if len(records)!=3:raise ValueError('Forbidden GRU work after H ceiling')
        return 'INCONCLUSIVE','SIMPLE_ALGORITHM_CEILING'
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm]
        if sorted(r['seed'] for r in rows)!=[16001,16002,16003]:return 'INCONCLUSIVE','INCOMPLETE_BUDGET'
        if arm=='H':continue
        iid=[r['metrics']['32']['kl_mean'] for r in rows];ood=[r['metrics']['64']['kl_mean'] for r in rows]
        if not all(np.isfinite(iid+ood)):return 'INCONCLUSIVE','ENGINEERING_ABORT'
    for arm in ('B','U','D'):
        iid=[r['metrics']['32']['kl_mean'] for r in records if r['arm']==arm]
        if max(iid)>=.15 or np.mean(iid)>=.1:return 'INCONCLUSIVE','LEARNING_FAILURE'
    for arm in ('B','U','D'):
        ood=[r['metrics']['64']['kl_mean'] for r in records if r['arm']==arm]
        if min(ood)<.04 or np.mean(ood)<.08:return 'INCONCLUSIVE','CONTROL_CEILING'
    for arm in ('B','U','D'):
        ood=[r['metrics']['64']['kl_mean'] for r in records if r['arm']==arm]
        if max(ood)>1 or np.mean(ood)>.8:return 'INCONCLUSIVE','CONTROL_FLOOR'
    return 'SUPPORTED','CONTROL_PILOT_WINDOW'


def evaluate(model:nn.Module,seed:int)->tuple[dict,dict]:
    arrays={};metrics={};values=[]
    model.eval()
    for length in [32,64]:
        root=1700000000+seed*100+length
        y=generate(4096,length,root)['y'];q=posterior(y);logps=[]
        prediction_started=time.monotonic()
        with torch.no_grad():
            for start in range(0,len(y),128):
                x=F.one_hot(torch.from_numpy(y[start:start+128].astype(np.int64)),16).float()
                logps.append(model(x).numpy())
        logp=np.concatenate(logps);kl=scores(q,logp);values.append(kl)
        arrays.update({f'y_{length}':y,f'target_{length}':q,f'logp_{length}':logp,f'kl_{length}':kl})
        metrics[str(length)]=dict(kl_mean=float(kl.mean()),eval_root=root,data_sha256=array_sha(y),target_sha256=array_sha(q),prediction_seconds=time.monotonic()-prediction_started)
    boot=bootstrap_means(np.column_stack(values),1800000000+seed,10000)
    arrays['bootstrap_means']=boot
    for column,length in enumerate([32,64]):metrics[str(length)]['ci']=np.quantile(boot[:,column],[.025,.975],method='linear').tolist()
    return metrics,arrays


def train_arm(arm:str,seed:int,output:Path,manifest:dict,deadline:float)->dict:
    path=output/f'seed-{seed}'/arm;path.mkdir(parents=True,exist_ok=False)
    model=build_model(arm,seed);initial_hash=tensor_hash(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    losses=[];data_hash=hashlib.sha256();started=time.monotonic();cpu_started=time.process_time();attempted=0
    try:
        model.train()
        for step in range(1,3001):
            if time.monotonic()>=deadline:raise TimeoutError('Shared pilot deadline reached')
            y=generate(64,32,1600000000+seed*10000+step)['y'];q=posterior(y)
            data_hash.update(y.tobytes());data_hash.update(q.tobytes())
            x=F.one_hot(torch.from_numpy(y.astype(np.int64)),16).float();target=torch.from_numpy(q).float()
            optimizer.zero_grad(set_to_none=True)
            logp=model(x);loss=-(target*logp).sum(-1).mean()
            if arm=='B':loss=loss+model.penalty()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            attempted+=1;manifest['research_updates']+=1
            optimizer.step();losses.append(float(loss.detach()))
            if step%500==0:
                print(json.dumps(dict(arm=arm,seed=seed,updates=step,loss=losses[-1],seconds=time.monotonic()-started)),flush=True)
                write_json(output/'manifest.json',manifest)
        training_seconds=time.monotonic()-started;training_cpu=time.process_time()-cpu_started
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),arm=arm,seed=seed),path/'final.pt')
        metrics,arrays=evaluate(model,seed)
        with (path/'predictions.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
        result=dict(arm=arm,seed=seed,status='COMPLETED',updates=attempted,train_losses=losses,
                    initial_sha256=initial_hash,final_state_sha256=tensor_hash(model),checkpoint_sha256=sha(path/'final.pt'),predictions_sha256=sha(path/'predictions.npz'),
                    train_data_sha256=data_hash.hexdigest(),metrics=metrics,training_seconds=training_seconds,training_cpu_seconds=training_cpu,arm_wall_seconds=time.monotonic()-started,
                    resources=dict(trainable_parameters=sum(p.numel() for p in model.parameters()),parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),dynamic_state_values=64 if arm=='D' else 16,
                                   examples=64*attempted,input_tokens=64*32*attempted,updates=attempted,internal_steps_per_example=32,
                                   executed_compute='NOT_PROFILED: no matched-effect claim',peak_memory='NOT_MEASURED',effective_degrees_of_freedom='NOT_MEASURED: tensor counts are not statistical degrees of freedom'))
        write_json(path/'result.json',result);return result
    except BaseException as error:
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),path/'failed.pt')
        write_json(path/'failure.json',dict(arm=arm,seed=seed,status='FAILED',attempted_updates=attempted,losses=losses,error=repr(error),seconds=time.monotonic()-started))
        raise


def run(config_path:Path,output_override:Path|None=None)->Path:
    started=time.monotonic();c=json.loads(config_path.read_text());validate_config(c)
    if git_output('status','--porcelain'):raise RuntimeError('Clean frozen checkout required')
    validation_path=Path('artifacts/validation/suite.json');validation=json.loads(validation_path.read_text())
    sources=source_fingerprints()
    if not validation.get('pass') or validation['source_sha256']!=sources:raise RuntimeError('Validation does not match source')
    preflight_path=Path('artifacts/validation/exp0016_preflight.json');preflight=json.loads(preflight_path.read_text())
    if (not preflight.get('pass_') or preflight['model_code_sha256']!=sha(Path(__file__))
            or preflight['config_sha256']!=sha(config_path)):
        raise RuntimeError('Engineering preflight does not match model/config')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    output=output_override or Path(c['output']);output.mkdir(parents=True,exist_ok=False)
    manifest=dict(experiment_id='EXP-0016',version='v1',config=c,config_sha256=sha(config_path),source_repo=str(Path.cwd()),git_commit=git_output('rev-parse','HEAD'),git_status='',source_sha256=sources,validation_sha256=sha(validation_path),
                  frozen_hashes={str(p):sha(p) for p in [config_path,Path(c['protocol']),Path(c['entry_contract']),Path('uv.lock')]},python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),executable_sha256=sha(Path(sys.executable).resolve()),
                  preflight_sha256=sha(preflight_path),status='RUNNING',records=[],research_updates=0,candidate_A_updates=0,gate1_claim=False)
    write_json(output/'manifest.json',manifest);deadline=started+1200
    try:
        for seed in [16001,16002,16003]:
            row=train_arm('H',seed,output,manifest,deadline);manifest['records'].append({k:v for k,v in row.items() if k!='train_losses'});write_json(output/'manifest.json',manifest)
        manifest['h_ceiling_stop']=hmm_ceiling(manifest['records'])
        write_json(output/'manifest.json',manifest)
        if not manifest['h_ceiling_stop']:
            for seed in [16001,16002,16003]:
                for arm in ['B','U','D']:
                    row=train_arm(arm,seed,output,manifest,deadline);manifest['records'].append({k:v for k,v in row.items() if k!='train_losses'});write_json(output/'manifest.json',manifest)
        if time.monotonic()>=deadline:raise TimeoutError('Pilot budget exhausted before analysis')
        manifest.update(status='COMPLETED',seconds=time.monotonic()-started)
    except BaseException as error:
        manifest.update(status='FAILED',seconds=time.monotonic()-started,error=repr(error));write_json(output/'manifest.json',manifest);raise
    write_json(output/'manifest.json',manifest);return output


def analyze(output:Path)->dict:
    if (output/'analysis.json').exists():raise FileExistsError('No analysis overwrite')
    if (output/'worker_failure.json').exists():
        failure=json.loads((output/'worker_failure.json').read_text());result=dict(experiment_id='EXP-0016',decision='INCONCLUSIVE',substatus=failure['substatus'],failure=failure,gate1_claim=False)
    else:
        manifest=json.loads((output/'manifest.json').read_text());validate_config(manifest['config'])
        if manifest['status']!='COMPLETED':raise ValueError('Incomplete pilot')
        records=[]
        for row in manifest['records']:
            path=output/f"seed-{row['seed']}"/row['arm'];stored=json.loads((path/'result.json').read_text())
            if sha(path/'predictions.npz')!=stored['predictions_sha256'] or sha(path/'final.pt')!=stored['checkpoint_sha256']:raise ValueError('Changed raw artifact')
            records.append(stored)
        decision,substatus=classify(records)
        result=dict(experiment_id='EXP-0016',decision=decision,substatus=substatus,gate1_claim=False,candidate_A_updates=0,research_updates=manifest['research_updates'],
                    records=[dict(arm=r['arm'],seed=r['seed'],metrics=r['metrics'],updates=r['updates'],training_seconds=r['training_seconds'],resources=r['resources']) for r in records],
                    interpretation='Control diagnostic only; three-seed pilot is not complete eight-seed qualification or Gate1 evidence',
                    next_action='Close fixed ORBIT task configuration' if substatus=='SIMPLE_ALGORITHM_CEILING' else 'Retain diagnosis; no automatic candidate training')
    write_json(output/'analysis.json',result);print(json.dumps(result,ensure_ascii=False),flush=True);return result


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    def terminated(signum,frame):raise TimeoutError('Supervisor terminated at shared wall-time cap')
    signal.signal(signal.SIGTERM,terminated)
    run(a.config,a.output)

if __name__=='__main__':main()
