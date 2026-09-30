"""EXP0020v2 exploratory learning, with three honest transition controls."""
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
from torch import nn
from torch.nn import functional as F
from .frozen_binary_probe import save_json
from .order_observation import sha
from .run import git_output,source_fingerprints

ARMS=('RAW','INV','REL')
KEYS=('b0','commands','common','leak','targets')
CELLS={'IID_H':(8,'HOLD',.4,0.),'IID_U':(8,'UPDATE',.4,0.),
       'LONG_H':(64,'HOLD',.4,0.),'LONG_U':(64,'UPDATE',.4,0.),
       'ROT_H':(64,'HOLD',np.pi,0.),'ROT_U':(64,'UPDATE',np.pi,0.),
       'LEAK01_U':(64,'UPDATE',np.pi,.1),'LEAK04_U':(64,'UPDATE',np.pi,.4)}


def rotations(rng:np.random.Generator,shape:tuple[int,...],amplitude:float)->np.ndarray:
    axis=rng.normal(size=(*shape,3));norm=np.linalg.norm(axis,axis=-1,keepdims=True)
    if np.any(norm==0):raise FloatingPointError('Zero random axis')
    axis=axis/norm;angle=rng.uniform(-amplitude,amplitude,size=shape)
    k=np.zeros((*shape,3,3));x,y,z=np.moveaxis(axis,-1,0)
    k[...,0,1]=-z;k[...,0,2]=y;k[...,1,0]=z;k[...,1,2]=-x;k[...,2,0]=-y;k[...,2,1]=x
    c=np.cos(angle)[...,None,None];s=np.sin(angle)[...,None,None]
    return (c*np.eye(3)+(1-c)*axis[..., :,None]*axis[...,None,:]+s*k).astype(np.float32)


def generate(n:int,length:int,root:list[int],mode:str,common:float=.4,leak:float=0.)->dict[str,np.ndarray]:
    assert n%2==0 and mode in ('MIX','HOLD','UPDATE')
    seeds=np.random.SeedSequence(root).spawn(4);initial,cmd,rot,extra=[np.random.default_rng(s) for s in seeds]
    b0=initial.permutation(np.repeat([0,1],n//2)).astype(np.int64)
    update=initial.permutation(np.repeat([False,True],n//2)) if mode=='MIX' else np.full(n,mode=='UPDATE')
    u=cmd.integers(0,2,size=(n,length),dtype=np.int64);u[~update]=0;u[update,0]=1
    target=np.bitwise_xor.accumulate(np.column_stack((b0,u)),axis=1).astype(np.float32)
    return dict(b0=b0,commands=u,common=rotations(rot,(n,length+1),common),leak=rotations(extra,(n,length),leak),targets=target)


def data_hash(data:dict[str,np.ndarray])->str:
    h=hashlib.sha256()
    for k in KEYS:h.update(k.encode());h.update(data[k].tobytes())
    return h.hexdigest()


def gram(h:torch.Tensor)->torch.Tensor:
    x,y=h[:,:,0],h[:,:,1]
    return torch.stack(((x*x).sum(1),(x*y).sum(1),(y*y).sum(1)),dim=1)


def radial(h:torch.Tensor)->torch.Tensor:
    r2=h.square().sum(dim=1,keepdim=True)
    safe=torch.sqrt(r2.clamp_min(1e-8))
    exact=torch.tanh(1.5*safe)/safe
    series=1.5-(1.5**3/3)*r2+(2*1.5**5/15)*r2.square()
    return h*torch.where(r2<1e-8,series,exact)


class LearnedMemory(nn.Module):
    def __init__(self,arm:str,seed:int=20001):
        super().__init__()
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm;self.encoder=nn.Embedding(2,6);self.readout=nn.Linear(3,1)
        self.controller=nn.Sequential(nn.Linear(8 if arm=='RAW' else 5,32),nn.Tanh(),nn.Linear(32,4 if arm=='REL' else 6))
        torch.manual_seed(seed)
        with torch.no_grad():
            self.encoder.weight.normal_(0,.5);self.readout.weight.normal_(0,.1);self.readout.bias.zero_()
            nn.init.xavier_uniform_(self.controller[0].weight,gain=1.)
            nn.init.xavier_uniform_(self.controller[2].weight,gain=.5)
            self.controller[0].bias.zero_();self.controller[2].bias.zero_()

    def transition(self,h:torch.Tensor,u:torch.Tensor)->torch.Tensor:
        code=F.one_hot(u,2).to(h.dtype)
        features=torch.cat((h.flatten(1) if self.arm=='RAW' else gram(h),code),dim=1)
        change=self.controller(features)
        if self.arm=='REL':
            next_h=h@(torch.eye(2,dtype=h.dtype,device=h.device)+change.reshape(-1,2,2))
        elif self.arm=='RAW':next_h=h+change.reshape(-1,3,2)
        else:next_h=change.reshape(-1,3,2)
        return radial(next_h)

    def forward(self,b0:torch.Tensor,commands:torch.Tensor,common:torch.Tensor,leak:torch.Tensor,
                return_gram:bool=False)->torch.Tensor|tuple[torch.Tensor,torch.Tensor]:
        # R/E implement the environment; no learned layer receives their entries.
        h=common[:,0]@radial(self.encoder(b0).reshape(-1,3,2))
        g=gram(h);grams=[g];logits=[self.readout(g).squeeze(-1)]
        for t in range(commands.shape[1]):
            distorted=common[:,t+1]@h
            second=(leak[:,t]@distorted[:,:,1:2]).squeeze(-1)
            distorted=torch.stack((distorted[:,:,0],second),dim=-1)
            h=self.transition(distorted,commands[:,t])
            g=gram(h);grams.append(g);logits.append(self.readout(g).squeeze(-1))
        z=torch.stack(logits,dim=1)
        return (z,torch.stack(grams,dim=1)) if return_gram else z


def tensors(data:dict[str,np.ndarray],start:int=0,end:int|None=None)->tuple[torch.Tensor,...]:
    return tuple(torch.from_numpy(data[k][start:end]) for k in KEYS[:-1])


def predict(model:LearnedMemory,data:dict[str,np.ndarray])->np.ndarray:
    model.eval();z=[]
    with torch.no_grad():
        for i in range(0,len(data['b0']),128):z.append(model(*tensors(data,i,i+128)).numpy())
    return np.concatenate(z)


def predict_with_gram(model:LearnedMemory,data:dict[str,np.ndarray])->tuple[np.ndarray,np.ndarray]:
    model.eval();zs=[];gs=[]
    with torch.no_grad():
        for i in range(0,len(data['b0']),128):
            z,g=model(*tensors(data,i,i+128),return_gram=True);zs.append(z.numpy());gs.append(g.numpy())
    return np.concatenate(zs),np.concatenate(gs)


def metrics(logits:np.ndarray,data:dict[str,np.ndarray])->dict:
    z=logits.astype(np.float64);target=data['targets'];correct=(z>=0)==target
    u=data['commands'];hold=u==0;flip=u==1
    if not np.isfinite(z).all():raise FloatingPointError('Nonfinite predictions')
    return dict(bce=float((np.logaddexp(0,z)-target*z).mean()),accuracy_all=float(correct.mean()),
                final_accuracy=float(correct[:,-1].mean()),initial_accuracy=float(correct[:,0].mean()),
                hold_accuracy=float(correct[:,1:][hold].mean()) if hold.any() else None,
                flip_accuracy=float(correct[:,1:][flip].mean()) if flip.any() else None,
                examples=len(z),operations=int(u.shape[1]),hold_count=int(hold.sum()),flip_count=int(flip.sum()))


def state_hash(model:nn.Module,prefix:str='')->str:
    h=hashlib.sha256()
    for name,v in sorted(model.state_dict().items()):
        if name.startswith(prefix):h.update(name.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


def length_at(step:int)->int:
    for stop,length in ((125,1),(250,2),(500,4),(1000,8)):
        if step<=stop:return length
    raise ValueError(step)


def learning_flag(row:dict)->bool:
    x=row['evaluation']
    return x['IID_H']['accuracy_all']>=.95 and x['IID_U']['accuracy_all']>=.95 and x['IID_U']['flip_accuracy']>=.95


def classify(rows:list[dict])->tuple[str,str]:
    if len(rows)!=3 or {r['arm'] for r in rows}!=set(ARMS) or any(r['updates']!=1000 for r in rows):
        return 'INCONCLUSIVE','ENGINEERING_ABORT'
    candidate=next(r for r in rows if r['arm']=='REL')
    return ('SUPPORTED','SINGLE_SEED_LEARNABILITY_ONLY') if learning_flag(candidate) else ('INCONCLUSIVE','EXPLORATORY_LEARNING_UNRESOLVED')


def validate_config(c:dict)->None:
    assert c['experiment_id']=='EXP-0020' and c['version']=='v2' and c['workflow']=='learned_protected_pilot' and c['seed']==20001 and c['arms']==list(ARMS)
    assert c['training']==dict(updates=1000,batch_size=64,lr=.003,betas=[.9,.999],eps=1e-8,clip=1.,curriculum=[[125,1],[250,2],[500,4],[1000,8]],monitor_steps=[0,125,250,500,1000],monitor_examples=128,common_amplitude=.4)
    assert c['data']==dict(train_namespace=202001,eval_namespace=203001,monitor_namespace=204001,eval_examples=1024)
    assert c['cells']=={k:list(v) for k,v in CELLS.items()}
    assert c['resource']==dict(worker_wall_seconds_cap=600,reproduction_wall_seconds_cap=180,research_update_cap=3000)
    assert c['learning_threshold']==.95 and c['gate1_claim'] is False and c['matching_criterion_version']=='G1-MATCH-v3'
    assert np.__version__=='2.3.3' and torch.__version__.split('+')[0]=='2.8.0'


def run(config:Path,output_override:Path|None=None)->Path:
    start=time.monotonic();c=json.loads(config.read_text());validate_config(c)
    assert not git_output('status','--porcelain'),'Clean frozen checkout required'
    suite=Path('artifacts/validation/suite.json');v=json.loads(suite.read_text());assert v['pass'] and v['source_sha256']==source_fingerprints()
    pre=Path('artifacts/validation/exp0020_v2_preflight.json');p=json.loads(pre.read_text());assert p['pass'] and p['model_source_sha256']==sha(Path(__file__)) and p['config_sha256']==sha(config)
    out=output_override or Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    manifest=dict(experiment_id='EXP-0020',version='v2',config=c,git_commit=git_output('rev-parse','HEAD'),git_status='',source_sha256=source_fingerprints(),
                  frozen_hashes={str(x):sha(x) for x in (config,Path(c['protocol']),Path(c['entry_contract']),Path('uv.lock'),suite,pre)},
                  python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),gate1_claim=False)
    save_json(out/'manifest.json',manifest)
    records=[];total_updates=0
    try:
        data_dir=out/'data';data_dir.mkdir()
        monitor=generate(128,8,[204001,20001,8,0],'MIX')
        with (data_dir/'monitor.npz').open('xb') as f:np.savez_compressed(f,**monitor)
        evaluation={}
        for cell,(length,mode,amp,leak) in CELLS.items():
            data=generate(1024,length,[203001,20001,length,1 if mode=='HOLD' else 2],mode,amp,leak)
            with (data_dir/f'{cell}.npz').open('xb') as f:np.savez_compressed(f,**data)
            evaluation[cell]=data
        for arm in ARMS:
            path=out/arm;path.mkdir();model=LearnedMemory(arm);opt=torch.optim.Adam(model.parameters(),lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=0)
            initial=dict(encoder=state_hash(model,'encoder.'),readout=state_hash(model,'readout.'),all=state_hash(model))
            losses=[];curves=[];history=hashlib.sha256();started=time.monotonic();cpu=time.process_time();updates=0
            def checkpoint(step:int)->None:
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),arm=arm,step=step),path/f'checkpoint-{step}.pt')
                z=predict(model,monitor)
                with (path/f'monitor-{step}.npz').open('xb') as f:np.savez_compressed(f,logits=z)
                curve=dict(step=step,metrics=metrics(z,monitor),wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,
                           checkpoint_sha256=sha(path/f'checkpoint-{step}.pt'),prediction_sha256=sha(path/f'monitor-{step}.npz'))
                curves.append(curve);print(json.dumps(dict(arm=arm,monitor=curve)),flush=True)
            checkpoint(0)
            try:
                for step in range(1,1001):
                    if time.monotonic()-start>=600:raise TimeoutError('Shared pilot budget')
                    data=generate(64,length_at(step),[202001,20001,step,0],'MIX')
                    history.update(bytes.fromhex(data_hash(data)));model.train();opt.zero_grad(set_to_none=True)
                    logits=model(*tensors(data));target=torch.from_numpy(data['targets'])
                    loss=F.binary_cross_entropy_with_logits(logits,target)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    total_updates+=1;updates+=1;opt.step();losses.append(float(loss.detach()))
                    if step in (125,250,500,1000):checkpoint(step)
            except BaseException as error:
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),path/'failed.pt')
                save_json(path/'failure.json',dict(error=repr(error),updates=updates,losses=losses,curves=curves));raise
            training_seconds=time.monotonic()-started;predictions={};scores={}
            for cell,data in evaluation.items():
                z,g=predict_with_gram(model,data);predictions[cell]=z;predictions[cell+'_gram']=g;scores[cell]=metrics(z,data)
                b=data['targets'][:,-1]
                means=np.stack([g[b==value,-1].astype(float).mean(0) for value in (0,1)])
                weights=model.readout.weight.detach().numpy()[0].astype(float)
                scores[cell]['final_gram_mean_by_b']=means.tolist()
                scores[cell]['mean_logit_separation_terms']=(weights*(means[1]-means[0])).tolist()
            with (path/'predictions.npz').open('xb') as f:np.savez_compressed(f,**predictions)
            record=dict(arm=arm,updates=updates,initial_hashes=initial,training_data_sha256=history.hexdigest(),losses=losses,curves=curves,evaluation=scores,
                        learned_readout_weight=model.readout.weight.detach().numpy()[0].tolist(),learned_readout_bias=float(model.readout.bias.detach()[0]),
                        learned_initial_gram=gram(radial(model.encoder.weight.reshape(2,3,2))).detach().numpy().tolist(),
                        predictions_sha256=sha(path/'predictions.npz'),training_monitor_seconds=training_seconds,arm_seconds=time.monotonic()-started,
                        resources=dict(parameters=sum(p.numel() for p in model.parameters()),state_values=6,examples=64000,operation_tokens=64*(125+250+1000+4000),
                                       updates=updates,executed_flops='NOT_MEASURED',peak_memory='NOT_MEASURED'))
            save_json(path/'result.json',record);records.append({k:v for k,v in record.items() if k!='losses'})
            print(json.dumps(dict(arm=arm,final=scores)),flush=True)
        save_json(out/'completion.json',dict(status='COMPLETED',seconds=time.monotonic()-start,updates=total_updates,records=records,
                                           data_files={x.name:sha(x) for x in data_dir.glob('*.npz') if not x.name.startswith('._')}))
    except BaseException as error:
        save_json(out/'run_failure.json',dict(error=repr(error),seconds=time.monotonic()-start,updates=total_updates,completed_records=records));raise
    return out


def analyze(out:Path)->dict:
    if (out/'worker_failure.json').exists():
        failure=json.loads((out/'worker_failure.json').read_text());p=out/'run_failure.json';detail=json.loads(p.read_text()) if p.exists() else {}
        sub='INCOMPLETE_BUDGET' if detail.get('error','').startswith('TimeoutError(') else failure['substatus']
        a=dict(experiment_id='EXP-0020',decision='INCONCLUSIVE',substatus=sub,failure=failure,detail=detail,gate1_claim=False)
    else:
        completed=json.loads((out/'completion.json').read_text());rows=completed['records']
        assert len({r['training_data_sha256'] for r in rows})==1
        assert len({r['initial_hashes']['encoder'] for r in rows})==1 and len({r['initial_hashes']['readout'] for r in rows})==1
        for r in rows:
            assert sha(out/r['arm']/'predictions.npz')==r['predictions_sha256']
            for curve in r['curves']:assert sha(out/r['arm']/f"checkpoint-{curve['step']}.pt")==curve['checkpoint_sha256']
        for name,digest in completed['data_files'].items():assert sha(out/'data'/name)==digest
        verdict,sub=classify(rows)
        a=dict(experiment_id='EXP-0020',decision=verdict,substatus=sub,records=rows,updates=completed['updates'],gate1_claim=False,
               learnability_by_arm={r['arm']:learning_flag(r) for r in rows},
               interpretation='One fixed exploratory seed; local learnability only, no confirmed method superiority or Gate1 claim.')
    save_json(out/'analysis.json',a);print(json.dumps({k:v for k,v in a.items() if k!='records'}),flush=True);return a


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    def abort(*_:Any)->None:raise TimeoutError('Worker shared deadline')
    signal.signal(signal.SIGTERM,abort);run(a.config,a.output)
