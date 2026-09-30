"""EXP0026 fixed schedules; target-free recurrent interventions."""
from __future__ import annotations
import numpy as np
import torch
from .learned_protected_pilot import gram,radial,tensors
from .protected_interventions import environment_step
from .refresh_factor_core import refresh_step,weighted

PERIODS=(2,4,8,16,32,64)
MAIN=('NONE','ONCE8')+tuple('K'+str(k) for k in PERIODS)
EXTRA=('PHASE16','PHASE32','FULL8','SCALE8','GAUGE8','ADAPT90','COUNT90')
SNAPSHOTS=(8,16,32,64,128,256,512,1024)


def distance(g:np.ndarray,z:np.ndarray,pg:np.ndarray)->np.ndarray:
    return np.linalg.norm(weighted(g.astype(float)-pg[(z>=0).astype(int)]),axis=-1)


def threshold(cal:dict)->float:
    p=cal['prototypes'];pg=gram(torch.from_numpy(p)).numpy()
    g=cal['calibration_gram'][:,1:];z=cal['calibration_logits'][:,1:]
    return float(np.quantile(distance(g,z,pg),.9,method='linear'))


def count_schedule(counts:np.ndarray,length:int)->np.ndarray:
    mask=np.zeros((len(counts),length+1),bool)
    if length<8:raise ValueError('Schedule needs t8')
    for i,n in enumerate(counts):
        n=int(n)
        if not 0<=n<=length-7:raise ValueError('Invalid count')
        if n:
            ts=8+np.arange(n,dtype=np.int64)*(length-7)//n
            mask[i,ts]=True
    if not np.array_equal(mask.sum(1),counts):raise RuntimeError('Count schedule lost events')
    return mask


def attempt_mask(condition:str,t:int,d:np.ndarray,theta:float,last:np.ndarray,counts_mask:np.ndarray|None)->np.ndarray:
    if condition=='NONE':return np.zeros(len(d),bool)
    if condition=='ONCE8':return np.full(len(d),t==8)
    if condition.startswith('K'):return np.full(len(d),t>=8 and (t-8)%int(condition[1:])==0)
    if condition.startswith('PHASE'):return np.full(len(d),t>=int(condition[5:]) and t%int(condition[5:])==0)
    if condition in ('FULL8','SCALE8','GAUGE8'):return np.full(len(d),t>=8 and (t-8)%8==0)
    if condition=='ADAPT90':return (t>=8)&(t-last>=2)&(d>theta)
    if condition=='COUNT90':
        if counts_mask is None:raise ValueError('COUNT needs frozen per-example mask')
        return counts_mask[:,t]
    raise ValueError(condition)


def lower_tau(d:np.ndarray,theta:float)->tuple[np.ndarray,np.ndarray]:
    hit=d[:,8:]>theta;anyhit=hit.any(1)
    value=np.where(anyhit,np.argmax(hit,axis=1),hit.shape[1]).astype(np.int64)
    return value,~anyhit


@torch.no_grad()
def rollout(model,data:dict,prototypes:np.ndarray,condition:str,rotation:str,theta:float,
            counts:np.ndarray|None=None,command_diagnostics:bool=False)->dict:
    if condition not in MAIN+EXTRA or rotation not in ('ORIGINAL','IDENTITY'):raise ValueError('Bad condition')
    n,length=data['commands'].shape
    if n%128:raise ValueError('Original batch128 required')
    target_free={k:data[k] for k in ('b0','commands','common','leak')}
    p=torch.from_numpy(prototypes.astype(np.float32));pg=gram(p).numpy()
    cm=count_schedule(counts,length) if condition=='COUNT90' else None
    chunks={};events={k:[] for k in ('i','t','reason','valid','pre_g','post_g','pre_z','post_z','target_error','dh','dg')}
    snap_times=tuple(t for t in SNAPSHOTS if t<=length)
    model.eval()
    for start in range(0,n,128):
        stop=start+128;b,u,r,e=tensors(target_free,start,stop)
        identity=torch.eye(3,dtype=b.dtype if b.is_floating_point() else torch.float32).expand(128,3,3)
        h=(r[:,0] if rotation=='ORIGINAL' else identity)@radial(model.encoder(b).reshape(-1,3,2))
        zlist=[];glist=[];ds=[];dposts=[];snap0=[];snap1=[];snapreset=[];snapvalid=[];snapg=[]
        g=gram(h);z=model.readout(g).squeeze(-1);d=distance(g.numpy(),z.numpy(),pg)
        zlist.append(z.numpy());ds.append(d);dposts.append(d)
        if condition=='NONE':glist.append(g.numpy())
        last=np.full(128,-100000,np.int64)
        for t in range(1,length+1):
            h=environment_step(model,h,u[:,t-1],r[:,t] if rotation=='ORIGINAL' else identity,e[:,t-1])
            g=gram(h);z=model.readout(g).squeeze(-1);gn=g.numpy();zn=z.numpy();d=distance(gn,zn,pg)
            zlist.append(zn);ds.append(d)
            if condition=='NONE':glist.append(gn)
            if command_diagnostics and t in snap_times:
                g0=gram(model.transition(h,torch.zeros(128,dtype=torch.int64))).numpy()
                g1=gram(model.transition(h,torch.ones(128,dtype=torch.int64))).numpy()
                proposal,audit=refresh_step(model,h,z,'EQUAL8',p)
                snap0.append(g0);snap1.append(g1);snapg.append(gn)
                snapreset.append(gram(proposal).numpy()-gn);snapvalid.append(audit['valid_event'])
            mask=attempt_mask(condition,t,d,theta,last,cm[start:stop] if cm is not None else None)
            if mask.any():
                index=np.flatnonzero(mask);ti=torch.from_numpy(index)
                method=condition if condition in ('FULL8','SCALE8','GAUGE8') else 'EQUAL8'
                changed,audit=refresh_step(model,h[ti],z[ti],method,p)
                postg=gram(changed);postz=model.readout(postg).squeeze(-1)
                events['i'].append(index+start);events['t'].append(np.full(len(index),t,np.int64))
                for key,value in (('reason',audit['invalid_reason']),('valid',audit['valid_event']),('pre_g',gn[index]),
                                  ('post_g',postg.numpy()),('pre_z',zn[index]),('post_z',postz.numpy()),
                                  ('target_error',audit['logit_target_error']),('dh',audit['state_delta_norm']),('dg',audit['gram_delta_norm'])):
                    events[key].append(value)
                # Clone to avoid mutating the arrays already stored for this step.
                h=h.clone();h[ti]=changed;last[mask]=t
                gp=gram(h);zp=model.readout(gp).squeeze(-1);dposts.append(distance(gp.numpy(),zp.numpy(),pg))
            else:dposts.append(d)
        for key,values in (('logits',zlist),('distance_pre',ds),('distance_post',dposts)):
            chunks.setdefault(key,[]).append(np.stack(values,1))
        if glist:chunks.setdefault('gram',[]).append(np.stack(glist,1))
        if command_diagnostics:
            for key,value in (('command_hold_gram',snap0),('command_flip_gram',snap1),('command_state_gram',snapg),('command_reset_delta',snapreset),('command_reset_valid',snapvalid)):
                chunks.setdefault(key,[]).append(np.stack(value,1))
    raw={key:np.concatenate(value,axis=0) for key,value in chunks.items()}
    for key,values in events.items():
        if values:raw['event_'+key]=np.concatenate(values)
        else:raw['event_'+key]=np.empty((0,3) if key in ('pre_g','post_g') else (0,),dtype=bool if key=='valid' else np.int64 if key in ('i','t','reason') else np.float64)
    actual=np.bincount(raw['event_i'],minlength=n).astype(np.int64)
    raw['attempt_counts']=actual;raw['theta']=np.array(theta);raw['snapshot_times']=np.array(snap_times if command_diagnostics else [],np.int64)
    if counts is not None:
        raw['source_adapt_counts']=counts.copy()
        if not np.array_equal(actual,counts):raise RuntimeError('COUNT not matched to own ADAPT')
    tau,censor=lower_tau(raw['distance_pre'],theta);raw['tau_lag']=tau;raw['tau_censored']=censor
    if not all(np.isfinite(v).all() for v in raw.values()):raise FloatingPointError('Nonfinite rollout')
    return raw
