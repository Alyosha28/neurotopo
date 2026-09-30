"""Registered prototype-shape/current-logit and current-shape/prototype-logit paths."""
from __future__ import annotations
import numpy as np
import torch
from .learned_protected_pilot import LearnedMemory,gram,radial,tensors
from .protected_interventions import environment_step
from .state_refresh_core import canonical_state

CONDITIONS=('NONE','GAUGE8','FULL8','EQUAL8','SCALE8')
# reason:0 valid/no event,1 denominator,2 ratio,3 nonfinite proposal,4 degenerate Gram,
# 5 logit mismatch,6 shape mismatch,7 sign mismatch. Mask distinguishes no event.


def weighted(g: np.ndarray) -> np.ndarray:
    v=np.asarray(g,float).copy();v[...,1]*=np.sqrt(2.);return v


def shape(g: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    v=weighted(g);norm=np.linalg.norm(v,axis=-1)
    return v/np.maximum(norm[...,None],1e-12),norm


@torch.no_grad()
def refresh_step(model: LearnedMemory,h: torch.Tensor,z: torch.Tensor,condition: str,p: torch.Tensor) -> tuple[torch.Tensor,dict]:
    if condition not in CONDITIONS[1:]:raise ValueError(condition)
    n=len(h);g=gram(h);gn=g.numpy().astype(float);zn=z.detach().numpy().astype(float)
    labels=(zn>=0).astype(np.int64);picked=p[torch.from_numpy(labels)]
    pg=gram(picked);pz=model.readout(pg).squeeze(-1).numpy().astype(float)
    bias=float(model.readout.bias.detach()[0]);q=zn-bias;qp=pz-bias
    ratio=np.ones(n);available=np.zeros(n,bool);reason=np.zeros(n,np.int64)
    selected=np.full(n,-1,np.int64);candidate=h
    if condition=='GAUGE8':
        hd=h.numpy().astype(float);candidate=torch.from_numpy(canonical_state(hd.swapaxes(1,2)@hd)[0])
    elif condition=='FULL8':candidate=picked;selected=labels
    else:
        selected=labels
        denominator=qp if condition=='EQUAL8' else q
        numerator=q if condition=='EQUAL8' else qp
        available=np.abs(denominator)>1e-6
        reason[~available]=1
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            ratio=np.divide(numerator,denominator,out=np.ones(n),where=available)
        finite_ratio=np.isfinite(ratio);bad=available&((ratio<=0)|~finite_ratio)
        reason[bad]=2;available&=finite_ratio
        usable=reason==0;factor=np.where(usable,ratio,1.)
        base=picked if condition=='EQUAL8' else h
        with np.errstate(over='ignore',invalid='ignore'):
            proposed=(base.numpy().astype(float)*np.sqrt(factor)[:,None,None]).astype(np.float32)
        finite_h=np.isfinite(proposed).all(axis=(1,2));reason[usable&~finite_h]=3
        proposed=np.where((reason==0)[:,None,None],proposed,h.numpy())
        candidate=torch.from_numpy(proposed)
        cg=gram(candidate);finite_g=torch.isfinite(cg).all(1).numpy();reason[(reason==0)&~finite_g]=3
        candidate=torch.where(torch.from_numpy(reason==0)[:,None,None],candidate,h)
        cg=gram(candidate);cz=model.readout(cg).squeeze(-1).numpy().astype(float)
        finite_z=np.isfinite(cz);reason[(reason==0)&~finite_z]=3
        ref=pg.numpy().astype(float) if condition=='EQUAL8' else gn
        sh,ns=shape(cg.numpy());sr,nr=shape(ref)
        reason[(reason==0)&((ns<1e-12)|(nr<1e-12))]=4
        target=zn if condition=='EQUAL8' else pz
        reason[(reason==0)&(np.abs(cz-target)>1e-5)]=5
        reason[(reason==0)&(np.linalg.norm(sh-sr,axis=1)>1e-5)]=6
        if condition=='EQUAL8':reason[(reason==0)&((cz>=0)!=(zn>=0))]=7
        candidate=torch.where(torch.from_numpy(reason==0)[:,None,None],candidate,h)
    postg=gram(candidate);postz=model.readout(postg).squeeze(-1)
    gp=postg.numpy().astype(float);zp=postz.numpy().astype(float)
    ref=pg.numpy().astype(float) if condition in ('FULL8','EQUAL8') else gn
    target=zn if condition in ('GAUGE8','EQUAL8') else pz
    shp,npost=shape(gp);shr,nref=shape(ref)
    logerr=np.abs(zp-target);shapeerr=np.linalg.norm(shp-shr,axis=1)
    if condition=='GAUGE8':reason[(logerr>1e-5)|(np.max(np.abs(gp-gn),axis=1)>1e-5)]=5
    delta=np.linalg.norm(weighted(gp-gn),axis=1)
    audit={'selected_labels':selected,'valid_event':reason==0,'invalid_reason':reason,
           'q_current':q,'q_proto':qp,'proposed_scale':np.where(np.isfinite(ratio),ratio,0.),
           'ratio_available':available,'logit_target_error':logerr,'shape_error':shapeerr,
           'state_delta_norm':torch.linalg.vector_norm((candidate-h).flatten(1),dim=1).numpy(),
           'gram_delta_norm':delta,'prototype_distance_before':np.linalg.norm(weighted(gn-pg.numpy()),axis=1),
           'prototype_distance_after':np.linalg.norm(weighted(gp-pg.numpy()),axis=1),
           'post_state_norm':torch.linalg.vector_norm(candidate.flatten(1),dim=1).numpy()}
    return candidate,audit


def refresh_rollout(model: LearnedMemory,data: dict,condition: str,prototypes: np.ndarray) -> dict:
    if condition not in CONDITIONS:raise ValueError(condition)
    n=len(data['b0']);length=data['commands'].shape[1];mask=np.zeros(length+1,bool)
    if condition!='NONE':mask[8::8]=True
    p=torch.from_numpy(prototypes.astype(np.float32));collected={};model.eval()
    keys=('selected_labels','valid_event','invalid_reason','q_current','q_proto','proposed_scale','ratio_available','logit_target_error','shape_error','state_delta_norm','gram_delta_norm','prototype_distance_before','prototype_distance_after','post_state_norm')
    with torch.no_grad():
        for start in range(0,n,128):
            b,u,r,e=tensors(data,start,start+128);h=r[:,0]@radial(model.encoder(b).reshape(-1,3,2));g=gram(h);z=model.readout(g).squeeze(-1)
            arrays={'logits':[z.numpy()],'gram':[g.numpy()],'post_logits':[z.numpy()],'post_gram':[g.numpy()]}
            def empty():
                return {k:np.full(len(b),-1,np.int64) if k=='selected_labels' else np.zeros(len(b),bool) if k in ('valid_event','ratio_available') else np.zeros(len(b),np.int64) if k=='invalid_reason' else np.zeros(len(b),float) for k in keys}
            for k,v in empty().items():arrays[k]=[v]
            for t in range(length):
                h=environment_step(model,h,u[:,t],r[:,t+1],e[:,t]);g=gram(h);z=model.readout(g).squeeze(-1)
                arrays['logits'].append(z.numpy());arrays['gram'].append(g.numpy())
                if mask[t+1]:h,details=refresh_step(model,h,z,condition,p);gp=gram(h);zp=model.readout(gp).squeeze(-1)
                else:details=empty();gp,zp=g,z
                arrays['post_logits'].append(zp.numpy());arrays['post_gram'].append(gp.numpy())
                for k,v in details.items():arrays[k].append(v)
            for k,v in arrays.items():collected.setdefault(k,[]).append(np.stack(v,1))
    result={k:np.concatenate(v) for k,v in collected.items()};result['refresh_mask']=mask
    return result
