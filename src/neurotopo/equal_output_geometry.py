"""Finite equal-logit rotations and held-out suffix execution for EXP0022."""
from __future__ import annotations
import numpy as np
import torch
from .learned_protected_pilot import LearnedMemory, gram, radial, tensors, rotations
from .protected_interventions import environment_step, rotation_matrix


def prefix_states(model: LearnedMemory, prefix: dict[str, np.ndarray]) -> np.ndarray:
    result=[]
    with torch.no_grad():
        for i in range(0,len(prefix['b0']),128):
            b0,u,r,e=tensors(prefix,i,i+128)
            h=r[:,0]@radial(model.encoder(b0).reshape(-1,3,2))
            for t in range(u.shape[1]):h=environment_step(model,h,u[:,t],r[:,t+1],e[:,t])
            result.append(h.numpy())
    return np.concatenate(result)


def suffix_data(seed: int, bits: np.ndarray, count: int, namespace: int) -> dict[str,np.ndarray]:
    """Unforced Bernoulli commands, independent of prefix/search/test streams."""
    streams=np.random.SeedSequence([namespace,seed]).spawn(2)
    cmd_rng,rot_rng=[np.random.default_rng(s) for s in streams]
    n=len(bits);u=cmd_rng.integers(0,2,(n,count,32),dtype=np.int64)
    b=np.broadcast_to(np.asarray(bits,dtype=np.int64)[:,None,None],(n,count,1))
    target=np.bitwise_xor.accumulate(np.concatenate((b,u),axis=2),axis=2).astype(np.float32)
    return {'commands':u,'common':rotations(rot_rng,(n,count,32),np.pi),'targets':target}


def _perpendicular(v: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """v[N,3], raw[N,K,3]; deterministic fallback does not use model targets."""
    unit=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
    projected=raw-np.sum(raw*unit[:,None,:],axis=-1,keepdims=True)*unit[:,None,:]
    size=np.linalg.norm(projected,axis=-1,keepdims=True);bad=size[...,0]<1e-10
    coordinate=np.eye(3)[np.argmin(np.abs(unit),axis=1)]
    fallback=coordinate-np.sum(coordinate*unit,axis=1,keepdims=True)*unit
    fallback/=np.maximum(np.linalg.norm(fallback,axis=1,keepdims=True),1e-12)
    return np.where(bad[...,None],fallback[:,None,:],projected/np.maximum(size,1e-12)),bad


def current_logits(model: LearnedMemory, h: np.ndarray) -> np.ndarray:
    shape=h.shape[:-2]
    with torch.no_grad():
        result=model.readout(gram(torch.from_numpy(h.reshape(-1,3,2).astype(np.float32)))).squeeze(-1).numpy()
    return result.reshape(shape)


def candidate_bank(model: LearnedMemory, h: np.ndarray, seed: int, r: float) -> dict[str,np.ndarray]:
    """Rotate one eigenbasis column; keep quadratic readout and total norm."""
    if r not in (.01,.05,.15):raise ValueError('Unregistered radius')
    original=np.asarray(h,dtype=np.float64);n=len(h);norm=np.linalg.norm(original,axis=(1,2))
    w=model.readout.weight.detach().numpy()[0].astype(np.float64)
    q=np.array([[w[0],w[1]/2],[w[1]/2,w[2]]]);values,u=np.linalg.eigh(q)
    transformed=original@u;column_norm=np.linalg.norm(transformed,axis=1)
    chosen=column_norm.argmax(axis=1);v=transformed[np.arange(n),:,chosen]
    size=np.linalg.norm(v,axis=1)
    raw=np.random.default_rng(np.random.SeedSequence([222004,seed])).normal(size=(n,16,3))
    tangent,fallback=_perpendicular(v,raw)
    angle=2*np.arcsin(np.clip(r*norm/(2*np.maximum(size,1e-12)),0,1))
    rotated=np.cos(angle)[:,None,None]*v[:,None,:]+np.sin(angle)[:,None,None]*size[:,None,None]*tangent
    changed=np.broadcast_to(transformed[:,None,:,:],(n,16,3,2)).copy()
    for i in range(n):changed[i,:,:,chosen[i]]=rotated[i]
    states=(changed@u.T).astype(np.float32)
    # Matched common-rotation control: axis perpendicular to largest original column.
    largest=np.linalg.norm(original,axis=1).argmax(axis=1)
    dominant=original[np.arange(n),:,largest]
    raw_axis=np.random.default_rng(np.random.SeedSequence([222005,seed])).normal(size=(n,1,3))
    axis,sham_fallback=_perpendicular(dominant,raw_axis);axis=axis[:,0]
    parallel=np.einsum('ni,nij->nj',axis,original)
    perp=np.sqrt(np.maximum(norm**2-np.sum(parallel**2,axis=1),0))
    theta=2*np.arcsin(np.clip(r*norm/(2*np.maximum(perp,1e-12)),0,1))
    sham_rotation=rotation_matrix(axis,theta)
    sham=(sham_rotation@original).astype(np.float32)
    base_logit=current_logits(model,h)
    logit_error=np.abs(current_logits(model,states)-base_logit[:,None])
    displacement=np.linalg.norm(states.astype(float)-original[:,None],axis=(2,3))
    radius_error=np.abs(displacement/np.maximum(r*norm[:,None],1e-12)-1)
    norm_error=np.abs(np.linalg.norm(states.astype(float),axis=(2,3))/np.maximum(norm[:,None],1e-12)-1)
    sham_logit_error=np.abs(current_logits(model,sham)-base_logit)
    sham_displacement=np.linalg.norm(sham.astype(float)-original,axis=(1,2))
    sham_radius_error=np.abs(sham_displacement/np.maximum(r*norm,1e-12)-1)
    sham_norm_error=np.abs(np.linalg.norm(sham.astype(float),axis=(1,2))/np.maximum(norm,1e-12)-1)
    valid=(norm[:,None]>1e-10)&(logit_error<=1e-5)&(radius_error<=1e-5)&(norm_error<=1e-5)
    sham_valid=(norm>1e-10)&(sham_logit_error<=1e-5)&(sham_radius_error<=1e-5)&(sham_norm_error<=1e-5)
    return {'states':states,'sham':sham,'valid':valid,'sham_valid':sham_valid,
            'logit_error':logit_error,'radius_error':radius_error,'norm_error':norm_error,
            'displacement':displacement,'sham_logit_error':sham_logit_error,'sham_radius_error':sham_radius_error,
            'sham_norm_error':sham_norm_error,'sham_displacement':sham_displacement,
            'directions':raw,'tangents':tangent,'eigbasis':u,'eigenvalues':values,'chosen_column':chosen,
            'direction_fallback':fallback,'sham_direction_fallback':sham_fallback[:,0],
            'sham_axis':axis,'sham_rotation':sham_rotation,'base_norm':norm,'base_logit':base_logit}


def rollout(model: LearnedMemory, states: np.ndarray, suffix: dict[str,np.ndarray]) -> tuple[np.ndarray,np.ndarray]:
    """States[N,K,3,2], returned logits[N,K,S,T+1]; no encoding repeated."""
    n,k=states.shape[:2];s,t=suffix['commands'].shape[1:]
    if states.shape[2:]!=(3,2) or suffix['commands'].shape[0]!=n:raise ValueError('Shape mismatch')
    repeated=np.broadcast_to(states[:,:,None],(n,k,s,3,2)).reshape(-1,3,2)
    prefix_index=np.repeat(np.arange(n),k*s);suffix_index=np.tile(np.arange(s),n*k)
    zs=[];gs=[]
    with torch.no_grad():
        for start in range(0,len(repeated),128):
            stop=start+128;ni=prefix_index[start:stop];si=suffix_index[start:stop]
            h=torch.from_numpy(np.array(repeated[start:stop],copy=True))
            commands=torch.from_numpy(suffix['commands'][ni,si]);common=torch.from_numpy(suffix['common'][ni,si])
            g=gram(h);z=[model.readout(g).squeeze(-1)];gseq=[g]
            for j in range(t):
                h=model.transition(common[:,j]@h,commands[:,j])
                g=gram(h);z.append(model.readout(g).squeeze(-1));gseq.append(g)
            zs.append(torch.stack(z,1).numpy());gs.append(torch.stack(gseq,1).numpy())
    return np.concatenate(zs).reshape(n,k,s,t+1),np.concatenate(gs).reshape(n,k,s,t+1,3)


def select_candidates(search_logits: np.ndarray, search_targets: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    z=search_logits[:,:,:,1:].astype(float);target=search_targets[:,None,:,1:]
    scores=(np.logaddexp(0,z)-target*z).mean(axis=(2,3))
    if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite search score')
    selected=np.argmax(np.where(valid,scores,-np.inf),axis=1)
    return selected,scores
