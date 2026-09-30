"""Frozen prototype refresh, explicitly fitted with calibration labels."""
from __future__ import annotations
import numpy as np
import torch
from .learned_protected_pilot import LearnedMemory,gram,radial,tensors
from .composition_core import predict_full
from .protected_interventions import environment_step


def canonical_state(matrix: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    g=np.asarray(matrix,np.float64);g=(g+g.swapaxes(-1,-2))/2
    values,vectors=np.linalg.eigh(g)
    if not np.isfinite(values).all() or values.min() < -1e-8:raise ValueError('Non-PSD state Gram')
    root=(vectors*np.sqrt(np.maximum(values,0))[...,None,:])@vectors.swapaxes(-1,-2)
    state=np.zeros((*root.shape[:-2],3,2),np.float32);state[...,:2,:]=root
    return state,values


def fit_prototypes(model: LearnedMemory, calibration: dict) -> tuple[dict,dict]:
    z,g,h=predict_full(model,calibration);labels=calibration['targets'][:,-1].astype(np.int64)
    if set(labels.tolist())!={0,1}:raise ValueError('Both calibration classes required')
    matrices=h.astype(float).swapaxes(1,2)@h.astype(float)
    means=np.stack([matrices[labels==b].mean(0) for b in (0,1)])
    prototypes,eigenvalues=canonical_state(means)
    with torch.no_grad():pz=model.readout(gram(torch.from_numpy(prototypes))).squeeze(-1).numpy()
    raw={'prototypes':prototypes,'prototype_grams':means,'prototype_logits':pz,
         'calibration_logits':z,'calibration_gram':g,'calibration_states':h,'calibration_labels':labels,'eigenvalues':eigenvalues}
    summary={'calibration_accuracy':float(np.mean((z[:,-1]>=0)==labels)),
             'prototype_labels_correct':bool(np.array_equal(pz>=0,np.array([False,True]))),
             'min_eigenvalue':float(eigenvalues.min()),'negative_eigenvalues_clamped':int(np.sum(eigenvalues<0)),
             'calibration_counts':[int(np.sum(labels==b)) for b in (0,1)],'prototype_values':12,'prototype_bytes':48}
    return raw,summary


def refresh_rollout(model: LearnedMemory,data: dict,condition: str,prototypes: np.ndarray) -> dict:
    if condition not in ('NONE','GAUGE8','RESET8','RESET1'):raise ValueError(condition)
    n=len(data['b0']);length=data['commands'].shape[1]
    mask=np.zeros(length+1,bool)
    if condition!='NONE':mask[1:]=np.arange(1,length+1)%(1 if condition=='RESET1' else 8)==0
    collected={k:[] for k in ('logits','gram','post_logits','post_gram','selected_labels','state_delta_norm')}
    model.eval();p=torch.from_numpy(np.asarray(prototypes,dtype=np.float32))
    with torch.no_grad():
        for start in range(0,n,128):
            b,u,r,e=tensors(data,start,start+128);h=r[:,0]@radial(model.encoder(b).reshape(-1,3,2))
            g=gram(h);z=model.readout(g).squeeze(-1)
            zs=[z.numpy()];gs=[g.numpy()];postz=[z.numpy()];postg=[g.numpy()];selected=[np.full(len(b),-1,np.int64)];deltas=[np.zeros(len(b),np.float32)]
            for t in range(length):
                h=environment_step(model,h,u[:,t],r[:,t+1],e[:,t]);g=gram(h);z=model.readout(g).squeeze(-1)
                zs.append(z.numpy());gs.append(g.numpy());label=np.full(len(b),-1,np.int64)
                if mask[t+1]:
                    before_h=h
                    if condition=='GAUGE8':
                        hd=h.numpy().astype(float);canonical,_=canonical_state(hd.swapaxes(1,2)@hd);h=torch.from_numpy(canonical)
                    else:
                        index=(z>=0).to(torch.int64);label=index.numpy();h=p[index]
                    gp=gram(h);zp=model.readout(gp).squeeze(-1)
                    delta=torch.linalg.vector_norm((h-before_h).flatten(1),dim=1).numpy()
                else:gp,zp=g,z;delta=np.zeros(len(b),np.float32)
                postz.append(zp.numpy());postg.append(gp.numpy());selected.append(label);deltas.append(delta)
            for k,v in [('logits',zs),('gram',gs),('post_logits',postz),('post_gram',postg),('selected_labels',selected),('state_delta_norm',deltas)]:collected[k].append(np.stack(v,axis=1))
    result={k:np.concatenate(v) for k,v in collected.items()};result['refresh_mask']=mask
    return result


def transition_table(model: LearnedMemory,prototypes: np.ndarray) -> dict:
    logits=[]
    with torch.no_grad():
        h=torch.from_numpy(np.asarray(prototypes,np.float32))
        for u in (0,1):logits.append(model.readout(gram(model.transition(h,torch.full((2,),u,dtype=torch.int64)))).squeeze(-1).numpy())
    z=np.stack(logits,axis=1)
    return {'logits':z,'predictions':(z>=0).astype(np.int64),'expected':np.array([[0,1],[1,0]],np.int64)}
