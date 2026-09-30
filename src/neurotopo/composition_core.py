"""EXP0023 readout composition regularizer and frozen diagnostics."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from .learned_protected_pilot import LearnedMemory,gram,radial,tensors
from .protected_interventions import environment_step

VARIANTS=('REL_BASE','REL_CYCLE','REL_EXTRA','INV_BASE')


def forward_state(model: LearnedMemory, data: dict[str,np.ndarray]) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    b0,u,r,e=tensors(data)
    h=r[:,0]@radial(model.encoder(b0).reshape(-1,3,2))
    g=gram(h);zs=[model.readout(g).squeeze(-1)];gs=[g]
    for t in range(u.shape[1]):
        h=environment_step(model,h,u[:,t],r[:,t+1],e[:,t])
        g=gram(h);zs.append(model.readout(g).squeeze(-1));gs.append(g)
    return torch.stack(zs,1),torch.stack(gs,1),h


def branch_logits(model: LearnedMemory, h: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    base=h.detach();zero=torch.zeros(len(h),dtype=torch.int64,device=h.device);one=torch.ones_like(zero)
    hold=model.transition(base,zero)
    flip=model.transition(base,one)
    flip2=model.transition(flip,one)
    return model.readout(gram(hold)).squeeze(-1),model.readout(gram(flip2)).squeeze(-1),hold,flip2


def auxiliary_loss(model: LearnedMemory, h: torch.Tensor, z0: torch.Tensor,
                   target: torch.Tensor, variant: str) -> torch.Tensor:
    if variant not in ('REL_CYCLE','REL_EXTRA'):raise ValueError(variant)
    zh,zff,_,_=branch_logits(model,h)
    if variant=='REL_CYCLE':
        anchor=z0.detach()
        return ((zh-anchor).square()+(zff-anchor).square()).mean()/2
    return (F.binary_cross_entropy_with_logits(zh,target)+F.binary_cross_entropy_with_logits(zff,target))/2


def loss_terms(model: LearnedMemory, data: dict[str,np.ndarray], variant: str,
               coefficient: float=.1) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    if variant not in VARIANTS or coefficient!=.1:raise ValueError('Unregistered training variant')
    z,_,h=forward_state(model,data);target=torch.from_numpy(data['targets'])
    main=F.binary_cross_entropy_with_logits(z,target)
    aux=auxiliary_loss(model,h,z[:,-1],target[:,-1],variant) if variant in ('REL_CYCLE','REL_EXTRA') else main.new_zeros(())
    return main+coefficient*aux,main,aux


def predict_full(model: LearnedMemory, data: dict[str,np.ndarray]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    zs=[];gs=[];hs=[];model.eval()
    with torch.no_grad():
        for i in range(0,len(data['b0']),128):
            part={k:v[i:i+128] for k,v in data.items()};z,g,h=forward_state(model,part)
            zs.append(z.numpy());gs.append(g.numpy());hs.append(h.numpy())
    return np.concatenate(zs),np.concatenate(gs),np.concatenate(hs)


def diagnostic_summary(raw: dict[str,np.ndarray]) -> dict[str,float]:
    z=raw['z0'].astype(float);dh=raw['z_hold'].astype(float)-z;df=raw['z_flip2'].astype(float)-z
    hold=float(np.mean(dh**2));flip=float(np.mean(df**2));d=(hold+flip)/2;energy=float(np.mean(z**2))
    g=raw['gram0'].astype(float);den=np.maximum(np.linalg.norm(g,axis=1),1e-8)
    return {'D':d,'D_rel':d/(energy+1e-6),'hold_mse':hold,'flip2_mse':flip,
            'hold_abs_max':float(np.max(np.abs(dh))),'flip2_abs_max':float(np.max(np.abs(df))),
            'logit_energy':energy,'current_accuracy':float(np.mean((z>=0)==raw['targets'])),
            'signed_margin_mean':float(np.mean((2*raw['targets']-1)*z)),
            'gram_hold_relative_mean':float(np.mean(np.linalg.norm(raw['gram_hold'].astype(float)-g,axis=1)/den)),
            'gram_flip2_relative_mean':float(np.mean(np.linalg.norm(raw['gram_flip2'].astype(float)-g,axis=1)/den))}


def diagnostic(model: LearnedMemory, data: dict[str,np.ndarray]) -> tuple[dict[str,np.ndarray],dict[str,float]]:
    z,g,h=predict_full(model,data)
    with torch.no_grad():
        zh,zff,hh,hff=branch_logits(model,torch.from_numpy(h))
        raw={'states':h,'gram0':g[:,-1],'z0':z[:,-1],'z_hold':zh.numpy(),'z_flip2':zff.numpy(),
             'gram_hold':gram(hh).numpy(),'gram_flip2':gram(hff).numpy(),
             'targets':data['targets'][:,-1].copy(),'prefix_logits':z,'prefix_gram':g}
    hold_sq=(raw['z_hold'].astype(float)-raw['z0'].astype(float))**2
    flip_sq=(raw['z_flip2'].astype(float)-raw['z0'].astype(float))**2
    d=(hold_sq+flip_sq)/2
    raw.update(hold_squared=hold_sq,flip2_squared=flip_sq,d_example=d,d_rel_example=d/(np.mean(raw['z0'].astype(float)**2)+1e-6))
    return raw,diagnostic_summary(raw)


def gram_transition(model: LearnedMemory, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Algebraic reference for the no-leak invariant quotient; not a new model."""
    if model.arm not in ('REL','INV'):raise ValueError('RAW has no Gram-only closure')
    code=F.one_hot(u,2).to(g.dtype);change=model.controller(torch.cat((g,code),1))
    if model.arm=='INV':return gram(radial(change.reshape(-1,3,2)))
    matrix=torch.stack((g[:,0],g[:,1],g[:,1],g[:,2]),1).reshape(-1,2,2)
    b=torch.eye(2,dtype=g.dtype,device=g.device)+change.reshape(-1,2,2)
    q=b.transpose(1,2)@matrix@b;r2=q.diagonal(dim1=1,dim2=2).clamp_min(0)
    safe=torch.sqrt(r2.clamp_min(1e-8));exact=torch.tanh(1.5*safe)/safe
    series=1.5-(1.5**3/3)*r2+(2*1.5**5/15)*r2.square()
    scale=torch.where(r2<1e-8,series,exact);next_g=q*scale[:,:,None]*scale[:,None,:]
    return torch.stack((next_g[:,0,0],next_g[:,0,1],next_g[:,1,1]),1)
