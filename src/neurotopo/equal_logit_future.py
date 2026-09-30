"""DYN-0001 frozen one-shot future-observability diagnostic."""
from __future__ import annotations
import argparse
import copy
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from .learned_protected_pilot import LearnedMemory, generate, gram, radial, state_hash, tensors
from .protected_interventions import environment_step
from .refresh_factor_core import CONDITIONS, refresh_step, weighted
from .order_observation import sha
from .run import git_output, source_fingerprints

SEEDS=tuple(range(25001,25013))+(23005,)
ARMS=('REL','INV')
CELLS=('LONG_H','LONG_U')
LAGS=(1,2,4,8,16,32,64,128,248,504,1016)
EPS=1e-3
CAP=1800
ROOT=Path('artifacts/EXP-0025/v2')


def _json(p:Path)->dict[str,Any]:return json.loads(p.read_text())
def _save(path:Path,value:dict)->None:
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def _extend(original:dict[str,np.ndarray],seed:int,cell:str)->dict[str,np.ndarray]:
    if cell not in CELLS or original['commands'].shape!=(512,256):raise ValueError('Wrong frozen prefix')
    mode,mode_id=('HOLD',1) if cell=='LONG_H' else ('UPDATE',2)
    continuation=generate(512,769,[266001,seed,mode_id,1024],mode,np.pi,0.)
    commands=np.concatenate((original['commands'],continuation['commands'][:,1:]),axis=1)
    common=np.concatenate((original['common'],continuation['common'][:,1:769]),axis=1)
    leak=np.concatenate((original['leak'],continuation['leak'][:,1:769]),axis=1)
    targets=np.bitwise_xor.accumulate(np.column_stack((original['b0'],commands)),axis=1).astype(np.float32)
    if commands.shape!=(512,1024) or common.shape!=(512,1025,3,3) or leak.shape!=(512,1024,3,3):
        raise RuntimeError('Extension shape mismatch')
    if not np.array_equal(targets[:,:257],original['targets']):raise RuntimeError('Original target prefix changed')
    result={'b0':original['b0'].copy(),'commands':commands,'common':common,'leak':leak,'targets':targets}
    for name in ('b0','commands','common','leak'):
        prefix=result[name] if name=='b0' else result[name][:,:257] if name=='common' else result[name][:,:256]
        if not np.array_equal(prefix,original[name]):raise RuntimeError('Frozen input prefix changed: '+name)
    return result


def _prefix(model:LearnedMemory,data:dict[str,np.ndarray])->torch.Tensor:
    chunks=[]
    with torch.no_grad():
        for start in range(0,512,128):
            b,u,r,e=tensors(data,start,start+128)
            h=r[:,0]@radial(model.encoder(b).reshape(-1,3,2))
            for t in range(8):h=environment_step(model,h,u[:,t],r[:,t+1],e[:,t])
            chunks.append(h)
    return torch.cat(chunks)


def _first_step_tangent(model:LearnedMemory,h:torch.Tensor,candidate:torch.Tensor,
                        valid:np.ndarray,data:dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    """Float64 Gram tangent; NaN marks unavailable local predictions."""
    n=len(h);available=np.zeros(n,bool)
    arrays={name:np.full(n,np.nan,np.float64) for name in
            ('current_tangent','next_tangent','double_exact_next','float_double_delta_error','gram_condition')}
    if not valid.any():return {'available':available,**arrays}
    model64=copy.deepcopy(model).double();model64.eval()
    base=h.detach().double();post=candidate.detach().double()
    g=base.transpose(1,2)@base;gp=post.transpose(1,2)@post;delta=gp-g
    eigen=torch.linalg.eigvalsh(g).numpy();condition=np.divide(eigen[:,1],eigen[:,0],out=np.full(n,np.inf),where=eigen[:,0]>0)
    ok=valid & np.isfinite(condition) & (eigen[:,0]>1e-8) & (condition<=1e6)
    arrays['gram_condition'][valid]=condition[valid]
    if not ok.any():return {'available':available,**arrays}
    indices=np.flatnonzero(ok);i=torch.from_numpy(indices)
    # dG = H^T dH + dH^T H = delta exactly for full-rank H.
    direction=.5*base[i]@torch.linalg.solve(g[i],delta[i])
    recovered=base[i].transpose(1,2)@direction+direction.transpose(1,2)@base[i]
    if torch.max(torch.abs(recovered-delta[i])).item()>1e-7:raise RuntimeError('Gram tangent identity failed')
    u=torch.from_numpy(data['commands'][indices,8]);r=torch.from_numpy(data['common'][indices,9]).double();e=torch.from_numpy(data['leak'][indices,8]).double()
    with torch.no_grad():
        def next_z(state:torch.Tensor)->torch.Tensor:
            return model64.readout(gram(environment_step(model64,state,u,r,e))).squeeze(-1)
        plus=next_z(base[i]+EPS*direction);minus=next_z(base[i]-EPS*direction)
        exact=next_z(post[i])-next_z(base[i]);central=(plus-minus)/(2*EPS)
        w=model64.readout.weight[0]
        current=w[0]*delta[i,0,0]+w[1]*delta[i,0,1]+w[2]*delta[i,1,1]
    if not torch.isfinite(central).all() or not torch.isfinite(exact).all():raise FloatingPointError('Nonfinite tangent')
    arrays['current_tangent'][indices]=current.numpy();arrays['next_tangent'][indices]=central.numpy();arrays['double_exact_next'][indices]=exact.numpy()
    available[indices]=True
    return {'available':available,**arrays}


def _one(model:LearnedMemory,data:dict[str,np.ndarray],proto:np.ndarray,
         parent:Path,seed:int,arm:str,cell:str)->tuple[dict,dict[str,np.ndarray]]:
    n,t=data['commands'].shape
    if (n,t)!=(512,1024):raise ValueError('Wrong frozen cell shape')
    h0=_prefix(model,data);g0=gram(h0)
    with torch.no_grad():z0=model.readout(g0).squeeze(-1)
    with np.load(parent/str(seed)/arm/'test'/'NONE'/f'{cell}.npz') as prev:
        old_z=prev['logits'][:,8:];old_g8=prev['gram'][:,8]
    if not np.array_equal(z0.numpy(),old_z[:,0]) or not np.array_equal(g0.numpy(),old_g8):
        raise RuntimeError('Frozen prefix differs from EXP0025')
    p=torch.from_numpy(proto.astype(np.float32))
    all_logits=[];all_postgram=[];validity=[];reasons=[];local=None
    cell_start=time.monotonic();cpu_start=time.process_time()
    for condition in CONDITIONS:
        with torch.no_grad():
            if condition=='NONE':
                current=h0;details={'valid_event':np.zeros(n,bool),'invalid_reason':np.zeros(n,np.int64)}
            else:current,details=refresh_step(model,h0,z0,condition,p)
            postg=gram(current).numpy();postz=model.readout(gram(current)).squeeze(-1).numpy()
            with np.load(parent/str(seed)/arm/'test'/condition/f'{cell}.npz') as frozen:
                if not np.array_equal(postg,frozen['post_gram'][:,8]) or not np.array_equal(postz,frozen['post_logits'][:,8]):
                    raise RuntimeError('First refresh differs from EXP0025')
                if condition!='NONE' and not np.array_equal(details['valid_event'],frozen['valid_event'][:,8]):
                    raise RuntimeError('Frozen event validity changed')
            if condition=='GAUGE8' and (np.abs(postz-z0.numpy()).max()>1e-5 or np.abs(postg-g0.numpy()).max()>1e-5):
                raise RuntimeError('Gauge control fidelity failed')
            if condition=='EQUAL8':
                valid=details['valid_event']
                if valid.any() and np.abs(postz[valid]-z0.numpy()[valid]).max()>1e-5:
                    raise RuntimeError('Equal current logit constraint failed')
                local=_first_step_tangent(model,h0,current,valid,data)
            logits=np.empty((n,1017),np.float32);logits[:,0]=z0.numpy()
            for start in range(0,n,128):
                stop=start+128;branch=current[start:stop]
                for step in range(8,1024):
                    u=torch.from_numpy(data['commands'][start:stop,step])
                    r=torch.from_numpy(data['common'][start:stop,step+1])
                    e=torch.from_numpy(data['leak'][start:stop,step])
                    branch=environment_step(model,branch,u,r,e)
                    logits[start:stop,step-7]=model.readout(gram(branch)).squeeze(-1).numpy()
            if not np.isfinite(logits).all():raise FloatingPointError('Nonfinite branch')
            all_logits.append(logits);all_postgram.append(postg)
            validity.append(details['valid_event']);reasons.append(details['invalid_reason'])
    zs=np.stack(all_logits);gs=np.stack(all_postgram);valid=np.stack(validity);reason=np.stack(reasons)
    # t8 pre scores are common by construction; suffix t9..1024 has no further refreshes.
    baseline_max_error=float(np.max(np.abs(zs[0,:,:249].astype(float)-old_z.astype(float))))
    if baseline_max_error>1e-6:raise RuntimeError('One-shot NONE differs from frozen full forward')
    for j in (3,4):
        if not np.array_equal(zs[j,~valid[j]],zs[0,~valid[j]]):raise RuntimeError('Fallback branch differs from NONE')
    if local is None:raise RuntimeError('Missing registered tangent')
    exact_one=zs[3,:,1].astype(float)-zs[0,:,1].astype(float)
    local['float_double_delta_error'][local['available']]=np.abs(local['double_exact_next'][local['available']]-exact_one[local['available']])
    raw={'logits':zs,'gram_at_8_pre':g0.numpy(),'gram_at_8_post':gs,
         'logit_at_8_pre':z0.numpy(),'valid_event_at_8':valid,'invalid_reason_at_8':reason,
         'tangent_available':local['available']}
    for name in ('current_tangent','next_tangent','double_exact_next','float_double_delta_error','gram_condition'):
        raw[name]=local[name]
    target=data['targets'][:,8:].astype(bool);base=zs[0]
    summaries={}
    for j,condition in enumerate(CONDITIONS):
        difference=zs[j].astype(float)-base.astype(float)
        sign=(zs[j]>=0)!=(base>=0)
        first=np.argmax(sign[:,1:],axis=1)+1;exists=sign[:,1:].any(axis=1)
        dg=np.linalg.norm(weighted(gs[j].astype(float)-g0.numpy().astype(float)),axis=1)
        def group(mask:np.ndarray)->dict[str,Any]:
            count=int(mask.sum())
            return {'n':count,'final_accuracy':float(np.mean((zs[j,mask,-1]>=0)==target[mask,-1])) if count else None}
        summaries[condition]={'attempts':n if j else 0,'valid_count':int(valid[j].sum()),
            'invalid_reasons':{str(x):int((reason[j]==x).sum()) for x in range(1,8)},
            'immediate_logit_abs_max':float(np.abs((model.readout(torch.from_numpy(gs[j])).squeeze(-1).detach().numpy()-z0.numpy())).max()),
            'gram_displacement_mean':float(dg.mean()),'gram_displacement_max':float(dg.max()),
            'final_accuracy':float(np.mean((zs[j,:,-1]>=0)==target[:,-1])),
            'accuracy_by_horizon':{str(horizon):float(np.mean((zs[j,:,horizon-8]>=0)==target[:,horizon-8]))
                                   for horizon in (256,512,1024)},
            'suffix_accuracy':float(np.mean((zs[j,:,1:]>=0)==target[:,1:])),
            'first_sign_divergence_count':int(exists.sum()),
            'first_sign_divergence_mean_lag':float(first[exists].mean()) if exists.any() else None,
            'lags':{str(k):{'mean_abs_logit_diff':float(np.abs(difference[:,k]).mean()),
                              'max_abs_logit_diff':float(np.abs(difference[:,k]).max()),
                              'sign_diff_fraction':float(sign[:,k].mean())} for k in LAGS},
            'by_true_initial_label':{str(label):group(data['b0']==label) for label in (0,1)},
            'by_pre_sign':{str(label):group((z0.numpy()>=0)==label) for label in (0,1)}}
    usable=local['available'];x=local['next_tangent'][usable];y=exact_one[usable]
    correlation=float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>1e-12 and np.std(y)>1e-12 else None
    if correlation is not None and not np.isfinite(correlation):correlation=None
    tangent={'available_count':int(usable.sum()),'unavailable_count':int((valid[3]&~usable).sum()),
             'valid_equal_count':int(valid[3].sum()),
             'current_tangent_abs_max':float(np.max(np.abs(local['current_tangent'][usable]))) if usable.any() else None,
             'next_tangent_mean_abs':float(np.mean(np.abs(x))) if len(x) else None,
             'exact_next_mean_abs':float(np.mean(np.abs(y))) if len(y) else None,
             'next_tangent_exact_correlation':correlation,
             'same_sign_nonzero_fraction':float(np.mean(np.sign(x)==np.sign(y))) if len(x) else None,
             'mean_abs_residual':float(np.mean(np.abs(x-y))) if len(x) else None,
             'float_double_delta_error_max':float(np.max(local['float_double_delta_error'][usable])) if usable.any() else None}
    result={'seed':seed,'arm':arm,'cell':cell,'conditions':summaries,'tangent':tangent,
            'baseline_max_abs_logit_error':baseline_max_error,
            'wall_seconds':time.monotonic()-cell_start,'cpu_seconds':time.process_time()-cpu_start,
            'suffix_transitions':len(CONDITIONS)*n*(1024-8),
            'float64_tangent_next_step_evaluations':4*int(usable.sum()),
            'frozen_model_state_sha256':state_hash(model)}
    return result,raw


def run(output:Path, parent:Path=ROOT, suffix_source:Path|None=None)->Path:
    if not output.is_absolute():raise ValueError('Absolute output required')
    if suffix_source is None or not suffix_source.is_absolute():raise ValueError('Absolute frozen v2 suffix source required')
    if output.exists():raise FileExistsError(output)
    started=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True)
    source=source_fingerprints();rows=[];files={};data_files={}
    inventory_path=Path('research/design/dyn0001_v2_suffix_inventory.json');inventory=_json(inventory_path)
    if str(suffix_source.resolve())!=str(Path(inventory['source']).resolve()) or len(inventory['files'])!=26:
        raise RuntimeError('Wrong fixed v2 suffix source')
    manifest={'study':'DYN-0001','version':'v3',
              'scope':'Exposed frozen models and T256 prefixes; new fixed T1024 suffixes; no training',
              'source_sha256':source,'git_commit':git_output('rev-parse','HEAD'),
              'analysis_script_sha256':sha(Path('scripts/analyze_dyn0001.py')),
              'design_sha256':sha(Path('research/design/dyn0001_equal_logit_future_observability_v3.md')),
              'suffix_inventory_sha256':sha(inventory_path),'v2_failure_sha256':sha(suffix_source/'failure.json'),
              'parent_manifest_sha256':sha(parent/'manifest.json'),'parent_completion_sha256':sha(parent/'completion.json'),
              'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),
              'conditions':CONDITIONS,'cells':CELLS,'seeds':SEEDS,'arms':ARMS,'lags':LAGS,
              'epsilon':EPS,'wall_cap_seconds':CAP,'training_updates':0,'prototype_fits':0,
              'new_suffix_streams':26,'extension_root_template':[266001,'seed','mode_id',1024]}
    _save(output/'manifest.json',manifest)
    try:
        # Hash-copy all v2-frozen inputs before loading any outcome model.
        for seed in SEEDS:
            for cell in CELLS:
                original_path=parent/'data'/str(seed)/f'test_{cell}.npz'
                input_path=output/'data'/str(seed)/f'{cell}.npz';input_path.parent.mkdir(parents=True,exist_ok=True)
                name=str(input_path.relative_to(output));frozen=suffix_source/name
                if sha(frozen)!=inventory['files'][name]['sha256']:raise RuntimeError('Frozen v2 suffix changed')
                with frozen.open('rb') as source_file,input_path.open('xb') as dest:shutil.copyfileobj(source_file,dest)
                if sha(input_path)!=inventory['files'][name]['sha256']:raise RuntimeError('Copied suffix hash mismatch')
                with np.load(original_path) as pf,np.load(input_path) as copied:
                    for key in ('b0','commands','common','leak','targets'):
                        prefix=copied[key] if key=='b0' else copied[key][:,:257] if key in ('common','targets') else copied[key][:,:256]
                        if not np.array_equal(prefix,pf[key]):raise RuntimeError('Frozen prefix mismatch '+key)
                data_files[str(input_path.relative_to(output))]={
                    'sha256':sha(input_path),'original_prefix_sha256':sha(original_path)}
        for seed in SEEDS:
            for arm in ARMS:
                checkpoint=parent/str(seed)/arm/'checkpoint-1000.pt'
                state=torch.load(checkpoint,map_location='cpu',weights_only=True)
                model=LearnedMemory(arm,seed=seed);model.load_state_dict(state['model']);model.eval()
                if state_hash(model)!=_json(parent/str(seed)/arm/'result.json')['frozen_model_state_sha256']:
                    raise RuntimeError('Frozen checkpoint state mismatch')
                with np.load(parent/str(seed)/arm/'prototypes.npz') as pf:proto=pf['prototypes'].copy()
                for cell in CELLS:
                    if time.monotonic()-started>CAP:raise TimeoutError('DYN0001 shared wall budget exhausted')
                    input_path=output/'data'/str(seed)/f'{cell}.npz'
                    with np.load(input_path) as f:data={k:f[k] for k in f.files}
                    row,raw=_one(model,data,proto,parent,seed,arm,cell)
                    folder=output/str(seed)/arm;folder.mkdir(parents=True,exist_ok=True)
                    path=folder/f'{cell}.npz'
                    with path.open('xb') as stream:np.savez_compressed(stream,**raw)
                    row.update(input_sha256=sha(input_path),checkpoint_sha256=sha(checkpoint),
                               prototype_sha256=sha(parent/str(seed)/arm/'prototypes.npz'),
                               raw_sha256=sha(path))
                    files[str(path.relative_to(output))]=row['raw_sha256'];rows.append(row)
                    print(json.dumps({'seed':seed,'arm':arm,'cell':cell,'status':'COMPLETED'}),flush=True)
        if len(rows)!=52 or len(data_files)!=26 or sum(r['suffix_transitions'] for r in rows)!=135_249_920:
            raise RuntimeError('Registered matrix/resource mismatch')
        completion={'status':'COMPLETED','rows':rows,'raw_sha256':files,'cells':52,'models':26,
                    'suffix_transitions':135_249_920,'training_updates':0,'prototype_fits':0,
                    'new_suffix_streams':26,'data_files':data_files,
                    'float64_tangent_next_step_evaluations':sum(r['float64_tangent_next_step_evaluations'] for r in rows),
                    'wall_seconds':time.monotonic()-started}
        _save(output/'completion.json',completion)
        if completion['wall_seconds']>CAP:raise TimeoutError('DYN0001 shared wall budget exhausted')
        return output
    except BaseException as error:
        _save(output/'failure.json',{'status':'INCOMPLETE','error':repr(error),'completed_cells':len(rows),
                                     'wall_seconds':time.monotonic()-started})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--parent',type=Path,default=ROOT)
    parser.add_argument('--suffix-source',type=Path,required=True)
    args=parser.parse_args();run(args.output,args.parent,args.suffix_source)
