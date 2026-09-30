"""EXP0027: finite reset closure and state-matched future computation tests."""
from __future__ import annotations
import argparse,hashlib,itertools,json,time,sys
from pathlib import Path
import numpy as np
import torch
from .learned_protected_pilot import LearnedMemory,gram,radial,tensors,state_hash
from .protected_interventions import environment_step
from .refresh_factor_core import refresh_step
from .run import source_fingerprints

SEEDS=(25002,25003,25008,25010)
ARMS=('REL','INV');ANCHORS=(8,16,24,32)
BRANCHES=('NONE','EQUAL_before','EQUAL_after1','EQUAL_after2','FULL_before','FULL_after1','FULL_after2','GAUGE_before','GAUGE_after1','GAUGE_after2')


def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1048576):h.update(b)
    return h.hexdigest()

def read(p:Path):return json.loads(p.read_text())
def save(p:Path,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def save_npz(p:Path,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.savez_compressed(f,**d)

def words(k:int)->np.ndarray:return np.array(list(itertools.product((0,1),repeat=k)),np.int64)

def targets(y:np.ndarray,u:np.ndarray)->np.ndarray:
    return np.bitwise_xor.accumulate(np.column_stack((y,u)),axis=1)


@torch.no_grad()
def branch(model,h0:np.ndarray,u:np.ndarray,proto:np.ndarray,method:str='NONE',when:int=0,
           rotations:np.ndarray|None=None,leak:np.ndarray|None=None)->dict:
    n,length=u.shape
    if n%128 or h0.shape!=(n,3,2):raise ValueError('Use fixed128 batches')
    if method not in ('NONE','FULL','EQUAL','GAUGE') or not 0<=when<=length:raise ValueError('Bad intervention')
    p=torch.from_numpy(proto.astype(np.float32));chunks={}
    for start in range(0,n,128):
        stop=start+128;h=torch.from_numpy(h0[start:stop].copy());identity=torch.eye(3).expand(128,3,3)
        zpre=[];zpost=[];event_pre=None;event_post=None;event_pre_z=None;event_post_z=None
        valid=np.zeros(128,bool);reason=np.zeros(128,np.int64);scale=np.full(128,np.nan)
        for t in range(length+1):
            if t:
                r=identity if rotations is None else torch.from_numpy(rotations[start:stop,t-1])
                e=identity if leak is None else torch.from_numpy(leak[start:stop,t-1])
                h=environment_step(model,h,torch.from_numpy(u[start:stop,t-1]),r,e)
            z=model.readout(gram(h)).squeeze(-1);zpre.append(z.numpy())
            if method!='NONE' and t==when:
                event_pre=h.numpy().copy();event_pre_z=z.numpy().copy()
                h,a=refresh_step(model,h,z,method+'8',p)
                event_post=h.numpy().copy();event_post_z=model.readout(gram(h)).squeeze(-1).numpy()
                valid=a['valid_event'];reason=a['invalid_reason'];scale=a['proposed_scale']
            zp=model.readout(gram(h)).squeeze(-1);zpost.append(zp.numpy())
        if method=='NONE':
            event_pre=h0[start:stop].copy();event_post=event_pre.copy()
            event_pre_z=zpre[0];event_post_z=zpre[0]
        row={'pre_logits':np.stack(zpre,1),'post_logits':np.stack(zpost,1),
             'event_pre_h':event_pre,'event_post_h':event_post,'event_pre_z':event_pre_z,'event_post_z':event_post_z,
             'event_valid':valid,'event_reason':reason,'event_scale':scale,'final_h':h.numpy()}
        for key,value in row.items():chunks.setdefault(key,[]).append(value)
    raw={k:np.concatenate(v) for k,v in chunks.items()}
    if not np.isfinite(raw['pre_logits']).all() or not np.isfinite(raw['post_logits']).all():raise FloatingPointError('Nonfinite logits')
    return raw


def pad128(h,u):
    n=len(h);indices=np.arange(((n+127)//128)*128)%n
    return h[indices],u[indices],n


def block_test(model,proto:np.ndarray,k:int,method:str)->tuple[dict,dict]:
    w=words(k);labels=np.repeat([0,1],len(w));commands=np.tile(w,(2,1));h=proto[labels]
    hp,up,n=pad128(h,commands);raw=branch(model,hp,up,proto,method,k)
    raw={key:value[:n] for key,value in raw.items()};target=targets(labels,commands)
    correct=(raw['pre_logits']>=0)==target
    returns=np.all(raw['event_post_h']==proto[target[:,-1]],axis=(1,2))
    failures=[]
    for i in range(n):
        bad=np.flatnonzero(~correct[i])
        if len(bad):failures.append({'initial_label':int(labels[i]),'commands':commands[i].tolist(),'first_wrong_step':int(bad[0]),'pre_logits':raw['pre_logits'][i].tolist()})
    summary={'method':method,'k':k,'unique_cases':n,'padded_cases':len(hp),'all_time_correct_cases':int(correct.all(1).sum()),
             'canonical_return_cases':int(returns.sum()),'valid_events':int(raw['event_valid'].sum()),
             'all_correct_and_closed':bool(correct.all() and returns.all() and raw['event_valid'].all()),
             'scale_min':float(np.nanmin(raw['event_scale'])),'scale_max':float(np.nanmax(raw['event_scale'])),
             'failures':failures,'scope':'Exact finite block certificate eligible only for FULL; EQUAL sampled prototype starts retain continuous-scale limitation.'}
    raw.update(commands=commands,labels=labels,targets=target,canonical_return=returns)
    return summary,raw


@torch.no_grad()
def initial_states(model,labels:np.ndarray)->np.ndarray:
    allh=[]
    for i in range(0,len(labels),128):allh.append((torch.eye(3).expand(128,3,3)@radial(model.encoder(torch.from_numpy(labels[i:i+128])).reshape(-1,3,2))).numpy())
    return np.concatenate(allh)


def warmup(model,proto:np.ndarray)->tuple[dict,dict]:
    w=words(8);labels=np.repeat([0,1],len(w));commands=np.tile(w,(2,1));h=initial_states(model,labels)
    raw=branch(model,h,commands,proto,'FULL',8);target=targets(labels,commands)
    correct=(raw['pre_logits']>=0)==target;closed=np.all(raw['event_post_h']==proto[target[:,-1]],axis=(1,2))
    raw.update(commands=commands,labels=labels,targets=target)
    return {'cases':512,'all_time_correct':int(correct.all(1).sum()),'correct_canonical_entry':int(closed.sum()),'pass':bool(correct.all() and closed.all())},raw


@torch.no_grad()
def snapshots(model,data:dict)->dict[int,np.ndarray]:
    collected={t:[] for t in ANCHORS}
    for start in range(0,512,128):
        b,u,r,e=tensors(data,start,start+128);h=r[:,0]@radial(model.encoder(b).reshape(-1,3,2))
        for t in range(1,max(ANCHORS)+1):
            h=environment_step(model,h,u[:,t-1],r[:,t],e[:,t-1])
            if t in collected:collected[t].append(h.numpy().copy())
    return {t:np.concatenate(chunks) for t,chunks in collected.items()}


@torch.no_grad()
def snapshot_readout(model,h:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    gs=[];zs=[]
    for start in range(0,len(h),128):
        g=gram(torch.from_numpy(h[start:start+128]));gs.append(g.numpy());zs.append(model.readout(g).squeeze(-1).numpy())
    return np.concatenate(gs),np.concatenate(zs)


def decode_branch(name):
    if name=='NONE':return 'NONE',0
    method,when=name.split('_');return method,0 if when=='before' else int(when[-1])


def compare(raws:dict,labels:np.ndarray,u:np.ndarray,context:dict,rotations=None,leak=None)->dict:
    target=targets(labels,u);base=raws['NONE']['pre_logits'];initial=(base[:,0]>=0)==labels
    base_correct=(base>=0)==target;baseline_all=base_correct.all(1)
    rows=[];witnesses=[]
    for name in BRANCHES:
        raw=raws[name];method,when=decode_branch(name)
        correct=(raw['pre_logits']>=0)==target
        if name=='NONE':eligible=initial&baseline_all
        else:
            instantaneous=(raw['event_post_z']>=0)==target[:,when]
            eligible=initial&baseline_all&raw['event_valid']&instantaneous
            if method=='EQUAL':eligible &= np.abs(raw['event_post_z'].astype(float)-raw['event_pre_z'].astype(float))<=1e-5
        error_after=(~correct[:,when+1:]).any(1)
        raw_damage=eligible&error_after if name!='NONE' else np.zeros(len(labels),bool)
        gauge_good=np.ones(len(labels),bool)
        if method in ('EQUAL','FULL'):
            gauge=raws['GAUGE_'+name.split('_')[1]]
            gauge_good=gauge['event_valid']&(((gauge['pre_logits']>=0)==target).all(1))
        baseline_resolvable=np.min((2*target-1)*base,axis=1)>=1e-4
        post_resolvable=(2*target[:,when]-1)*raw['event_post_z']>=1e-4
        resolved_error=np.min((2*target[:,when+1:]-1)*raw['pre_logits'][:,when+1:],axis=1)<=-1e-4
        damage=raw_damage&gauge_good&baseline_resolvable&post_resolvable&resolved_error
        restore=initial&~baseline_all&correct.all(1)&raw['event_valid']&((raw['event_post_z']>=0)==target[:,when]) if name!='NONE' else np.zeros(len(labels),bool)
        flip=u==1;prev_correct=(raw['post_logits'][:,:-1]>=0)==target[:,:-1]
        flip_eligible=flip&prev_correct
        follow_mask=flip_eligible[:,:-2] if u.shape[1]>2 else np.zeros((len(labels),0),bool)
        after_state=(raw['post_logits']>=0)==target
        follow_ok=after_state[:,1:-2]&correct[:,2:-1]&correct[:,3:] if u.shape[1]>2 else follow_mask
        rows.append({'branch':name,'tests':len(labels),'initial_correct':int(initial.sum()),'baseline_all_correct':int(baseline_all.sum()),
            'intervention_valid':int(raw['event_valid'].sum()) if name!='NONE' else None,
            'eligible_for_damage':int(eligible.sum()),'classification_damage_tests':int(raw_damage.sum()),
            'gauge_confounded_damage_tests':int((raw_damage&~gauge_good).sum()),'damage_tests':int(damage.sum()),'repair_tests':int(restore.sum()),
            'both_all_correct':int((baseline_all&correct.all(1)).sum()),'both_have_error':int((~baseline_all&~correct.all(1)).sum()),
            'terminal_accuracy':float(correct[:,-1].mean()),'min_margin':float(((2*target-1)*raw['pre_logits']).min()),
            'conditional_flip_events':int(flip_eligible.sum()),'conditional_flip_correct':int((flip_eligible&correct[:,1:]).sum()),
            'conditional_flip_follow2_events':int(follow_mask.sum()),'conditional_flip_follow2_correct':int((follow_mask&follow_ok).sum())})
        if damage.any():
            ids=np.flatnonzero(damage);resolved=(2*target[ids,when+1:]-1)*raw['pre_logits'][ids,when+1:]<=-1e-4
            first=np.argmax(resolved,axis=1)+when+1
            has_flip=np.array([bool(u[i,when:t].any()) for i,t in zip(ids,first)])
            rows[-1]['damage_with_post_event_flip']=int(has_flip.sum())
            rows[-1]['next_flip_direct_damage']=int(np.sum((first==when+1)&(u[ids,when]==1)))
            for witness_kind,selection in (('any',np.ones(len(ids),bool)),('with_post_event_flip',has_flip)):
                if not selection.any():continue
                chosen_ids=ids[selection];chosen_first=first[selection]
                min_step=int(chosen_first.min());i=int(chosen_ids[np.flatnonzero(chosen_first==min_step)[0]])
                witness={**context,'branch':name,'witness_kind':witness_kind,'flat_example_index':i,'initial_label':int(labels[i]),'event_time':when,
                         'label_at_event':int(target[i,when]),'first_wrong_step':min_step,'steps_after_event':min_step-when,
                         'commands':u[i].tolist(),'failing_suffix':u[i,when:min_step].tolist(),
                         'full_state_before':raw['event_pre_h'][i].tolist(),'full_state_after':raw['event_post_h'][i].tolist(),
                         'event_pre_logit':float(raw['event_pre_z'][i]),'event_post_logit':float(raw['event_post_z'][i]),
                         'baseline_logits':base[i].tolist(),'intervened_pre_logits':raw['pre_logits'][i].tolist(),
                         'targets':target[i].tolist(),'valid':bool(raw['event_valid'][i]),'invalid_reason':int(raw['event_reason'][i]),
                         'gauge_control_pass':bool(gauge_good[i]),'baseline_min_margin':float(((2*target[i]-1)*base[i]).min())}
                if rotations is not None:witness['rotation_suffix']=rotations[i,when:min_step].tolist();witness['leak_suffix']=leak[i,when:min_step].tolist()
                witnesses.append(witness)
        else:
            rows[-1]['damage_with_post_event_flip']=0;rows[-1]['next_flip_direct_damage']=0
    return {'context':context,'branches':rows,'witnesses':witnesses}


def paired(model,h,labels,u,proto,context,r=None,e=None):
    raws={}
    for name in BRANCHES:
        method,when=decode_branch(name);raws[name]=branch(model,h,u,proto,method,when,r,e)
    summary=compare(raws,labels,u,context,r,e)
    merged={name+'__'+key:value for name,raw in raws.items() for key,value in raw.items()}
    merged.update(commands=u,labels=labels,targets=targets(labels,u),initial_h=h)
    return summary,merged


@torch.no_grad()
def periodic(model,data,proto,k,phase=None):
    n,T=data['commands'].shape;p=torch.from_numpy(proto.astype(np.float32));pieces=[];events=[]
    for start in range(0,n,128):
        b,u,r,e=tensors(data,start,start+128);h=r[:,0]@radial(model.encoder(b).reshape(-1,3,2));zs=[model.readout(gram(h)).squeeze(-1).numpy()];ev=[]
        for t in range(1,T+1):
            h=environment_step(model,h,u[:,t-1],r[:,t],e[:,t-1]);z=model.readout(gram(h)).squeeze(-1);zs.append(z.numpy())
            event=t>=8 and ((t-8)%k==0 if phase is None else t%k==phase)
            if event:
                h,a=refresh_step(model,h,z,'EQUAL8',p);ev.append(np.stack((np.full(128,t),a['valid_event'],a['invalid_reason'],a['logit_target_error']),axis=1))
        pieces.append(np.stack(zs,1));events.append(np.stack(ev,1))
    return {'logits':np.concatenate(pieces),'events':np.concatenate(events)}


def offphase(z,target):
    ts=(255,256,257,511,512,513,1023,1024)
    return {'times':list(ts),'accuracies':[float(np.mean((z[:,t]>=0)==target[:,t])) for t in ts],
            'all_time_accuracy':float(np.mean((z>=0)==target)),'ever_wrong':int(np.any((z>=0)!=target,axis=1).sum())}


def run(out:Path,base:Path):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    parent=base/'EXP-0025/v2';inputs=base/'DYN-0001/v3';previous=base/'EXP-0026/v2';old=read(previous/'completion.json')
    lookup={(r['seed'],r['arm'],r['cell'],r['condition']):r for r in old['rows'] if r['phase']=='main' and r['rotation']=='ORIGINAL'}
    sources={str(p):sha(p) for p in (Path(__file__),Path('experiments/EXP-0027/v2.md'),parent/'manifest.json',parent/'completion.json',previous/'manifest.json',previous/'completion.json',inputs/'manifest.json')}
    save(out/'manifest.json',{'experiment':'EXP-0027','version':'v2','source_sha256':source_fingerprints(),'sources':sources,
          'models':[[s,a] for s in SEEDS for a in ARMS],'anchors':ANCHORS,'branches':BRANCHES,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
          'state_inventory':{'recursive':'H:6 float32','controller':'feedforward stateless','weights':'frozen','external_context':'commands, rotations, time/schedule phase; not reset','FULL':'two fixed H states','EQUAL':'prototype class plus continuous scale; invalid fallback retains H'},'training_updates':0})
    blocks=[];micros=[];natural=[];phases=[];rawfiles={};completed=[]
    try:
        for seed in SEEDS:
            for arm in ARMS:
                folder=parent/str(seed)/arm;cp=folder/'checkpoint-1000.pt';pp=folder/'prototypes.npz';state=torch.load(cp,map_location='cpu',weights_only=True)
                parentrow=read(folder/'result.json')
                if sha(cp)!=parentrow['artifact_sha256']['checkpoint-1000.pt'] or sha(pp)!=parentrow['artifact_sha256']['prototypes.npz']:raise RuntimeError('Parent model artifact changed')
                model=LearnedMemory(arm,seed);model.load_state_dict(state['model']);model.eval();before=state_hash(model)
                if before!=parentrow['frozen_model_state_sha256']:raise RuntimeError('Frozen model state changed')
                with np.load(pp) as f:proto=f['prototypes'].copy()
                dest=out/str(seed)/arm;dest.mkdir(parents=True)
                warm,warmraw=warmup(model,proto);p=dest/'warmup.npz';save_npz(p,warmraw);rawfiles[str(p.relative_to(out))]=sha(p)
                for method in ('FULL','EQUAL'):
                    for k in (2,4):
                        summary,raw=block_test(model,proto,k,method);summary.update(seed=seed,arm=arm,warmup=warm,
                            finite_certificate=bool(method=='FULL' and summary['all_correct_and_closed'] and warm['pass']))
                        blocks.append(summary);p=dest/f'block_{method}_{k}.npz';save_npz(p,raw);rawfiles[str(p.relative_to(out))]=sha(p)
                for cell in ('LONG_H','LONG_U'):
                    ip=inputs/'data'/str(seed)/f'{cell}.npz'
                    if sha(ip)!=lookup[seed,arm,cell,'NONE']['input_sha256']:raise RuntimeError('Original input hash mismatch')
                    with np.load(ip) as f:data={k:f[k] for k in f.files}
                    snap=snapshots(model,data)
                    oldrow=lookup[seed,arm,cell,'NONE'];oldpath=previous/oldrow['raw_path']
                    if sha(oldpath)!=oldrow['raw_sha256']:raise RuntimeError('Old raw hash mismatch')
                    with np.load(oldpath) as f:
                        for anchor in ANCHORS:
                            g,z=snapshot_readout(model,snap[anchor])
                            if not np.array_equal(g,f['gram'][:,anchor]) or not np.array_equal(z,f['logits'][:,anchor]):raise RuntimeError('Snapshot replay mismatch')
                    for anchor,h in snap.items():
                        context={'seed':seed,'arm':arm,'cell':cell,'anchor':anchor,'checkpoint_sha256':sha(cp),'input_sha256':sha(ip),'prototype_sha256':sha(pp)}
                        w=words(4);u=np.tile(w,(512,1));expanded=np.repeat(h,len(w),axis=0);labels=np.repeat(data['targets'][:,anchor].astype(np.int64),len(w))
                        summary,raw=paired(model,expanded,labels,u,proto,{**context,'kind':'exhaustive4_identity'})
                        p=dest/f'micro_{cell}_{anchor}.npz';save_npz(p,raw);rawfiles[str(p.relative_to(out))]=sha(p);micros.append(summary)
                        if cell=='LONG_U':
                            u=data['commands'][:,anchor:anchor+32];r=data['common'][:,anchor+1:anchor+33];e=data['leak'][:,anchor:anchor+32]
                            summary,raw=paired(model,h,data['targets'][:,anchor].astype(np.int64),u,proto,{**context,'kind':'natural32_original'},r,e)
                            p=dest/f'natural_{anchor}.npz';save_npz(p,raw);rawfiles[str(p.relative_to(out))]=sha(p);natural.append(summary)
                    for condition in ('NONE','K2','K4','K8','K16','K32','K64','PHASE16','PHASE32'):
                        row=lookup[seed,arm,cell,condition];op=previous/row['raw_path']
                        if sha(op)!=row['raw_sha256']:raise RuntimeError('Reused phase raw changed')
                        with np.load(op) as f:metrics=offphase(f['logits'],data['targets'])
                        phases.append({'seed':seed,'arm':arm,'cell':cell,'condition':condition,'origin':'reused_EXP0026',**metrics})
                    if seed in (25002,25003):
                        specs=[(3,None),(5,None),(4,1),(4,2),(4,3),(16,4),(16,12)]
                        for k,phase in specs:
                            name=f'K{k}_ANCHOR8' if phase is None else f'K{k}_PHASE{phase}'
                            raw=periodic(model,data,proto,k,phase);p=dest/f'phase_{cell}_{name}.npz';save_npz(p,raw);rawfiles[str(p.relative_to(out))]=sha(p)
                            phases.append({'seed':seed,'arm':arm,'cell':cell,'condition':name,'origin':'new','attempts':int(np.prod(raw['events'].shape[:2])),'valid_events':int(raw['events'][:,:,1].sum()),**offphase(raw['logits'],data['targets'])})
                if state_hash(model)!=before:raise RuntimeError('Frozen weights changed')
                completed.append([seed,arm]);print(json.dumps({'model':[seed,arm],'completed':len(completed),'seconds':time.monotonic()-start}),flush=True)
        witnesses=[w for group in (micros,natural) for row in group for w in row['witnesses']]
        if (len(completed),len(blocks),len(micros),len(natural),len(phases),sum(r['origin']=='new' for r in phases))!=(8,32,64,32,200,56):
            raise RuntimeError('Fixed matrix coverage mismatch')
        equal=[w for w in witnesses if w['branch'].startswith('EQUAL')];full=[w for w in witnesses if w['branch'].startswith('FULL')]
        key=lambda w:(w['steps_after_event'],w['seed'],w['arm'],w['anchor'],w['flat_example_index'],w['branch'])
        verdict={'decision':'SUPPORTED' if equal else 'PARTIALLY_SUPPORTED' if full else 'INCONCLUSIVE','substatus':'EXPOSED_STATE_COMPUTATION_INCOMPATIBILITY' if equal else 'HARD_RESET_INCOMPATIBILITY' if full else 'NO_WITNESS_IN_FIXED_SUFFIX_TESTS'}
        result={'experiment':'EXP-0027','version':'v2',**verdict,'scope':'Exposed-state causal witnesses, not unseen-model prediction or general tradeoff','blocks':blocks,'micro':micros,'natural':natural,'phases':phases,
                'shortest_equal_witness':min(equal,key=key) if equal else None,
                'shortest_equal_flip_witness':min([w for w in equal if w['witness_kind']=='with_post_event_flip'],key=key) if any(w['witness_kind']=='with_post_event_flip' for w in equal) else None,'shortest_full_witness':min(full,key=key) if full else None,
                'models':completed,'raw_sha256':rawfiles,'run_seconds':time.monotonic()-start,'training_updates':0,'prototype_fits':0,
                'resources':{'micro_tests':5242880,'micro_transitions':20971520,'natural_tests':163840,'natural_transitions':5242880,
                             'new_phase_rollouts':56,'new_phase_transitions':29360128,'block_padded_transitions':12288,'warmup_transitions':32768,'snapshot_transitions':262144}}
        save(out/'results.json',result)
        print(json.dumps({**verdict,'seconds':result['run_seconds'],'raw_files':len(rawfiles)}))
        return result
    except BaseException as e:
        save(out/'failure.json',{'error':repr(e),'completed_models':completed,'seconds':time.monotonic()-start});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--base',type=Path,required=True);a=p.parse_args();run(a.output,a.base)
