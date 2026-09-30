"""Independent array reduction, direct counterexample replay, and block H traces.
The original EXP0027 files are immutable. Supplements reproduce the same fixed tests.
"""
import argparse,itertools,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from neurotopo.learned_protected_pilot import LearnedMemory,gram,radial
from neurotopo.refresh_factor_core import refresh_step
from neurotopo.run import source_fingerprints

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--base',type=Path,required=True);a=p.parse_args();out=a.run;base=a.base
start=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
load=lambda p:json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def save_npz(p,values):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:np.savez_compressed(f,**values)

def step(m,h,u,r=None,e=None):
 eye=torch.eye(3).expand(len(h),3,3);r=eye if r is None else r;e=eye if e is None else e
 rotated=r@h;return m.transition(torch.stack((rotated[:,:,0],(e@rotated[:,:,1:2]).squeeze(-1)),dim=-1),u)

@torch.no_grad()
def trace(m,h,commands,r=None,e=None):
 states=[h.numpy().copy()];zs=[m.readout(gram(h)).squeeze(-1).numpy()]
 for t in range(commands.shape[1]):
  h=step(m,h,torch.from_numpy(commands[:,t]),None if r is None else r[:,t],None if e is None else e[:,t])
  states.append(h.numpy().copy());zs.append(m.readout(gram(h)).squeeze(-1).numpy())
 return np.stack(states,1),np.stack(zs,1),h

manifest=load(out/'manifest.json');results=load(out/'results.json')
assert manifest['source_sha256']==source_fingerprints()
for path,h in manifest['sources'].items():assert digest(path)==h
assert len(results['raw_sha256'])==192 and len(results['micro'])==64 and len(results['natural'])==32 and len(results['phases'])==200
for path,h in results['raw_sha256'].items():assert digest(out/path)==h
models={};protos={}
for seed,arm in results['models']:
 folder=base/'EXP-0025/v2'/str(seed)/arm
 state=torch.load(folder/'checkpoint-1000.pt',map_location='cpu',weights_only=True);m=LearnedMemory(arm,seed);m.load_state_dict(state['model']);m.eval();models[seed,arm]=m
 with np.load(folder/'prototypes.npz') as f:protos[seed,arm]=f['prototypes'].copy()

# Fill the planned block-by-block H traces omitted from the compact initial runner.
# No command set or parameter is changed; each replay must match the saved logits.
block_files={};block_checks=[]
for seed,arm in results['models']:
 m=models[seed,arm];proto=protos[seed,arm];folder=out/str(seed)/arm
 for method,k in [('FULL',2),('FULL',4),('EQUAL',2),('EQUAL',4),('WARMUP',8)]:
  source=folder/('warmup.npz' if method=='WARMUP' else f'block_{method}_{k}.npz')
  with np.load(source) as f:old={key:f[key] for key in f.files}
  n=len(old['labels']);indices=np.arange(((n+127)//128)*128)%n;u=old['commands'][indices];labels=old['labels'][indices]
  hs=[];zs=[];post=[]
  for j in range(0,len(indices),128):
   with torch.no_grad():
    h=(torch.eye(3).expand(128,3,3)@radial(m.encoder(torch.from_numpy(labels[j:j+128])).reshape(-1,3,2))) if method=='WARMUP' else torch.from_numpy(proto[labels[j:j+128]])
    states,z,hend=trace(m,h,u[j:j+128]);index=(z[:,-1]>=0).astype(int)
    if method in ('WARMUP','FULL'):after=torch.from_numpy(proto[index])
    else:after,_=refresh_step(m,hend,torch.from_numpy(z[:,-1]),'EQUAL8',torch.from_numpy(proto))
   hs.append(states);zs.append(z);post.append(after.numpy())
  H=np.concatenate(hs)[:n];Z=np.concatenate(zs)[:n];P=np.concatenate(post)[:n]
  assert np.array_equal(Z,old['pre_logits']) and np.array_equal(P,old['event_post_h'])
  dest=out/'block_state_supplement'/str(seed)/arm/(method+'_'+str(k)+'.npz')
  save_npz(dest,{'H_before_each_score':H,'H_after_reset':P,'pre_logits':Z,'commands':old['commands'],'targets':old['targets']})
  block_files[str(dest.relative_to(out))]=digest(dest);block_checks.append([seed,arm,method,k,n,True])

# Recompute all paired endpoint and damage denominators from the raw arrays.
paired_checked=0;witness_count=0;max_replay_error=0.;by_model={};verified_witnesses=[]
for group in ('micro','natural'):
 for result in results[group]:
  ctx=result['context'];seed,arm,cell,t=ctx['seed'],ctx['arm'],ctx['cell'],ctx['anchor'];m=models[seed,arm]
  path=out/str(seed)/arm/(f'micro_{cell}_{t}.npz' if group=='micro' else f'natural_{t}.npz')
  with np.load(path) as f:raw={key:f[key] for key in f.files}
  u=raw['commands'];y=raw['labels'];target=np.bitwise_xor.accumulate(np.column_stack((y,u)),axis=1)
  assert np.array_equal(target,raw['targets'])
  z0=raw['NONE__pre_logits'];initial=(z0[:,0]>=0)==y;baseok=((z0>=0)==target).all(1)
  for row in result['branches']:
   name=row['branch'];z=raw[name+'__pre_logits'];post=raw[name+'__post_logits'];when=0 if name=='NONE' or name.endswith('before') else int(name[-1]);method=name.split('_')[0]
   correct=(z>=0)==target
   assert np.array_equal(z[:,:when+1],z0[:,:when+1])
   keep=np.arange(z.shape[1])!=when if name!='NONE' else np.ones(z.shape[1],bool)
   assert np.array_equal(z[:,keep],post[:,keep])
   valid=raw[name+'__event_valid'];pre=raw[name+'__event_pre_z'];after=raw[name+'__event_post_z']
   if name=='NONE':eligible=initial&baseok;classification=np.zeros(len(y),bool);goodg=np.ones(len(y),bool)
   else:
    eligible=initial&baseok&valid&((after>=0)==target[:,when])
    if method=='EQUAL':eligible&=np.abs(after.astype(float)-pre.astype(float))<=1e-5
    classification=eligible&np.any(~correct[:,when+1:],1)
    if method in ('EQUAL','FULL'):
     gauge='GAUGE_'+name.split('_')[1];goodg=raw[gauge+'__event_valid']&(((raw[gauge+'__pre_logits']>=0)==target).all(1))
    else:goodg=np.ones(len(y),bool)
   resolvable=(np.min((2*target-1)*z0,axis=1)>=1e-4)&((2*target[:,when]-1)*after>=1e-4)&(np.min((2*target[:,when+1:]-1)*z[:,when+1:],axis=1)<=-1e-4)
   damage=classification&goodg&resolvable
   assert int(eligible.sum())==row['eligible_for_damage'] and int(damage.sum())==row['damage_tests']
   assert int(classification.sum())==row['classification_damage_tests']
   assert int((classification&~goodg).sum())==row['gauge_confounded_damage_tests']
   assert float(correct[:,-1].mean())==row['terminal_accuracy']
   if name!='NONE':
    assert np.array_equal(valid,raw[name+'__event_reason']==0)
    if method=='EQUAL':
     assert np.all(np.abs(after[valid].astype(float)-pre[valid].astype(float))<=1e-5)
     assert np.array_equal(raw[name+'__event_pre_h'][~valid],raw[name+'__event_post_h'][~valid])
   update=(u==1)&((post[:,:-1]>=0)==target[:,:-1]);assert int(update.sum())==row['conditional_flip_events']
   assert int((update&correct[:,1:]).sum())==row['conditional_flip_correct']
   follow=update[:,:-2];ok=((post[:,1:-2]>=0)==target[:,1:-2])&correct[:,2:-1]&correct[:,3:]
   assert int(follow.sum())==row['conditional_flip_follow2_events'] and int((follow&ok).sum())==row['conditional_flip_follow2_correct']
  for w in result['witnesses']:
   name=w['branch'];idx=w['flat_example_index'];when=w['event_time'];stop=w['first_wrong_step'];commands=np.tile(w['failing_suffix'],(128,1))
   preH=np.array(w['full_state_before'],np.float32);postH=np.array(w['full_state_after'],np.float32)
   assert np.array_equal(preH,raw[name+'__event_pre_h'][idx]) and np.array_equal(postH,raw[name+'__event_post_h'][idx])
   r=torch.from_numpy(np.repeat(np.array(w['rotation_suffix'],np.float32)[None],128,axis=0)) if 'rotation_suffix' in w else None
   e=torch.from_numpy(np.repeat(np.array(w['leak_suffix'],np.float32)[None],128,axis=0)) if 'leak_suffix' in w else None
   _,before_z,_=trace(m,torch.from_numpy(np.repeat(preH[None],128,axis=0)),commands,r,e)
   _,after_z,_=trace(m,torch.from_numpy(np.repeat(postH[None],128,axis=0)),commands,r,e)
   expected0=z0[idx,when:stop+1];expected1=raw[name+'__pre_logits'][idx,when:stop+1].copy();expected1[0]=w['event_post_logit']
   err=max(float(np.max(np.abs(before_z[0].astype(float)-expected0))),float(np.max(np.abs(after_z[0].astype(float)-expected1))))
   assert err<=2e-6
   label=w['label_at_event'];expected=np.bitwise_xor.accumulate(np.r_[label,w['failing_suffix']]);assert np.all((before_z[0]>=0)==expected) and (after_z[0,-1]>=0)!=expected[-1]
   if name.startswith('EQUAL'):
    proto=protos[seed,arm][int(w['event_pre_logit']>=0)]
    with torch.no_grad():pz=m.readout(gram(torch.from_numpy(np.repeat(proto[None],128,axis=0)))).squeeze(-1)[0].item()
    bias=float(m.readout.bias.detach()[0]);ratio=(w['event_pre_logit']-bias)/(pz-bias)
    calculated=(proto.astype(float)*np.sqrt(ratio)).astype(np.float32);assert np.max(np.abs(calculated-postH))<=1e-7
   max_replay_error=max(max_replay_error,err);witness_count+=1
   verified_witnesses.append({'seed':seed,'arm':arm,'group':group,'anchor':t,'branch':name,'steps_after_event':w['steps_after_event'],'suffix':w['failing_suffix'],'replay_max_error':err})
  paired_checked+=1

# Independently score new and reused phase logs with the fixed nonaligned times.
old=load(base/'EXP-0026/v2/completion.json');lookup={(r['seed'],r['arm'],r['cell'],r['condition']):r for r in old['rows'] if r['phase']=='main' and r['rotation']=='ORIGINAL'}
for row in results['phases']:
 seed,arm,cell,condition=(row[k] for k in ('seed','arm','cell','condition'))
 if row['origin']=='new':p=out/str(seed)/arm/f'phase_{cell}_{condition}.npz'
 else:p=base/'EXP-0026/v2'/lookup[seed,arm,cell,condition]['raw_path']
 with np.load(p) as f,np.load(base/'DYN-0001/v3/data'/str(seed)/f'{cell}.npz') as d:
  z=f['logits'];target=d['targets'];scores=[float(((z[:,t]>=0)==target[:,t]).mean()) for t in row['times']]
  assert scores==row['accuracies'] and float(((z>=0)==target).mean())==row['all_time_accuracy']
  if row['origin']=='new':
   k=int(condition.split('_')[0][1:]);phase=None if 'ANCHOR' in condition else int(condition.split('PHASE')[1])
   ts=[t for t in range(8,1025) if ((t-8)%k==0 if phase is None else t%k==phase)]
   assert np.array_equal(f['events'][:,:,0],np.repeat(np.array(ts)[None],512,axis=0))
   good=f['events'][:,:,1].astype(bool);assert np.all(f['events'][:,:,3][good]<=1e-5)

verification={'pass':True,'original_raw_files_verified':192,'paired_contexts_verified':paired_checked,'phase_records_verified':200,
              'block_and_warmup_replays':len(block_checks),'block_trace_supplement_files':block_files,
              'witnesses_replayed':witness_count,'maximum_witness_replay_error':max_replay_error,'witness_records':verified_witnesses,
              'scope':'Independent array reduction and direct model-transition replay of all recorded witnesses; model transition shared, no training or full reexecution of every micro branch.',
              'supplement_reason':'Initial compact runner retained event H but omitted per-step block H. Same frozen blocks/warmups replayed and logits/post-reset H matched exactly; per-step H added in separate immutable files.',
              'verification_seconds':time.monotonic()-start,'source_script_sha256':digest(Path(__file__)),'results_sha256':digest(out/'results.json')}
save(out/'verification.json',verification)
print(json.dumps({k:verification[k] for k in ('pass','paired_contexts_verified','block_and_warmup_replays','witnesses_replayed','maximum_witness_replay_error','verification_seconds')}))
