"""Frozen paired CO histories: exact full-state equality, not distance-as-information."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import signal
import sys
import time
from typing import Any
import numpy as np
import torch
from .order_observation import ControlGRU, input_tensor, sha
from .frozen_binary_probe import save_json, standard_state_hash
from .run import git_output, source_fingerprints

SEEDS=(17001,17002,17003)
TIMES=(0,1,2,4,8,16,32,64,128)
SAMPLE_INDICES=np.array([12+k+(12+k)//4-1 for k in TIMES])
PREFIX_INDEX=14


def prefix_pools() -> list[np.ndarray]:
    words=((np.arange(4096)[:,None] >> np.arange(11,-1,-1))&1).astype(np.int8)
    r=(words==0).sum(1);q=3*((12-r)%2)+r%3
    return [words[q==i] for i in range(6)]


def visible(operations:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    """Independent integer CO state update; includes post-operation sensors."""
    n,length=operations.shape
    assert length%4==0
    tokens=np.empty((n,length*5//4),dtype=np.int8);states=np.empty_like(tokens)
    j=np.zeros(n,dtype=np.int8);b=np.zeros(n,dtype=np.int8);t=0
    for op_index in range(length):
        op=operations[:,op_index];j=(j+(op==0))%3;b=b^(op==1)
        tokens[:,t]=op;states[:,t]=3*b+j;t+=1
        if (op_index+1)%4==0:
            tokens[:,t]=3+(j==0);states[:,t]=3*b+j;t+=1
    return tokens,states


def paired_data(seed:int,namespace:int=191001,n:int=384)->dict[str,np.ndarray]:
    pools=prefix_pools();a,b=np.random.SeedSequence([namespace,seed]).spawn(2)
    rng=np.random.default_rng(a);suffix=np.random.default_rng(b).integers(0,2,(n,128),dtype=np.int8)
    pre=np.empty((3,n,12),dtype=np.int8)
    for i in range(n):
        j=i%3;zero,one=pools[j],pools[j+3]
        ia=int(rng.integers(len(zero)));ib=int(rng.integers(len(one)))
        ic=int(rng.integers(len(zero)-1));ic+=ic>=ia
        pre[0,i]=zero[ia];pre[1,i]=one[ib];pre[2,i]=zero[ic]
    ops=np.concatenate((pre,np.broadcast_to(suffix,(3,n,128))),axis=2)
    toks=[];states=[]
    for branch in range(3):
        tok,state=visible(ops[branch]);toks.append(tok);states.append(state)
    data=dict(operations=ops,tokens=np.stack(toks),states=np.stack(states))
    check_pairing(data)
    return data


def check_pairing(data:dict[str,np.ndarray])->None:
    tok,q,ops=data['tokens'],data['states'],data['operations']
    assert np.array_equal(tok[0,:,15:],tok[1,:,15:]) and np.array_equal(tok[0,:,15:],tok[2,:,15:])
    assert np.array_equal(q[0,:,14:]%3,q[1,:,14:]%3) and np.array_equal(q[0,:,14:]%3,q[2,:,14:]%3)
    assert np.all(q[0,:,14:]//3 != q[1,:,14:]//3) and np.array_equal(q[0,:,14:],q[2,:,14:])
    assert np.all(np.any(ops[0,:,:12]!=ops[2,:,:12],axis=1))


def bit_equal(a:np.ndarray,b:np.ndarray)->np.ndarray:
    assert a.dtype==b.dtype and a.dtype in (np.dtype('float32'),np.dtype('float64'))
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError('Nonfinite state cannot certify equality')
    u=np.uint32 if a.dtype==np.float32 else np.uint64
    return np.all(np.ascontiguousarray(a).view(u)==np.ascontiguousarray(b).view(u),axis=-1)


def pair_curves(a:np.ndarray,b:np.ndarray)->dict[str,np.ndarray]:
    eq=bit_equal(a,b)
    assert not np.any(eq[:,:-1]&~eq[:,1:]),'Exact merger revived under identical future'
    delta=a.astype(np.float64)-b.astype(np.float64)
    distance=np.sqrt(np.sum(delta*delta,axis=-1))
    if not np.isfinite(distance).all():raise FloatingPointError('Nonfinite distance')
    first=np.where(eq.any(1),eq.argmax(1),-1)
    return dict(equal=eq,distance=distance,first_relative_token=first)


def curve_summary(curves:dict[str,np.ndarray])->dict:
    eq,d=curves['equal'],curves['distance'];initial=~eq[:,0];count=int(initial.sum())
    new=initial&eq[:,-1]
    return dict(initial_distinct_count=count,initial_distinct_fraction=float(initial.mean()),
                terminal_equal_count=int(eq[:,-1].sum()),terminal_equal_fraction=float(eq[:,-1].mean()),
                new_merger_count=int(new.sum()),terminal_merge_fraction_among_initial_distinct=float(new.sum()/count) if count else None,
                equality_fraction_by_token=eq.mean(0).tolist(),median_distance_by_token=np.median(d,axis=0).tolist(),
                first_merge_relative_token=[int(x) if x>=0 else None for x in curves['first_relative_token']])


def stats(values:list[float])->dict:
    x=np.asarray(values);mean=float(x.mean());sd=float(x.std(ddof=1));h=4.302652729911275*sd/np.sqrt(3)
    return dict(values=values,mean=mean,sd=sd,t95_df2=[float(mean-h),float(mean+h)])


def decision(records:list[dict])->tuple[str,str]:
    expected={(s,a,d) for s in SEEDS for a in ('D64','D16') for d in ('float32','float64')}
    if len(records)!=12 or {(r['seed'],r['arm'],r['precision']) for r in records}!=expected:
        return 'INCONCLUSIVE','ENGINEERING_ABORT'
    rows=[r['opposite_b'] for r in records if r['arm']=='D64' and r['precision']=='float32']
    if all(r['initial_distinct_fraction']>=.95 and r['terminal_merge_fraction_among_initial_distinct'] is not None and r['terminal_merge_fraction_among_initial_distinct']>=.9 for r in rows):
        return 'SUPPORTED','FINITE_PRECISION_PAIR_COLLAPSE'
    if any(r['new_merger_count']>0 for r in rows):
        return 'PARTIALLY_SUPPORTED','PARTIAL_PAIR_COLLAPSE'
    return 'INCONCLUSIVE','NO_NEW_EXACT_MERGER_OBSERVED'


def infer_branch(model:ControlGRU,tokens:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    hidden=[];logits=[];dtype=next(model.parameters()).dtype
    assert len(tokens)%128==0
    with torch.no_grad():
        for i in range(0,len(tokens),128):
            states,_=model.gru(input_tensor(tokens[i:i+128]).to(dtype))
            z=model.readout(states)
            hidden.append(states.numpy());logits.append(z.numpy())
    return np.concatenate(hidden),np.concatenate(logits)


def evaluate_condition(source:dict,precision:str,data:dict)->tuple[dict,dict]:
    arm,seed=source['arm'],source['seed']
    assert sha(Path(source['path']))==source['sha256'] and sha(Path(source['probe_path']))==source['probe_sha256']
    model=ControlGRU(arm,seed)
    model.load_state_dict(torch.load(source['path'],map_location='cpu',weights_only=True)['model'])
    model.eval();model.requires_grad_(False);model.to(dtype=torch.float32 if precision=='float32' else torch.float64)
    before=standard_state_hash(model);hs=[];zs=[]
    for branch in range(3):
        h,z=infer_branch(model,data['tokens'][branch]);hs.append(h);zs.append(z)
        if not np.isfinite(h).all() or not np.isfinite(z).all():raise FloatingPointError('Nonfinite forward')
    repeat_h,repeat_z=infer_branch(model,data['tokens'][0,:128])
    assert bit_equal(hs[0][:128],repeat_h).all() and bit_equal(zs[0][:128],repeat_z).all()
    assert standard_state_hash(model)==before and all(p.grad is None for p in model.parameters())
    assert sha(Path(source['path']))==source['sha256']
    opposite=pair_curves(hs[0][:,14:],hs[1][:,14:]);same=pair_curves(hs[0][:,14:],hs[2][:,14:])
    raw={}
    for name,curves in [('opposite',opposite),('same',same)]:
        raw.update({name+'_'+k:v for k,v in curves.items()})
    snapshots=np.stack([h[:,SAMPLE_INDICES] for h in hs]);zshots=np.stack([z[:,SAMPLE_INDICES] for z in zs])
    raw.update(hidden=snapshots,logits=zshots,sample_token_indices=SAMPLE_INDICES)
    lp=zshots.astype(float)-np.logaddexp.reduce(zshots.astype(float),axis=-1)[...,None]
    probabilities=np.exp(lp).reshape(3,len(hs[0]),len(TIMES),2,3)
    original_p1=probabilities[:,:,:,1,:].sum(-1)
    original_j=probabilities.sum(-2).argmax(-1)
    with np.load(source['probe_path']) as p:
        mean,scale,w=p['fit_mean'],p['fit_scale'],p['true_weights']
    probe_z=((snapshots.astype(float)-mean)/scale)@w[:-1]+w[-1]
    probe_p1=np.exp(-np.logaddexp(0.,-probe_z))
    if not all(np.isfinite(x).all() for x in (probe_z,probe_p1,original_p1)):
        raise FloatingPointError('Nonfinite readout')
    raw.update(original_b_probability=original_p1,original_j_prediction=original_j,probe_b_probability=probe_p1)
    truth=data['states'][:,:,SAMPLE_INDICES]//3
    readouts=[]
    for idx,k in enumerate(TIMES):
        record=dict(suffix_operations=k)
        for name,prob in [('original',original_p1),('fixed_probe',probe_p1)]:
            preds=prob[:2,:,idx]>=.5;target=truth[:2,:,idx];correct=preds==target
            record[name]=dict(accuracy=float(correct.mean()),both_correct_fraction=float(np.all(correct,axis=0).mean()),
                              signed_probability_contrast=float(((prob[1,:,idx]-prob[0,:,idx])*(target[1].astype(float)-target[0].astype(float))).mean()))
        record['original_j_accuracy']=float((original_j[:2,:,idx]==data['states'][:2,:,SAMPLE_INDICES[idx]]%3).mean())
        readouts.append(record)
    row=dict(seed=seed,arm=arm,precision=precision,opposite_b=curve_summary(opposite),same_b=curve_summary(same),
             readouts=readouts,state_hash_before_after=before,parent_checkpoint_sha256=source['sha256'],
             duplicate_forward_bitwise_equal=True,model_updates=0,probe_updates=0,
             resources=dict(hidden_values=model.hidden,parameter_count=sum(p.numel() for p in model.parameters()),
                            scientific_sequences=1152,duplicate_control_sequences=128,tokens_per_sequence=175,
                            precision_bytes=4 if precision=='float32' else 8,executed_flops='NOT_MEASURED',peak_memory='NOT_MEASURED'))
    return row,raw


def validate_config(c:dict)->None:
    assert c['experiment_id']=='EXP-0019' and c['version']=='v2' and c['workflow']=='co_paired_dynamics' and c['gate1_claim'] is False
    assert c['task']==dict(prefix_operations=12,suffix_operations=128,triples=384,namespace=191001,sample_suffix_operations=list(TIMES))
    assert c['precisions']==['float32','float64'] and c['thresholds']==dict(initial_distinct_min=.95,terminal_merge_min=.90)
    assert c['resource']==dict(worker_wall_seconds_cap=180,reproduction_wall_seconds_cap=180,research_update_cap=0)
    assert len(c['sources'])==6 and {(s['seed'],s['arm']) for s in c['sources']}=={(s,a) for s in SEEDS for a in ('D64','D16')}
    assert np.__version__=='2.3.3' and torch.__version__.split('+')[0]=='2.8.0'


def run(config:Path,output_override:Path|None=None)->Path:
    started=time.monotonic();c=json.loads(config.read_text());validate_config(c)
    assert not git_output('status','--porcelain'),'Clean frozen checkout required'
    suite=Path('artifacts/validation/suite.json');v=json.loads(suite.read_text());assert v['pass'] and v['source_sha256']==source_fingerprints()
    output=output_override or Path(c['output']);output.mkdir(parents=True,exist_ok=False)
    manifest=dict(experiment_id='EXP-0019',version='v2',config=c,git_commit=git_output('rev-parse','HEAD'),git_status='',source_sha256=source_fingerprints(),
                  frozen_hashes={str(p):sha(p) for p in (config,Path(c['protocol']),Path(c['entry_contract']),Path('uv.lock'),suite)},
                  python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),gru_updates=0,probe_updates=0,gate1_claim=False)
    save_json(output/'manifest.json',manifest);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    records=[];data_cache={}
    try:
        for source in c['sources']:
            seed=source['seed']
            if seed not in data_cache:
                data_cache[seed]=paired_data(seed)
                with (output/f'data-{seed}.npz').open('xb') as f:np.savez_compressed(f,**data_cache[seed])
            for precision in c['precisions']:
                if time.monotonic()-started>=180:raise TimeoutError('Shared run budget')
                now=time.monotonic();row,raw=evaluate_condition(source,precision,data_cache[seed])
                path=output/f'seed-{seed}'/source['arm']/precision;path.mkdir(parents=True)
                with (path/'trajectories.npz').open('xb') as f:np.savez_compressed(f,**raw)
                row.update(seconds=time.monotonic()-now,trajectories_sha256=sha(path/'trajectories.npz'),data_sha256=sha(output/f'data-{seed}.npz'))
                save_json(path/'result.json',row);records.append(row)
                print(json.dumps({k:row[k] for k in ['seed','arm','precision','seconds']} | {'opposite_terminal_equal':row['opposite_b']['terminal_equal_fraction'],'same_terminal_equal':row['same_b']['terminal_equal_fraction']}),flush=True)
        save_json(output/'completion.json',dict(status='COMPLETED',seconds=time.monotonic()-started,records=records,gru_updates=0,probe_updates=0))
    except BaseException as error:
        save_json(output/'run_failure.json',dict(error=repr(error),seconds=time.monotonic()-started,completed_records=records));raise
    return output


def analyze(output:Path)->dict:
    if (output/'worker_failure.json').exists():
        failure=json.loads((output/'worker_failure.json').read_text());p=output/'run_failure.json';detail=json.loads(p.read_text()) if p.exists() else {}
        sub='INCOMPLETE_BUDGET' if detail.get('error','').startswith('TimeoutError(') else failure['substatus']
        a=dict(experiment_id='EXP-0019',decision='INCONCLUSIVE',substatus=sub,failure=failure,detail=detail,gate1_claim=False)
    else:
        completion=json.loads((output/'completion.json').read_text());rows=completion['records']
        for r in rows:
            assert sha(output/f"seed-{r['seed']}"/r['arm']/r['precision']/'trajectories.npz')==r['trajectories_sha256']
            assert sha(output/f"data-{r['seed']}.npz")==r['data_sha256']
        verdict,sub=decision(rows);summaries=[]
        for arm in ('D64','D16'):
            for precision in ('float32','float64'):
                rs=[r for r in rows if r['arm']==arm and r['precision']==precision]
                for pair in ('opposite_b','same_b'):
                    summaries.append(dict(arm=arm,precision=precision,pair=pair,
                                          initial_distinct=stats([r[pair]['initial_distinct_fraction'] for r in rs]),
                                          terminal_equal=stats([r[pair]['terminal_equal_fraction'] for r in rs]),
                                          newly_merged=stats([r[pair]['new_merger_count']/384 for r in rs])))
        a=dict(experiment_id='EXP-0019',decision=verdict,substatus=sub,records=rows,summaries=summaries,
               gru_updates=0,probe_updates=0,gate1_claim=False,scope='Exact full-state mergers for registered conditional prefix pairs; distances and fixed readouts are descriptive.')
    save_json(output/'analysis.json',a);print(json.dumps({k:v for k,v in a.items() if k not in ('records','summaries')}),flush=True);return a


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    def stop(*_:Any)->None:raise TimeoutError('Worker shared deadline')
    signal.signal(signal.SIGTERM,stop);run(a.config,a.output)
