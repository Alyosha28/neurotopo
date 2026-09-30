"""One supervised frozen-trajectory replay with independent raw-array reductions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from .co_paired_dynamics import paired_data,evaluate_condition
from .order_observation import sha
from .frozen_binary_probe import save_json
from .run import source_fingerprints


def replay(run:Path,output:Path)->dict:
    start=time.monotonic()
    def abort(*_):raise TimeoutError('180-second one-pass audit limit')
    signal.signal(signal.SIGALRM,abort);signal.setitimer(signal.ITIMER_REAL,180)
    try:
        m=json.loads((run/'manifest.json').read_text());a=json.loads((run/'analysis.json').read_text())
        assert m['source_sha256']==source_fingerprints()
        for path,digest in m['frozen_hashes'].items():assert sha(Path(path))==digest
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        data={};rows=[];max32=max64=0.
        for seed in (17001,17002,17003):
            generated=paired_data(seed)
            with np.load(run/f'data-{seed}.npz') as saved:
                assert all(np.array_equal(generated[k],saved[k]) for k in generated)
            # Alternative integer derivation of every state from prefix operation counts.
            ops=generated['operations'];rcount=np.cumsum(ops==0,axis=2);scount=np.cumsum(ops==1,axis=2)
            q=3*(scount%2)+rcount%3
            for t in range(140):
                idx=t+t//4
                np.testing.assert_array_equal(generated['states'][:,:,idx],q[:,:,t])
                if (t+1)%4==0:
                    np.testing.assert_array_equal(generated['states'][:,:,idx+1],q[:,:,t])
                    np.testing.assert_array_equal(generated['tokens'][:,:,idx+1],3+(q[:,:,t]%3==0))
            data[seed]=generated
        for source in m['config']['sources']:
            seed,arm=source['seed'],source['arm']
            for precision in ('float32','float64'):
                path=run/f'seed-{seed}'/arm/precision;stored=json.loads((path/'result.json').read_text())
                assert sha(path/'trajectories.npz')==stored['trajectories_sha256']
                assert sha(run/f'data-{seed}.npz')==stored['data_sha256']
                replay_row,replayed=evaluate_condition(source,precision,data[seed])
                tol=1e-6 if precision=='float32' else 1e-12
                with np.load(path/'trajectories.npz') as arrays:
                    for name,new in replayed.items():
                        old=arrays[name]
                        if new.dtype.kind in 'biu':assert np.array_equal(new,old)
                        else:
                            error=float(np.max(np.abs(new-old)));assert error<=tol
                            if precision=='float32':max32=max(max32,error)
                            else:max64=max(max64,error)
                    # Independently derive the fractions and first-merger records from raw flags.
                    for field,prefix in [('opposite_b','opposite'),('same_b','same')]:
                        flags=arrays[prefix+'_equal'];dist=arrays[prefix+'_distance']
                        assert flags.shape==(384,161) and np.isfinite(dist).all()
                        assert not np.any(flags[:,:-1]&~flags[:,1:])
                        initial_count=sum(not bool(x) for x in flags[:,0])
                        new_count=sum((not bool(x[0])) and bool(x[-1]) for x in flags)
                        first=[]
                        for pair in flags:
                            first.append(next((i for i,v in enumerate(pair) if v),None))
                        expected=dict(initial_distinct_count=initial_count,initial_distinct_fraction=initial_count/384,
                                      terminal_equal_count=int(flags[:,-1].sum()),terminal_equal_fraction=float(flags[:,-1].mean()),
                                      new_merger_count=new_count,terminal_merge_fraction_among_initial_distinct=new_count/initial_count if initial_count else None,
                                      first_merge_relative_token=first)
                        for k,v in expected.items():assert stored[field][k]==v
                        np.testing.assert_allclose(stored[field]['median_distance_by_token'],np.median(dist,axis=0),rtol=0,atol=1e-12)
                        np.testing.assert_array_equal(stored[field]['equality_fraction_by_token'],flags.mean(0))
                    for sr,rr in zip(stored['readouts'],replay_row['readouts']):
                        assert sr['suffix_operations']==rr['suffix_operations']
                        assert sr['original_j_accuracy']==rr['original_j_accuracy']
                        for name in ('original','fixed_probe'):
                            for metric in sr[name]:assert abs(sr[name][metric]-rr[name][metric])<=1e-12
                    # Independent direct-probability scoring of stored fixed readout arrays.
                    truths=data[seed]['states'][:,:,arrays['sample_token_indices']]//3
                    for i,record in enumerate(stored['readouts']):
                        logits=arrays['logits'][:2,:,i].astype(float)
                        scores=np.exp(logits-logits.max(-1,keepdims=True)).reshape(2,384,2,3)
                        jhat=scores.sum(-2).argmax(-1)
                        actual_j=data[seed]['states'][:2,:,arrays['sample_token_indices'][i]]%3
                        assert float((jhat==actual_j).mean())==record['original_j_accuracy']
                        np.testing.assert_array_equal(jhat,arrays['original_j_prediction'][:2,:,i])
                        for name,key in [('original','original_b_probability'),('fixed_probe','probe_b_probability')]:
                            p=arrays[key][:2,:,i];truth=truths[:2,:,i];correct=(p>=.5)==truth
                            assert float(correct.mean())==record[name]['accuracy']
                            assert float(np.all(correct,axis=0).mean())==record[name]['both_correct_fraction']
                            contrast=float(((p[1]-p[0])*(truth[1].astype(float)-truth[0].astype(float))).mean())
                            assert abs(contrast-record[name]['signed_probability_contrast'])<=1e-12
                rows.append(stored)
        assert len(rows)==12
        primary=[r['opposite_b'] for r in rows if r['arm']=='D64' and r['precision']=='float32']
        if all(r['initial_distinct_fraction']>=.95 and r['terminal_merge_fraction_among_initial_distinct'] is not None and r['terminal_merge_fraction_among_initial_distinct']>=.9 for r in primary):
            verdict,sub='SUPPORTED','FINITE_PRECISION_PAIR_COLLAPSE'
        elif any(r['new_merger_count']>0 for r in primary):verdict,sub='PARTIALLY_SUPPORTED','PARTIAL_PAIR_COLLAPSE'
        else:verdict,sub='INCONCLUSIVE','NO_NEW_EXACT_MERGER_OBSERVED'
        assert (verdict,sub)==(a['decision'],a['substatus'])
        for summary in a['summaries']:
            rs=[r[summary['pair']] for r in rows if r['arm']==summary['arm'] and r['precision']==summary['precision']]
            for name,values in [('initial_distinct',[r['initial_distinct_fraction'] for r in rs]),('terminal_equal',[r['terminal_equal_fraction'] for r in rs]),('newly_merged',[r['new_merger_count']/384 for r in rs])]:
                mean=float(np.mean(values));sd=float(np.std(values,ddof=1));half=4.302652729911275*sd/np.sqrt(3)
                assert abs(mean-summary[name]['mean'])<=1e-12 and abs(sd-summary[name]['sd'])<=1e-12
                np.testing.assert_allclose(summary[name]['t95_df2'],[mean-half,mean+half],atol=1e-12,rtol=0)
        result=dict(pass_=True,decision=verdict,substatus=sub,conditions_replayed=12,max_float32_error=max32,max_float64_error=max64,
                    exact_flags_match=True,physics_verified=True,gru_updates=0,probe_updates=0,seconds=time.monotonic()-start,
                    primary_counts=[{k:r[k] for k in ['initial_distinct_count','new_merger_count','terminal_merge_fraction_among_initial_distinct']} for r in primary],
                    source_sha256=source_fingerprints(),scope='Frozen numerical trajectory replay plus independent physical/count/statistic reductions; no training')
        save_json(output,result);return result
    except BaseException as error:
        save_json(output.with_name(output.stem+'-failure.json'),dict(error=repr(error),seconds=time.monotonic()-start));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    print(json.dumps({k:v for k,v in replay(args.run,args.output).items() if k!='source_sha256'}))
