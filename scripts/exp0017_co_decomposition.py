"""Read-only post-hoc output decomposition; no model execution or data generation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def components(z: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    z = z.astype(np.float64)
    lp = (z - np.logaddexp.reduce(z, axis=1)[:, None]).reshape(-1, 2, 3)
    lj, lb = np.logaddexp.reduce(lp, axis=1), np.logaddexp.reduce(lp, axis=2)
    i = np.arange(len(z))
    b, j = truth // 3, truth % 3
    joint, jloss, bloss = -lp[i,b,j], -lj[i,j], -lb[i,b]
    conditional = lp[i,:,j] - lj[i,j,None]
    cp1 = np.exp(conditional[:,1])
    return dict(joint_nll=joint, j_nll=jloss, b_given_j_nll=joint-jloss,
                b_nll=bloss, j_given_b_nll=joint-bloss,
                j_accuracy=(lj.argmax(1)==j).astype(float),
                b_accuracy=(lb.argmax(1)==b).astype(float),
                conditional_b_accuracy=(conditional.argmax(1)==b).astype(float),
                joint_accuracy=(z.argmax(1)==truth).astype(float),
                conditional_b_uniform_mae=np.abs(cp1-.5),
                conditional_b_uniform_mse=(cp1-.5)**2,
                marginal_b_uniform_mae=np.abs(np.exp(lb[:,1])-.5),
                true_b_fraction=b.astype(float))


def independent(z: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Independent scalar indexing from six explicit normalized probabilities."""
    x = np.exp(z.astype(np.float64)-z.max(axis=1)[:,None])
    x /= x.sum(1)[:,None]
    values = {k:[] for k in ('joint_nll','j_nll','b_given_j_nll','b_nll','j_given_b_nll',
                            'j_accuracy','b_accuracy','conditional_b_accuracy','joint_accuracy',
                            'conditional_b_uniform_mae','conditional_b_uniform_mse',
                            'marginal_b_uniform_mae','true_b_fraction')}
    for row, target in zip(x, truth):
        b, j = divmod(int(target),3)
        pj = [float(row[t]+row[t+3]) for t in range(3)]
        pb = [float(sum(row[:3])),float(sum(row[3:]))]
        cond = [float(row[j]/pj[j]),float(row[j+3]/pj[j])]
        obs = dict(joint_nll=-math.log(float(row[target])),j_nll=-math.log(pj[j]),
                   b_given_j_nll=-math.log(cond[b]),b_nll=-math.log(pb[b]),
                   j_given_b_nll=-math.log(float(row[target])/pb[b]),
                   j_accuracy=float(pj.index(max(pj))==j),b_accuracy=float(pb.index(max(pb))==b),
                   conditional_b_accuracy=float(cond.index(max(cond))==b),
                   joint_accuracy=float(int(np.argmax(row))==target),
                   conditional_b_uniform_mae=abs(cond[1]-.5),conditional_b_uniform_mse=(cond[1]-.5)**2,
                   marginal_b_uniform_mae=abs(pb[1]-.5),true_b_fraction=float(b))
        for name,value in obs.items():values[name].append(value)
    return {name:math.fsum(value)/len(value) for name,value in values.items()}


def checks(m: dict) -> dict[str,bool]:
    return dict(j_nll=m['j_nll']<=.05,j_accuracy=m['j_accuracy']>=.95,
                binary_error_share=m['binary_error_share']>=.9,
                binary_nll_near_ln2=abs(m['b_given_j_nll']-math.log(2))<=.03,
                conditional_uniform=m['conditional_b_uniform_mae']<=.05,
                conditional_accuracy=.47<=m['conditional_b_accuracy']<=.53,
                target_balance=.47<=m['true_b_fraction']<=.53)


def main() -> None:
    output=Path('research/evidence/EXP0017-CO-DECOMP')
    output.mkdir(exist_ok=False)
    started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('60-second shared analysis budget')))
    signal.setitimer(signal.ITIMER_REAL,60)
    try:
        # Hand distributions, reviewed before touching component measurements.
        for logits, target, jloss, bgiven in [([-1000,0,-1000,-1000,0,-1000],4,0,math.log(2)),
                                             ([-1000,-1000,-1000,0,0,0],4,math.log(3),0),
                                             ([0]*6,4,math.log(3),math.log(2))]:
            got=components(np.array([logits],dtype=float),np.array([target]))
            assert abs(got['j_nll'][0]-jloss)<1e-12 and abs(got['b_given_j_nll'][0]-bgiven)<1e-12
        rows=[];input_hashes={};max_identity=max_independent=max_stored=0.
        for arm in ('D64','D16'):
            for seed in (17001,17002,17003):
                base=Path('artifacts/EXP-0017/v2')/f'seed-{seed}'/'CO'/arm
                record=json.loads((base/'result.json').read_text())
                inp=base/'predictions.npz';input_hashes[str(inp)]=sha(inp)
                input_hashes[str(base/'result.json')]=sha(base/'result.json')
                assert sha(inp)==record['predictions_sha256']
                with np.load(inp) as a:
                    for length in (32,128):
                        z=a[f'logits_{length}'];q=a[f'target_{length}'];truth=a[f'latent_{length}']
                        assert len(z)==4096 and np.array_equal(q,np.eye(6)[truth])
                        assert np.all((truth>=0)&(truth<6)) and a[f'tokens_{length}'].shape[1]==length*5//4
                        vals=components(z,truth)
                        identity=max(float(np.max(np.abs(vals['joint_nll']-vals['j_nll']-vals['b_given_j_nll']))),
                                     float(np.max(np.abs(vals['joint_nll']-vals['b_nll']-vals['j_given_b_nll']))))
                        max_identity=max(max_identity,identity);assert identity<=1e-12
                        stored=float(np.max(np.abs(vals['joint_nll']-a[f'kl_{length}'])))
                        max_stored=max(max_stored,stored);assert stored<=1e-10
                        means={name:float(v.mean()) for name,v in vals.items()}
                        other=independent(z,truth)
                        err=max(abs(means[name]-other[name]) for name in means)
                        max_independent=max(max_independent,err);assert err<=1e-12
                        means['conditional_b_uniform_rms']=math.sqrt(means['conditional_b_uniform_mse'])
                        means['binary_error_share']=means['b_given_j_nll']/means['joint_nll']
                        means['j_error_share']=means['j_nll']/means['joint_nll']
                        groups=[]
                        for state in range(6):
                            selected=truth==state
                            groups.append({'j':state%3,'b':state//3,'count':int(selected.sum()),
                                           'means':{name:float(v[selected].mean()) for name,v in vals.items()} if selected.any() else None})
                        rows.append(dict(arm=arm,seed=seed,operations=length,means=means,checks=checks(means),groups=groups))
        primary=[r for r in rows if r['arm']=='D64']
        if all(r['means']['binary_error_share']<.5 for r in primary):
            decision,substatus='FALSIFIED','LOCAL_BINARY_DOMINANCE'
        elif all(all(r['checks'].values()) for r in primary):
            decision,substatus='SUPPORTED','POSTHOC_OUTPUT_BINARY_UNRESOLVED'
        elif all(r['means']['j_error_share']<.5 for r in primary):
            decision,substatus='PARTIALLY_SUPPORTED','OUTPUT_ERROR_CONCENTRATED'
        else:decision,substatus='INCONCLUSIVE','MIXED_OUTPUT_FAILURE'
        aggregates=[]
        for arm in ('D64','D16'):
            for length in (32,128):
                select=[r for r in rows if r['arm']==arm and r['operations']==length]
                summary={}
                for name in select[0]['means']:
                    v=np.array([r['means'][name] for r in select]);mean=float(v.mean());sd=float(v.std(ddof=1));half=4.302652729911275*sd/math.sqrt(3)
                    summary[name]=dict(mean=mean,sd=sd,t95_df2=[mean-half,mean+half])
                aggregates.append(dict(arm=arm,operations=length,metrics=summary))
        assert all(sha(Path(p))==h for p,h in input_hashes.items())
        result=dict(status='ANALYZED',decision=decision,substatus=substatus,scope='Post-hoc output decomposition only; no hidden-state inference, no model or data generation',
                    rows=rows,aggregates=aggregates,verification=dict(hand_distributions_pass=True,max_chain_rule_error=max_identity,max_independent_mean_error=max_independent,max_stored_kl_error=max_stored,input_files_unchanged=True),
                    input_sha256=input_hashes,script_sha256=sha(Path(__file__)),design_sha256=sha(Path('research/design/exp0017_co_decomposition_protocol.md')),
                    seconds=time.monotonic()-started,new_training_updates=0,new_research_samples=0,numpy=np.__version__,python=sys.version)
        encoded=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        with (output/'analysis.json').open('x') as stream:stream.write(encoded)
        print(json.dumps({'decision':decision,'substatus':substatus,'seconds':result['seconds'],'verification':result['verification'],'aggregates':aggregates},ensure_ascii=False))
    except BaseException as error:
        failure={'error':repr(error),'traceback':traceback.format_exc(),'seconds':time.monotonic()-started}
        with (output/f'failure-{time.time_ns()}.json').open('x') as stream:json.dump(failure,stream,indent=2)
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
