"""One bounded checkpoint replay of exploratory learned memory; never retrains."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from .learned_protected_pilot import LearnedMemory,generate,predict,predict_with_gram,CELLS,KEYS
from .frozen_binary_probe import save_json
from .order_observation import sha
from .run import source_fingerprints


def score(z:np.ndarray,d:dict[str,np.ndarray])->dict:
    # Independent target and loss reductions; no calls to the study metrics.
    target=np.column_stack((d['b0'],(d['b0'][:,None]+np.cumsum(d['commands'],axis=1))%2))
    np.testing.assert_array_equal(target,d['targets'])
    p=z.astype(float);correct=(p>=0)==target;u=d['commands']
    loss=np.maximum(p,0)-target*p+np.log1p(np.exp(-np.abs(p)))
    return dict(bce=float(loss.sum()/loss.size),accuracy_all=float(np.count_nonzero(correct)/correct.size),
                final_accuracy=float(np.mean(correct[:,-1])),initial_accuracy=float(np.mean(correct[:,0])),
                hold_accuracy=float(np.mean(correct[:,1:][u==0])) if np.any(u==0) else None,
                flip_accuracy=float(np.mean(correct[:,1:][u==1])) if np.any(u==1) else None)


def compare_metrics(actual:dict,stored:dict)->None:
    for key,value in actual.items():
        if value is None:assert stored[key] is None
        else:assert abs(value-stored[key])<=1e-8,(key,value,stored[key])


def replay(run:Path,output:Path)->dict:
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('180-second single replay budget')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,180)
    try:
        manifest=json.loads((run/'manifest.json').read_text());analysis=json.loads((run/'analysis.json').read_text());completed=json.loads((run/'completion.json').read_text())
        assert manifest['source_sha256']==source_fingerprints()
        for path,digest in manifest['frozen_hashes'].items():assert sha(Path(path))==digest
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        for name,digest in completed['data_files'].items():assert sha(run/'data'/name)==digest
        with np.load(run/'data'/'monitor.npz') as f:monitor={k:f[k] for k in KEYS}
        regenerated=generate(128,8,[204001,20001,8,0],'MIX');assert all(np.array_equal(monitor[k],regenerated[k]) for k in KEYS)
        datasets={}
        for cell,(length,mode,amp,leak) in CELLS.items():
            regenerated=generate(1024,length,[203001,20001,length,1 if mode=='HOLD' else 2],mode,amp,leak)
            with np.load(run/'data'/f'{cell}.npz') as f:
                for k in KEYS:assert np.array_equal(regenerated[k],f[k])
            datasets[cell]=regenerated
        max_logit=max_gram=0.;rows=[]
        for arm in ('RAW','INV','REL'):
            path=run/arm;record=json.loads((path/'result.json').read_text());model=LearnedMemory(arm)
            assert record['updates']==len(record['losses'])==1000
            assert [c['step'] for c in record['curves']]==[0,125,250,500,1000]
            for curve in record['curves']:
                cp=path/f"checkpoint-{curve['step']}.pt";assert sha(cp)==curve['checkpoint_sha256']
                assert sha(path/f"monitor-{curve['step']}.npz")==curve['prediction_sha256']
                state=torch.load(cp,map_location='cpu',weights_only=True);assert state['step']==curve['step'] and state['arm']==arm
                model.load_state_dict(state['model']);z=predict(model,monitor)
                with np.load(path/f"monitor-{curve['step']}.npz") as f:err=float(np.max(np.abs(z-f['logits'])));assert err<=1e-6;max_logit=max(max_logit,err)
                compare_metrics(score(z,monitor),curve['metrics'])
            assert sha(path/'predictions.npz')==record['predictions_sha256'];metrics={}
            with np.load(path/'predictions.npz') as f:
                for cell,d in datasets.items():
                    z,g=predict_with_gram(model,d)
                    ez=float(np.max(np.abs(z-f[cell])));eg=float(np.max(np.abs(g-f[cell+'_gram'])))
                    assert ez<=1e-6 and eg<=1e-6;max_logit=max(max_logit,ez);max_gram=max(max_gram,eg)
                    metrics[cell]=score(z,d);compare_metrics(metrics[cell],record['evaluation'][cell])
                    truth=d['targets'][:,-1];means=np.stack([g[truth==b,-1].astype(float).mean(0) for b in (0,1)])
                    np.testing.assert_allclose(means,record['evaluation'][cell]['final_gram_mean_by_b'],atol=1e-10,rtol=0)
                    terms=model.readout.weight.detach().numpy()[0].astype(float)*(means[1]-means[0])
                    np.testing.assert_allclose(terms,record['evaluation'][cell]['mean_logit_separation_terms'],atol=1e-10,rtol=0)
            rows.append(dict(arm=arm,evaluation=metrics,training_data_sha256=record['training_data_sha256'],initial_hashes=record['initial_hashes']))
        assert completed['updates']==analysis['updates']==3000
        assert len({r['training_data_sha256'] for r in rows})==1
        assert len({r['initial_hashes']['encoder'] for r in rows})==len({r['initial_hashes']['readout'] for r in rows})==1
        flags={r['arm']:(r['evaluation']['IID_H']['accuracy_all']>=.95 and r['evaluation']['IID_U']['accuracy_all']>=.95 and r['evaluation']['IID_U']['flip_accuracy']>=.95) for r in rows}
        verdict,sub=('SUPPORTED','SINGLE_SEED_LEARNABILITY_ONLY') if flags['REL'] else ('INCONCLUSIVE','EXPLORATORY_LEARNING_UNRESOLVED')
        assert (verdict,sub)==(analysis['decision'],analysis['substatus']) and flags==analysis['learnability_by_arm']
        result=dict(pass_=True,decision=verdict,substatus=sub,learning_flags=flags,models=3,monitor_checkpoints=15,evaluation_conditions=24,
                    max_logit_error=max_logit,max_gram_error=max_gram,original_training_updates=3000,replay_training_updates=0,
                    seconds=time.monotonic()-started,scope='Checkpoint and data replay plus independent XOR/metrics; no independent retraining or model-selection use',source_sha256=source_fingerprints())
        save_json(output,result);return result
    except BaseException as error:
        save_json(output.with_name(output.stem+'-failure.json'),dict(error=repr(error),seconds=time.monotonic()-started));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps({k:v for k,v in replay(a.run,a.output).items() if k!='source_sha256'}))
