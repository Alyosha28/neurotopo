"""One frozen, bounded replay with independent target and loss reductions."""
from __future__ import annotations
import argparse
import json
import signal
import time
from pathlib import Path
import numpy as np
import torch
from . import equal_output_geometry as geom
from .learned_protected_pilot import LearnedMemory, generate, state_hash
from .order_observation import sha
from .run import source_fingerprints,write_json


def independent_metrics(z: np.ndarray, target: np.ndarray) -> dict:
    p=z.astype(float);truth=target[:,None]
    correct=(p>=0)==truth;loss=np.maximum(p,0)-truth*p+np.log1p(np.exp(-np.abs(p)))
    return {'future_bce':loss[:,:,:,1:].mean(axis=(0,2,3)),
            'future_accuracy':correct[:,:,:,1:].mean(axis=(0,2,3)),
            'final_accuracy':correct[:,:,:,-1].mean(axis=(0,2)),
            'current_accuracy':correct[:,:,:,0].mean(axis=(0,2)),
            'current_bce':loss[:,:,:,0].mean(axis=(0,2)),
            'accuracy_by_time':correct.mean(axis=(0,2)),
            'bce_by_time':loss.mean(axis=(0,2))}


def replay(run: Path, output: Path) -> dict:
    started=time.monotonic()
    def abort(*_):raise TimeoutError('300-second single replay cap')
    signal.signal(signal.SIGALRM,abort);signal.setitimer(signal.ITIMER_REAL,300)
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        manifest=json.loads((run/'manifest.json').read_text());completed=json.loads((run/'completion.json').read_text())
        assert manifest['source_sha256']==source_fingerprints()
        for key in ('frozen_hashes','parent_artifact_sha256'):
            for p,digest in manifest[key].items():assert sha(Path(p))==digest
        for p,digest in completed['data_files'].items():assert sha(run/p)==digest
        analysis=json.loads((run/'analysis.json').read_text());curves=json.loads((run/'curves.json').read_text());assert sha(run/'curves.json')==analysis['curves_sha256']
        max_error=0.;summaries=[];count=0
        for seed in [20002,20003,20004,20005,20001]:
            prefix=generate(64,32,[222001,seed,32,2],'UPDATE',np.pi,0.)
            np.testing.assert_array_equal(prefix['targets'][:,1:],(prefix['b0'][:,None]+np.cumsum(prefix['commands'],axis=1))%2)
            search=geom.suffix_data(seed,prefix['targets'][:,-1],4,222002)
            test=geom.suffix_data(seed,prefix['targets'][:,-1],16,222003)
            for name,data in [('prefix',prefix),('search',search),('test',test)]:
                with np.load(run/str(seed)/'data'/f'{name}.npz') as f:
                    for k,v in data.items():np.testing.assert_array_equal(v,f[k])
                if name!='prefix':
                    truth=(prefix['targets'][:,-1,None,None]+np.cumsum(data['commands'],axis=2))%2
                    np.testing.assert_array_equal(truth,data['targets'][:,:,1:])
            for arm in ('RAW','INV','REL'):
                m=LearnedMemory(arm,seed);cp=torch.load(run/str(seed)/arm/'checkpoint.pt',weights_only=True,map_location='cpu');m.load_state_dict(cp['model']);m.eval()
                before=state_hash(m);h=geom.prefix_states(m,prefix)
                for radius in (.01,.05,.15):
                    row=next(r for r in completed['records'] if r['seed']==seed and r['arm']==arm and r['radius']==radius)
                    path=run/row['path']
                    for name,digest in row['artifact_sha256'].items():assert sha(path/name)==digest
                    bank=geom.candidate_bank(m,h,seed,radius)
                    search_states=np.where(bank['valid'][:,:,None,None],bank['states'],h[:,None])
                    search_logits,_=geom.rollout(m,search_states,search)
                    z=search_logits[:,:,:,1:].astype(float);target=search['targets'][:,None,:,1:]
                    score=(np.maximum(z,0)-target*z+np.log1p(np.exp(-np.abs(z)))).mean(axis=(2,3))
                    selected,_=geom.select_candidates(search_logits,search['targets'],bank['valid'])
                    eligible=bank['valid'].any(axis=1)
                    best=np.max(np.where(bank['valid'],score,-np.inf),axis=1)
                    assert np.all(best[eligible]-score[np.arange(64),selected][eligible]<=1e-10)
                    with np.load(path/'raw.npz') as raw:
                        np.testing.assert_array_equal(raw['prefix_states'],h)
                        for key,value in bank.items():
                            stored_key={'states':'candidate_states','sham':'sham_state'}.get(key,key)
                            np.testing.assert_array_equal(raw[stored_key],value)
                        np.testing.assert_array_equal(raw['selection_indices'],selected)
                        np.testing.assert_allclose(raw['selection_scores'],score,atol=1e-10,rtol=0)
                        ni=np.arange(64);sv=bank['valid'][ni,selected];rv=bank['valid'][:,0];hv=bank['sham_valid']
                        states=np.stack([h,np.where(rv[:,None,None],bank['states'][:,0],h),np.where(sv[:,None,None],bank['states'][ni,selected],h),np.where(hv[:,None,None],bank['sham'],h)],axis=1)
                        np.testing.assert_array_equal(raw['condition_states'],states)
                        logits,grams=geom.rollout(m,states,test)
                        err=max(float(np.abs(logits-raw['test_logits']).max()),float(np.abs(grams-raw['test_gram']).max()));assert err<=1e-6;max_error=max(max_error,err)
                        # Direct quadratic/current-logit and Euclidean constraint checks.
                        g=h.swapaxes(1,2)@h;sh=bank['sham'];np.testing.assert_allclose(sh.swapaxes(1,2)@sh,g,atol=3e-6,rtol=0)
                        valid_prefix=bank['valid'].all(axis=1)&bank['sham_valid']
                        current_error=np.abs(logits[:,:,:,0]-logits[:,0:1,:,0]).max()
                        if valid_prefix.all():assert current_error<=1e-5
                        metrics=independent_metrics(logits,test['targets'])
                        stored=next(c for c in analysis['cells'] if c['seed']==seed and c['arm']==arm and c['radius']==radius)
                        for k,name in enumerate(('NONE','RANDOM','SELECTED','SHAM')):
                            for key,field in [('future_bce','future_bce'),('future_accuracy','future_accuracy'),('final_accuracy','last_accuracy'),('current_accuracy','current_accuracy')]:
                                assert abs(metrics[key][k]-stored[name][field])<1e-10,(seed,arm,radius,name,key)
                            curve=curves[f'{seed}/{arm}/r{round(radius*1000):03d}'][name]
                            np.testing.assert_allclose(metrics['accuracy_by_time'][k],curve['accuracy_by_step_0_to_32'],atol=1e-12,rtol=0)
                            np.testing.assert_allclose(metrics['bce_by_time'][k],curve['bce_by_step_0_to_32'],atol=1e-10,rtol=0)
                            wrong=(logits[:,k,:,1:]>=0)!=test['targets'][:,:,1:]
                            exists=wrong.any(axis=-1);first=np.argmax(wrong,axis=-1)+1
                            hist=[int(np.sum(exists&(first==t))) for t in range(1,33)]
                            assert hist==stored[name]['first_error_counts_1_to_32']
                            assert int(np.sum(~exists))==stored[name]['right_censored_no_error_count']
                        summaries.append({'seed':seed,'arm':arm,'radius':radius,'valid_fraction':float(valid_prefix.mean()),'max_current_logit_error':float(current_error),**{k:v.tolist() for k,v in metrics.items()}})
                        count+=1
                assert state_hash(m)==before
        # Independent fixed INV primary reductions, not a call to analyzer classifier.
        primary=[r for r in summaries if r['arm']=='INV' and r['seed']!=20001 and r['radius']==.15]
        qualified=all(r['valid_fraction']>=.99 and r['current_accuracy'][0]>=.95 and r['future_accuracy'][0]>=.95 for r in primary)
        effects=[r['future_bce'][2]-r['future_bce'][0] for r in primary]
        drops=[r['final_accuracy'][0]-r['final_accuracy'][2] for r in primary]
        strong=sum(x>=.05 and y>=.05 for x,y in zip(effects,drops))
        smaller=sum(x>=.02 for x in effects)>=3 and sum(y>=.02 for y in drops)>=2
        decision='INCONCLUSIVE' if not qualified else 'SUPPORTED' if strong>=3 else 'PARTIALLY_SUPPORTED' if smaller else 'INCONCLUSIVE'
        assert analysis['decision']==decision
        directional=[r['future_bce'][2]-r['future_bce'][1] for r in primary]
        secondary=float(np.mean(directional))>=.02 and sum(x>0 for x in directional)>=3
        assert secondary==analysis['S_directional_search_pass']
        result={'pass':True,'decision':decision,'qualified':qualified,'primary_effects':effects,'primary_final_drops':drops,'models':15,'cells':count,'max_error':max_error,'research_updates':0,'replay_updates':0,'seconds':time.monotonic()-started,'scope':'Shared frozen geometry/forward replay plus independent XOR/loss/selection and verdict reductions; no retraining or broader search.'}
        write_json(output,result);write_json(output.with_name(output.stem+'_metrics.json'),{'cells':summaries});return result
    except BaseException as exc:
        write_json(output.with_name(output.stem+'_failure.json'),{'error':repr(exc),'seconds':time.monotonic()-started});raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(replay(a.run,a.output)))
