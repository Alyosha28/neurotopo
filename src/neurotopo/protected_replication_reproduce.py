"""Bounded frozen-checkpoint verification, with independently reduced metrics."""
from __future__ import annotations
import argparse
import json
import signal
import time
from pathlib import Path
import numpy as np
import torch
from .learned_protected_pilot import LearnedMemory, generate, predict, predict_with_gram, CELLS, KEYS
from .learned_protected_pilot_reproduce import score, compare_metrics
from .protected_interventions import intervention_data, intervention_rollout
from .order_observation import sha
from .run import source_fingerprints, write_json


def replay(run: Path, output: Path) -> dict:
    start=time.monotonic()
    def abort(*_): raise TimeoutError('Registered 180-second replay cap')
    signal.signal(signal.SIGALRM,abort);signal.setitimer(signal.ITIMER_REAL,180)
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        manifest=json.loads((run/'manifest.json').read_text())
        assert manifest['source_sha256']==source_fingerprints()
        for name,digest in {**manifest['frozen_hashes'],**manifest['pilot_artifact_sha256']}.items():assert sha(Path(name))==digest
        completion=json.loads((run/'completion.json').read_text())
        for name,digest in completion['data_files'].items():assert sha(run/name)==digest
        error=0.;models=cells=conditions=checkpoints=0;contrasts=[];replications=[];qualified=[]
        for seed in [20002,20003,20004,20005,20001]:
            base=run/str(seed)
            datasets={}
            for cell,(length,mode,amp,leak) in CELLS.items():
                datasets[cell]=generate(1024,length,[203001,seed,length,1 if mode=='HOLD' else 2],mode,amp,leak)
            datasets['monitor']=generate(128,8,[204001,seed,8,0],'MIX')
            datasets['intervention']=intervention_data(seed)
            for name,data in datasets.items():
                with np.load(base/'data'/f'{name}.npz') as saved:
                    for key in KEYS:np.testing.assert_array_equal(saved[key],data[key])
            seed_scores={};learning={};rots={};none={}
            for arm in ('RAW','INV','REL'):
                path=base/arm;record=json.loads((path/'result.json').read_text())
                for name,digest in record['artifact_sha256'].items():assert sha(path/name)==digest
                model=LearnedMemory(arm,seed)
                for curve in record['curves']:
                    cp=path/f"checkpoint-{curve['step']}.pt"
                    assert sha(cp)==curve['checkpoint_sha256']
                    state=torch.load(cp,weights_only=True,map_location='cpu');model.load_state_dict(state['model'])
                    z=predict(model,datasets['monitor'])
                    with np.load(path/f"monitor-{curve['step']}.npz") as f:np.testing.assert_array_equal(z,f['logits'])
                    compare_metrics(score(z,datasets['monitor']),curve['metrics']);checkpoints+=1
                state=torch.load(path/'checkpoint-1000.pt',weights_only=True,map_location='cpu');model.load_state_dict(state['model']);models+=1
                ev={}
                with np.load(path/'predictions.npz') as saved:
                    for cell in CELLS:
                        z,g=predict_with_gram(model,datasets[cell])
                        e=max(float(np.abs(z-saved[cell]).max()),float(np.abs(g-saved[cell+'_gram']).max()))
                        assert e<=1e-6;error=max(error,e)
                        ev[cell]=score(z,datasets[cell]);compare_metrics(ev[cell],record['evaluation'][cell]);cells+=1
                        truth=datasets[cell]['targets'][:,-1]
                        means=np.stack([g[truth==bit,-1].astype(float).mean(0) for bit in (0,1)])
                        terms=model.readout.weight.detach().numpy()[0].astype(float)*(means[1]-means[0])
                        np.testing.assert_allclose(terms,record['evaluation'][cell]['mean_logit_separation_terms'],atol=1e-10,rtol=0)
                effects=intervention_rollout(model,datasets['intervention'],seed)
                z,g=predict_with_gram(model,datasets['intervention'])
                np.testing.assert_array_equal(effects['NONE']['logits'],z[:,32:]);np.testing.assert_array_equal(effects['NONE']['gram'],g[:,32:])
                target=datasets['intervention']['targets'][:,32:]
                independ={}
                for name,arrays in effects.items():
                    with np.load(path/f'intervention-{name}.npz') as saved:
                        for key,value in arrays.items():np.testing.assert_array_equal(value,saved[key])
                    h,v=arrays['before'],arrays['after']
                    if name in ('NONE','NORM_SHAM','ANGLE_SHAM'):
                        np.testing.assert_allclose(v.swapaxes(1,2)@v,h.swapaxes(1,2)@h,atol=3e-6,rtol=0)
                    if name=='ANGLE':np.testing.assert_allclose(np.linalg.norm(v,axis=1),np.linalg.norm(h,axis=1),atol=3e-6,rtol=0)
                    if name=='NORM':
                        np.testing.assert_allclose(v,h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-8),atol=3e-6,rtol=0)
                    p=arrays['logits'].astype(float);correct=(p>=0)==target
                    loss=np.maximum(p,0)-target*p+np.log1p(np.exp(-np.abs(p)))
                    for label,index in [('immediate',0),('next',1),('final',-1),('suffix',slice(1,None))]:
                        for suffix,value in [('accuracy',float(correct[:,index].mean())),('bce',float(loss[:,index].mean()))]:
                            assert abs(value-record['intervention'][name][label+'_'+suffix])<1e-8
                    independ[name]=float(correct[:,0].mean());conditions+=1
                seed_scores[arm]={'N':independ['NORM_SHAM']-independ['NORM'],'A':independ['ANGLE_SHAM']-independ['ANGLE']}
                learning[arm]=ev['IID_H']['accuracy_all']>=.95 and ev['IID_U']['accuracy_all']>=.95 and ev['IID_U']['flip_accuracy']>=.95
                rots[arm]=ev['ROT_U']['final_accuracy'];none[arm]=independ['NONE']
            cn=seed_scores['INV']['N']-seed_scores['REL']['N'];ca=seed_scores['REL']['A']-seed_scores['INV']['A']
            if seed!=20001:
                contrasts.append(cn>=.10 and ca>=.10 and min(none['INV'],none['REL'])>=.95)
                replications.append(all(learning[a] and rots[a]>=.95 and rots[a]-rots['RAW']>=.20 for a in ('INV','REL')))
                qualified.append(all(learning[a] and none[a]>=.95 for a in ('INV','REL')))
        R,M=all(replications),all(contrasts)
        decision='SUPPORTED' if R and M else 'PARTIALLY_SUPPORTED' if R or M else 'FALSIFIED' if all(qualified) else 'INCONCLUSIVE'
        analysis=json.loads((run/'analysis.json').read_text());assert decision==analysis['decision']
        assert completion['updates']==12000
        result={'pass':True,'decision':decision,'replication':R,'mechanism':M,'models':models,'evaluation_cells':cells,'intervention_conditions':conditions,'monitor_checkpoints':checkpoints,'max_error':error,'replay_updates':0,'seconds':time.monotonic()-start,'scope':'All checkpoints/cells replayed; independent XOR, loss, contrast and verdict reductions. Shared forward/intervention implementation plus independent invariants; no retraining.'}
        write_json(output,result);return result
    except BaseException as exc:
        write_json(output.with_name(output.stem+'-failure.json'),{'error':repr(exc),'seconds':time.monotonic()-start});raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(replay(a.run,a.output)))
