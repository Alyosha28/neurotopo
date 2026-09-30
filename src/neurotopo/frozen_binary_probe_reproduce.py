"""One bounded independent numerical audit of frozen probe data and stationarity."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from .order_observation import ControlGRU,generate,input_tensor,sha
from .run import source_fingerprints
from .frozen_binary_probe import save_json


def replay(run:Path,output:Path)->dict:
    started=time.monotonic()
    def timed_out(*_):raise TimeoutError('120-second single audit budget')
    signal.signal(signal.SIGALRM,timed_out);signal.setitimer(signal.ITIMER_REAL,120)
    try:
        m=json.loads((run/'manifest.json').read_text());a=json.loads((run/'analysis.json').read_text())
        assert m['source_sha256']==source_fingerprints()
        for name,digest in m['frozen_hashes'].items():assert sha(Path(name))==digest
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        records=[];maximum_hidden=maximum_score=maximum_gradient=0.;paired={};total_updates=0
        for source in m['config']['sources']:
            seed,arm=source['seed'],source['arm'];path=run/f'seed-{seed}'/arm
            row=json.loads((path/'result.json').read_text());assert sha(path/'features.npz')==row['features_sha256']
            assert sha(Path(source['path']))==source['sha256']
            model=ControlGRU(arm,seed);model.load_state_dict(torch.load(source['path'],map_location='cpu',weights_only=True)['model'])
            model.requires_grad_(False);model.eval();test={}
            with np.load(path/'features.npz') as data:
                for split,n,count,ident in [('fit',32,8192,0),('test32',32,4096,1),('test128',128,4096,2)]:
                    gen=generate(count,n,'CO',[181001,seed,ident,n]);tok=data[f'{split}_tokens'];latent=data[f'{split}_latent']
                    assert np.array_equal(tok,gen['tokens']) and np.array_equal(latent,gen['latent_final'])
                    paired.setdefault((seed,split),[]).append(tok.tobytes())
                    hs=[];zs=[]
                    with torch.no_grad():
                        for i in range(0,count,128):
                            hidden,_=model.gru(input_tensor(tok[i:i+128]));hs.append(hidden[:,-1].numpy());zs.append(model.readout(hidden)[:,-1].numpy())
                    h=np.concatenate(hs);z=np.concatenate(zs)
                    eh=float(np.max(np.abs(h-data[f'{split}_hidden'])));ez=float(np.max(np.abs(z-data[f'{split}_original_logits'])))
                    maximum_hidden=max(maximum_hidden,eh,ez);assert max(eh,ez)<=1e-6
                fit_tokens={x.tobytes() for x in data['fit_tokens']}
                assert not fit_tokens.intersection(x.tobytes() for x in data['test32_tokens'])
                h=data['fit_hidden'].astype(float);mean=h.mean(0);std=h.std(0);std=np.where(std<1e-8,1.,std)
                np.testing.assert_allclose(mean,data['fit_mean'],rtol=0,atol=1e-12)
                np.testing.assert_allclose(std,data['fit_scale'],rtol=0,atol=1e-12)
                y=(data['fit_latent']//3).astype(float);assert np.array_equal(y,data['fit_true_labels'])
                shuffled=np.random.default_rng(np.random.SeedSequence([184001,seed,0 if arm=='D64' else 1])).permutation(y)
                assert np.array_equal(shuffled,data['fit_shuffled_labels'])
                x=np.column_stack(((h-mean)/std,np.ones(len(h))))
                for kind,yfit in [('true',y),('shuffle',shuffled)]:
                    w=data[f'{kind}_weights'];assert np.array_equal(w,np.array(row['fits'][kind]['weights']))
                    prob=(1+np.tanh((x@w)/2))/2
                    grad=np.mean(x*(prob-yfit)[:,None],axis=0);grad[:-1]+=.001*w[:-1]
                    g=float(np.max(np.abs(grad)));maximum_gradient=max(maximum_gradient,g)
                    assert g<=1e-7 and row['fits'][kind]['converged'];total_updates+=row['fits'][kind]['updates']
                    for n in (32,128):
                        split=f'test{n}';features=data[f'{split}_hidden'].astype(float)
                        score=((features-mean)/std)@w[:-1]+w[-1]
                        err=float(np.max(np.abs(score-data[f'{split}_{kind}_logits'])));maximum_score=max(maximum_score,err);assert err<=1e-10
                        truth=(data[f'{split}_latent']//3).astype(float)
                        acc=float(((score>=0)==truth).mean());bce=float((np.logaddexp(0,score)-truth*score).mean())
                        assert abs(acc-row['test'][str(n)][kind]['accuracy'])<=1e-10
                        assert abs(bce-row['test'][str(n)][kind]['bce'])<=1e-10
                        test.setdefault(str(n),{})[kind]=dict(accuracy=acc,bce=bce,positive_fraction=float(truth.mean()))
                for n in ('32','128'):
                    z=data[f'test{n}_original_logits'].astype(float);z=np.exp(z-z.max(1)[:,None]);z/=z.sum(1)[:,None]
                    z=z.reshape(-1,2,3);latent=data[f'test{n}_latent']
                    parent=dict(b_accuracy=float((z.sum(2).argmax(1)==latent//3).mean()),j_accuracy=float((z.sum(1).argmax(1)==latent%3).mean()))
                    assert parent==row['test'][n]['parent']
                    test[n]['parent']=parent;test[n]['gap']=test[n]['true']['accuracy']-test[n]['shuffle']['accuracy']
            assert sha(Path(source['path']))==source['sha256'] and all(p.grad is None for p in model.parameters())
            records.append(dict(seed=seed,arm=arm,test=test))
        assert len(records)==6 and all(len(x)==2 and x[0]==x[1] for x in paired.values())
        assert total_updates==a['probe_updates'] and total_updates<=600
        primary=[r for r in records if r['arm']=='D64']
        valid=all(.47<=r['test'][n]['true']['positive_fraction']<=.53 and .47<=r['test'][n]['shuffle']['accuracy']<=.53 and r['test'][n]['parent']['b_accuracy']<=.55 and r['test'][n]['parent']['j_accuracy']>=.95 for r in primary for n in ('32','128'))
        strong=lambda r,n:r['test'][n]['true']['accuracy']>=.8 and r['test'][n]['gap']>=.2
        if not valid:decision,sub='INCONCLUSIVE','CONTROL_FAILURE'
        elif all(strong(r,n) for r in primary for n in ('32','128')):decision,sub='SUPPORTED','LINEAR_INFORMATION_GENERALIZES'
        elif all(strong(r,'32') for r in primary):decision,sub='PARTIALLY_SUPPORTED','LENGTH_LIMITED_LINEAR_READOUT'
        elif any(strong(r,'32') for r in primary) or all(r['test']['32']['true']['accuracy']>=.55 and r['test']['32']['gap']>=.05 for r in primary):decision,sub='PARTIALLY_SUPPORTED','LIMITED_LINEAR_SIGNAL'
        elif all(r['test']['32']['true']['accuracy']<=.55 and r['test']['32']['gap']<=.05 for r in primary):decision,sub='INCONCLUSIVE','NO_USEFUL_LINEAR_READOUT'
        else:decision,sub='INCONCLUSIVE','MIXED_WEAK_READOUT'
        assert (decision,sub)==(a['decision'],a['substatus'])
        for summary in a['summaries']:
            rows=[r for r in records if r['arm']==summary['arm']];n=str(summary['operations'])
            for name,vals in [('true_accuracy',[r['test'][n]['true']['accuracy'] for r in rows]),('shuffle_accuracy',[r['test'][n]['shuffle']['accuracy'] for r in rows]),('gap',[r['test'][n]['gap'] for r in rows]),('true_bce',[r['test'][n]['true']['bce'] for r in rows]),('original_b_accuracy',[r['test'][n]['parent']['b_accuracy'] for r in rows])]:
                center=float(np.mean(vals));sd=float(np.std(vals,ddof=1));half=4.302652729911275*sd/np.sqrt(3)
                assert abs(center-summary[name]['mean'])<=1e-10 and abs(sd-summary[name]['sd'])<=1e-10
                np.testing.assert_allclose([center-half,center+half],summary[name]['t95_df2'],rtol=0,atol=1e-10)
        result=dict(pass_=True,decision=decision,substatus=sub,records=records,max_hidden_or_parent_logit_error=maximum_hidden,max_probe_logit_error=maximum_score,
                    max_gradient_inf=maximum_gradient,probe_updates=total_updates,original_gru_updates=0,seconds=time.monotonic()-started,
                    source_sha256=source_fingerprints(),scope='Single-pass data/hidden replay, convex stationarity, predictions and branch; no refitting or GRU training')
        save_json(output,result);return result
    except BaseException as error:
        save_json(output.with_name(output.stem+'-failure.json'),dict(error=repr(error),seconds=time.monotonic()-started))
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps({k:v for k,v in replay(a.run,a.output).items() if k not in ('records','source_sha256')}))
