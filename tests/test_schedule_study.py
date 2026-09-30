import itertools
import numpy as np
import torch
from neurotopo.schedule_core import count_schedule,attempt_mask,lower_tau,rollout,MAIN,EXTRA,threshold
from neurotopo.schedule_analysis import window,summarize
from neurotopo.learned_protected_pilot import LearnedMemory,generate,state_hash
from neurotopo.state_refresh_core import fit_prototypes
from neurotopo.refresh_factor_core import refresh_rollout


def test_all_window_patterns_are_exhaustive():
    for bits in itertools.product((False,True),repeat=6):
        label,k=window(list(bits));ids=np.flatnonzero(bits)
        if not len(ids):assert label=='NO_WINDOW'
        elif len(ids)==6:assert label=='RIGHT_CENSORED_64'
        elif not np.array_equal(ids,np.arange(ids[0],ids[-1]+1)):assert label=='NONMONOTONIC_WINDOW'
        elif ids[0]==0:assert label=='PREFIX_WINDOW'
        elif ids[-1]==5:assert label=='SUFFIX_WINDOW_RIGHT_CENSORED'
        else:assert label=='BOUNDED_WINDOW'
    assert window([False]*5+[True])[0]=='SUFFIX_WINDOW_RIGHT_CENSORED'


def test_count_pairing_and_adapt_cooldown():
    counts=np.array([0,1,2,509]);mask=count_schedule(counts,1024)
    np.testing.assert_array_equal(mask.sum(1),counts);assert not mask[:,:8].any()
    assert mask[1,8] and mask[2,516]
    last=np.array([8,6]);d=np.ones(2)*2
    np.testing.assert_array_equal(attempt_mask('ADAPT90',9,d,1.,last,None),[False,True])
    # Attempt cooldown is independent of validity: caller updates last on every attempt.
    np.testing.assert_array_equal(attempt_mask('ADAPT90',10,d,1.,last,None),[True,True])


def test_tau_censor_lower_median_boundary():
    d=np.zeros((512,1025));d[:256,8]=2
    tau,c=lower_tau(d,1.);assert np.sort(tau)[255]==0 and c.sum()==256
    d[255,8]=0;tau,c=lower_tau(d,1.);assert np.sort(tau)[255]==1017 and c.sum()==257
    d[:]=0;d[:,1024]=2;tau,c=lower_tau(d,1.);assert np.all(tau==1016) and not c.any()


def test_core_schedule_replay_count_and_target_isolation():
    torch.set_num_threads(1);m=LearnedMemory('REL',269026);before=state_hash(m)
    cal=generate(128,8,[269026,1],'MIX');data=generate(128,32,[269026,2],'UPDATE',np.pi,0.)
    p,_=fit_prototypes(m,cal);theta=threshold(p)
    adapt=None
    for c in MAIN+EXTRA:
        raw=rollout(m,data,p['prototypes'],c,'ORIGINAL',theta,adapt if c=='COUNT90' else None,c=='NONE')
        summarize(raw,data,c,theta)
        if c=='ADAPT90':adapt=raw['attempt_counts']
        if c in ('NONE','K8','FULL8','SCALE8','GAUGE8'):
            old=refresh_rollout(m,data,'EQUAL8' if c=='K8' else c,p['prototypes'])
            np.testing.assert_array_equal(raw['logits'],old['logits'])
        if c=='COUNT90':np.testing.assert_array_equal(raw['attempt_counts'],adapt)
    altered=dict(data);altered['targets']=1-data['targets']
    a=rollout(m,data,p['prototypes'],'K4','IDENTITY',theta)
    b=rollout(m,altered,p['prototypes'],'K4','IDENTITY',theta)
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    assert state_hash(m)==before


def decision_fixture():
    rows=[]
    for seed in (*range(25001,25013),23005):
        for arm in ('REL','INV'):
            for cell in ('LONG_H','LONG_U'):
                for rotation in ('ORIGINAL','IDENTITY'):
                    for c in MAIN+EXTRA if rotation=='ORIGINAL' else MAIN:
                        acc=.5 if arm=='REL' and seed in (25002,25003) and cell=='LONG_H' and c not in ('K2','K4','K8') else 1.
                        m={f'final_{t}':acc for t in (256,512,1024)}
                        m.update(attempts=100,valid_fraction=1.,valid_logit_error_max=0.,gauge_logit_max=0.,gauge_gram_max=0.)
                        rows.append({'seed':seed,'arm':arm,'cell':cell,'rotation':rotation,'condition':c,'phase':'main','summary':{'metrics':m}})
    return rows


def test_decision_control_failure_and_invalid_k_denominators():
    from neurotopo.schedule_analysis import decisions
    rows=decision_fixture();d=decisions(rows)
    assert d['window_seeds']==[25002,25003] and d['substatus']=='MULTIPLE_EXPOSED_SCHEDULE_WINDOWS'
    for row in rows:
        if row['seed']==25002 and row['condition']=='GAUGE8':row['summary']['metrics']['gauge_gram_max']=.1
    d=decisions(rows);assert d['window_seeds']==[25003] and d['control_inconclusive_seeds']==[25002]
    for row in rows:
        if row['seed']==25003 and row['condition'] in ('K2','K4','K8'):row['summary']['metrics']['valid_fraction']=.98
    d=decisions(rows);assert d['substatus']=='CONTROL_INCONCLUSIVE' and d['decision']=='INCONCLUSIVE'
