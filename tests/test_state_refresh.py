import numpy as np
import torch
import pytest
from neurotopo.learned_protected_pilot import LearnedMemory,generate,predict_with_gram
from neurotopo.state_refresh_core import canonical_state,fit_prototypes,refresh_rollout,transition_table


def test_psd_root_fit_and_negative_eigenvalues():
    h=np.random.default_rng(249024).normal(size=(16,3,2));g=h.swapaxes(1,2)@h
    root,e=canonical_state(g);np.testing.assert_allclose(root.swapaxes(1,2)@root,g,atol=2e-6,rtol=1e-6)
    with pytest.raises(ValueError):canonical_state(np.diag([-1e-3,1.])[None])
    r,e=canonical_state(np.diag([-1e-10,1.])[None]);assert e.min()<0 and np.isfinite(r).all()
    m=LearnedMemory('REL',249024);d=generate(16,8,[249024,0],'MIX');raw,s=fit_prototypes(m,d)
    matrices=raw['calibration_states'].astype(float).swapaxes(1,2)@raw['calibration_states'].astype(float)
    for b in (0,1):np.testing.assert_allclose(raw['prototype_grams'][b],matrices[raw['calibration_labels']==b].mean(0),atol=0,rtol=0)
    assert raw['prototypes'].shape==(2,3,2)


def test_none_scoring_times_gauge_and_no_target_input():
    torch.set_num_threads(1);m=LearnedMemory('REL',249024);cal=generate(16,8,[249024,1],'MIX');p,_=fit_prototypes(m,cal)
    data=generate(16,16,[249024,2],'UPDATE',np.pi,0.);z,g=predict_with_gram(m,data)
    none=refresh_rollout(m,data,'NONE',p['prototypes']);np.testing.assert_array_equal(none['logits'],z);np.testing.assert_array_equal(none['gram'],g)
    for condition in ('GAUGE8','RESET8','RESET1'):
        raw=refresh_rollout(m,data,condition,p['prototypes']);changed=dict(data);changed['targets']=1-data['targets']
        other=refresh_rollout(m,changed,condition,p['prototypes'])
        for key in raw:np.testing.assert_array_equal(raw[key],other[key])
        assert not raw['refresh_mask'][0]
        if condition!='RESET1':np.testing.assert_array_equal(raw['logits'][:,:9],none['logits'][:,:9])
        if condition=='GAUGE8':
            assert np.max(np.abs(raw['post_gram']-raw['gram']))<1e-5
            assert np.max(np.abs(raw['post_logits']-raw['logits']))<1e-5
        else:
            mask=raw['refresh_mask'];np.testing.assert_array_equal(raw['selected_labels'][:,mask],(raw['logits'][:,mask]>=0).astype(int))
            assert (raw['selected_labels'][:,~mask]==-1).all()
    table=transition_table(m,p['prototypes']);np.testing.assert_array_equal(table['expected'],[[0,1],[1,0]])


def test_case_decision_guards_and_partial_branch():
    from neurotopo.state_refresh_analysis import decide
    rows=[]
    for seed in range(23001,23007):
        for arm in ('REL_BASE','INV_BASE'):
            evaluation={}
            for condition in ('NONE','GAUGE8','RESET8','RESET1'):
                evaluation[condition]={}
                for cell in ('IID_H','IID_U','LONG_H','LONG_U'):
                    acc=(.75 if condition in ('NONE','GAUGE8') else .98) if cell.startswith('LONG') else 1.
                    evaluation[condition][cell]={'metrics':{'final_accuracy':acc,'accuracy_all':1.,'flip_accuracy':1.},'diagnostics':{'local_logit_abs_max':0.,'local_gram_abs_max':0.}}
            rows.append({'seed':seed,'arm':arm,'calibration':{'calibration_accuracy':1.,'prototype_labels_correct':True},'evaluation':evaluation})
    assert decide(rows)['decision']=='SUPPORTED'
    case=next(r for r in rows if r['seed']==23005 and r['arm']=='REL_BASE')
    case['evaluation']['GAUGE8']['LONG_H']['metrics']['final_accuracy']=.7
    assert decide(rows)['substatus']=='CASE_QUALIFICATION_FAILURE'
    case['evaluation']['GAUGE8']['LONG_H']['metrics']['final_accuracy']=.75
    for cell in ('LONG_H','LONG_U'):case['evaluation']['RESET8'][cell]['metrics']['final_accuracy']=.82
    assert decide(rows)['decision']=='PARTIALLY_SUPPORTED'
