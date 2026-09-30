"""Engineering-only exact-readout geometry tests; no research checkpoint/data."""
import numpy as np
import torch
from neurotopo.learned_protected_pilot import LearnedMemory,generate,predict_with_gram,state_hash
from neurotopo.equal_output_geometry import candidate_bank,prefix_states,suffix_data,rollout,select_candidates,_perpendicular


def test_equal_logit_radius_and_sham_constraints():
    torch.set_num_threads(1)
    h=np.random.default_rng(229022).normal(size=(64,3,2)).astype(np.float32)
    for arm in ('RAW','INV','REL'):
        model=LearnedMemory(arm,229022)
        before=state_hash(model)
        for r in (.01,.05,.15):
            b=candidate_bank(model,h,229022,r)
            assert b['valid'].all() and b['sham_valid'].all()
            original=h.swapaxes(1,2)@h;v=b['sham']
            np.testing.assert_allclose(v.swapaxes(1,2)@v,original,rtol=1e-6,atol=2e-6)
            assert np.max(np.abs(b['states'][:,0]-h))>0
        assert state_hash(model)==before


def test_zero_rankone_and_direction_fallback():
    torch.set_num_threads(1);m=LearnedMemory('INV',229022)
    z=candidate_bank(m,np.zeros((4,3,2),np.float32),229022,.15)
    assert not z['valid'].any() and not z['sham_valid'].any()
    assert np.isfinite(z['states']).all()
    h=np.ones((8,3,2),np.float32);h[:,:,1]*=-1
    b=candidate_bank(m,h,229022,.15);assert b['valid'].all() and b['sham_valid'].all()
    v=np.array([[1.,0,0]]);d,bad=_perpendicular(v,np.array([[[2.,0,0]]]))
    assert bad.all();np.testing.assert_allclose((d*v[:,None]).sum(-1),0);np.testing.assert_allclose(np.linalg.norm(d,axis=-1),1)


def test_prefix_and_suffix_none_replay_and_rng_isolation():
    torch.set_num_threads(1)
    data=generate(16,64,[229022,0], 'UPDATE',np.pi,0.)
    prefix={k:(v[:,:32] if k in ('commands','leak') else v[:,:33] if k in ('common','targets') else v) for k,v in data.items()}
    suffix={'commands':data['commands'][:,None,32:],'common':data['common'][:,None,33:], 'targets':data['targets'][:,None,32:]}
    for arm in ('RAW','INV','REL'):
        m=LearnedMemory(arm,229022);h=prefix_states(m,prefix);z,g=rollout(m,h[:,None],suffix);full,fg=predict_with_gram(m,data)
        np.testing.assert_allclose(z[:,0,0],full[:,32:],atol=2e-6,rtol=0)
        np.testing.assert_allclose(g[:,0,0],fg[:,32:],atol=2e-6,rtol=0)
    a=suffix_data(229022,data['targets'][:,32],4,229023);b=suffix_data(229022,data['targets'][:,32],16,229024)
    assert not np.array_equal(a['commands'],b['commands'][:,:4])
    assert (a['commands'][:,:,0]==0).any() and (a['commands'][:,:,0]==1).any()
    np.testing.assert_array_equal(a['targets'][:,:,1:],(a['targets'][:,:,:1]+np.cumsum(a['commands'],axis=2))%2)


def test_selection_only_search_scores_ties_and_invalid_candidates():
    target=np.zeros((2,4,33),np.float32);z=np.zeros((2,16,4,33),np.float32)
    z[:,3,:,1:]=2;valid=np.ones((2,16),bool);valid[1,3]=False
    idx,score=select_candidates(z,target,valid)
    np.testing.assert_array_equal(idx,[3,0]);assert score[0,3]>score[0,0]
    changed=z.copy();changed[:,:,:,0]=1000
    idx2,score2=select_candidates(changed,target,valid)
    np.testing.assert_array_equal(idx,idx2);np.testing.assert_array_equal(score,score2)


def test_registered_primary_decisions_and_first_error_censoring():
    from neurotopo.equal_output_stress_analysis import _decision,_condition
    rows=[]
    for seed in (20002,20003,20004,20005,20001):
        for arm in ('RAW','INV','REL'):
            for radius in (.01,.05,.15):
                rows.append({'seed':seed,'arm':arm,'radius':radius,'all_candidate_and_sham_valid_fraction':1.,
                             'NONE':{'current_accuracy':1.,'future_accuracy':1.},
                             'selected_minus_none_future_bce':.06,'test_last_accuracy_decline':.06,
                             'selected_minus_random_future_bce':.03,'sham_equivariance_alarm':False})
    assert _decision(rows)['decision']=='SUPPORTED'
    for r in rows:
        if r['seed'] in (20002,20003) and r['arm']=='INV':r['selected_minus_none_future_bce']=.03;r['test_last_accuracy_decline']=.03
    assert _decision(rows)['decision']=='PARTIALLY_SUPPORTED'
    for r in rows:
        if r['seed']==20004 and r['arm']=='INV':r['NONE']['future_accuracy']=.94
    assert _decision(rows)['substatus']=='QUALIFICATION_FAILURE'
    target=np.ones((2,2,33),np.float32);z=np.ones_like(target);z[0,0,32]=-1
    score,_=_condition(z,target)
    assert score['first_error_counts_1_to_32'][-1]==1
    assert score['right_censored_no_error_count']==3
