import copy
import numpy as np
import pytest
import torch
from neurotopo.learned_protected_pilot import LearnedMemory,gram,generate,state_hash
from neurotopo.refresh_factor_core import refresh_step,refresh_rollout,shape
from neurotopo.refresh_replication_analysis import refresh_diagnostics,decide,rate,SEEDS,CONDITIONS,CELLS
from neurotopo.composition_core import predict_full
from neurotopo.state_refresh_core import fit_prototypes


def algebra_fixture():
    m=LearnedMemory('REL',259025)
    with torch.no_grad():m.readout.weight.copy_(torch.tensor([[1.,0.,2.]]));m.readout.bias.fill_(.1)
    h=torch.tensor([[[.2,.1],[.1,.3],[.2,.2]]]);p=torch.tensor([[[.1,0.],[0.,.2],[0.,0.]],[[.4,.1],[.1,.2],[0.,0.]]])
    return m,h,p

@pytest.mark.parametrize('condition',['EQUAL8','SCALE8'])
def test_feasible_algebra(condition):
    m,h,p=algebra_fixture()
    with torch.no_grad():z=m.readout(gram(h)).squeeze(-1)
    new,d=refresh_step(m,h,z,condition,p)
    assert d['valid_event'].all() and d['gram_delta_norm'][0]>1e-6
    with torch.no_grad():actual=m.readout(gram(new)).squeeze(-1);pz=m.readout(gram(p[1:])).squeeze(-1)
    torch.testing.assert_close(actual,z if condition=='EQUAL8' else pz,atol=1e-5,rtol=0)
    np.testing.assert_allclose(shape(gram(new).numpy())[0],shape(gram(p[1:] if condition=='EQUAL8' else h).numpy())[0],atol=1e-5,rtol=0)

@pytest.mark.parametrize('condition',['EQUAL8','SCALE8'])
def test_negative_ratio_falls_back(condition):
    m,h,p=algebra_fixture()
    with torch.no_grad():
        m.readout.weight.copy_(torch.tensor([[1.,0.,-1.]]));m.readout.bias.fill_(10.)
        z=m.readout(gram(h)).squeeze(-1)
    new,d=refresh_step(m,h,z,condition,p)
    assert not d['valid_event'][0] and d['invalid_reason'][0]==2
    assert torch.equal(new,h) and d['state_delta_norm'][0]==0

@pytest.mark.parametrize('condition',['EQUAL8','SCALE8'])
def test_zero_denominator_fallback(condition):
    m,h,p=algebra_fixture()
    with torch.no_grad():m.readout.weight.zero_();z=m.readout(gram(h)).squeeze(-1)
    new,d=refresh_step(m,h,z,condition,p)
    assert not d['valid_event'][0] and d['invalid_reason'][0]==1 and torch.equal(new,h)

@pytest.mark.parametrize('arm',['REL','INV'])
def test_schedule_target_isolation_and_original(arm):
    torch.set_num_threads(1)
    m=LearnedMemory(arm,259025);before=state_hash(m)
    data=generate(16,16,[259025,1],'UPDATE');cal=generate(16,8,[259025,2],'MIX');p,_=fit_prototypes(m,cal)
    z,g,_=predict_full(m,data)
    for c in CONDITIONS:
        raw=refresh_rollout(m,data,c,p['prototypes']);diag=refresh_diagnostics(raw,c)
        if c=='NONE':
            np.testing.assert_array_equal(z,raw['logits']);np.testing.assert_array_equal(g,raw['gram']);assert diag['valid_fraction'] is None
        else:assert raw['refresh_mask'].nonzero()[0].tolist()==[8,16]
        altered=dict(data);altered['targets']=1-data['targets']
        again=refresh_rollout(m,altered,c,p['prototypes'])
        for k in raw:np.testing.assert_array_equal(raw[k],again[k])
        np.testing.assert_array_equal(raw['logits'][:,:9],z[:,:9])
    assert state_hash(m)==before


def decision_rows():
    rows=[]
    for seed in (*SEEDS,23005):
        for arm in ('REL','INV'):
            r={'seed':seed,'arm':arm,'calibration':{'calibration_accuracy':1.,'prototype_labels_correct':True},'qualification':{'NONE':{}},'evaluation':{}}
            for c in CONDITIONS:
                r['evaluation'][c]={}
                for cell in CELLS:
                    acc=.75 if seed in SEEDS[:2] and c in ('NONE','GAUGE8') and cell.startswith('LONG') else 1.
                    r['evaluation'][c][cell]={'metrics':{'final_accuracy':acc,'accuracy_all':1.,'flip_accuracy':1.},'diagnostics':{'local_logit_abs_max':0.,'local_gram_abs_max':0.,'valid_fraction':1.,'valid_target_logit_error_max':0.,'valid_sign_changes':0,'valid_nonzero_gram_fraction':1.}}
            r['qualification']['NONE']=copy.deepcopy(r['evaluation']['NONE']);rows.append(r)
    return rows


def test_decision_new_seed_and_reference_exclusion():
    rows=decision_rows();d=decide(rows)
    assert d['decision']=='SUPPORTED' and d['qualified_failure_seeds']==list(SEEDS[:2])
    assert d['M_conditional_repair']['denominator']==2
    for r in rows:
        if r['seed']==23005:r['calibration']['calibration_accuracy']=0.
    assert decide(rows)==d


def test_gauge_m_eligibility_and_original_denominator():
    rows=decision_rows();rows[0]['evaluation']['GAUGE8']['IID_H']['diagnostics']['local_logit_abs_max']=1e-3
    d=decide(rows);assert d['M']=='M_INCONCLUSIVE_ELIGIBILITY' and d['R']=='R_NOT_SUPPORTED'
    assert d['seed_decisions'][0]['M_exclusion']=='GAUGE'
    assert d['methods']['EQUAL8']['repair_original_failures']['denominator']==2


def test_final_failure_disappearing_and_success_harm():
    rows=decision_rows()
    for c in ('NONE','GAUGE8'):
        for cell in ('LONG_H','LONG_U'):rows[0]['evaluation'][c][cell]['metrics']['final_accuracy']=1.
    d=decide(rows);assert d['methods']['FULL8']['repair_original_failures']['rate']==.5
    rows=decision_rows()
    for cell in ('LONG_H','LONG_U'):rows[4]['evaluation']['FULL8'][cell]['metrics']['final_accuracy']=.9
    assert decide(rows)['R']=='R_NOT_SUPPORTED'


def test_wilson_empty_and_small_sample():
    assert rate(0,0)['rate'] is None
    assert rate(2,2)['wilson95'][0]<.35
    with pytest.raises(ValueError):rate(2,1)
