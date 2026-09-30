"""Engineering roots only; no research checkpoint or outcome inspected."""
import copy
import numpy as np
import torch
from neurotopo.learned_protected_pilot import LearnedMemory,generate,tensors,gram
from neurotopo.composition_core import forward_state,auxiliary_loss,diagnostic,gram_transition,loss_terms
from neurotopo.composition_repair_analysis import decide,SEEDS,ARMS,CELLS,average_ranks,spearman


def test_forward_exact_and_branch_detach_and_counts():
    torch.set_num_threads(1);d=generate(16,8,[239023,0],'MIX')
    for arm in ('REL','INV'):
        m=LearnedMemory(arm,239023);z,g,h=forward_state(m,d)
        original=m(*tensors(d),return_gram=True)
        torch.testing.assert_close(z,original[0],atol=0,rtol=0);torch.testing.assert_close(g,original[1],atol=0,rtol=0)
    for variant in ('REL_CYCLE','REL_EXTRA'):
        m=LearnedMemory('REL',239023);z,g,h=forward_state(m,d);h.retain_grad();z.retain_grad()
        counts={'transition':0,'readout':0}
        handle=m.readout.register_forward_hook(lambda *args: counts.__setitem__('readout',counts['readout']+1))
        original=m.transition
        def transition(*args):counts['transition']+=1;return original(*args)
        m.transition=transition
        aux=auxiliary_loss(m,h,z[:,-1],torch.from_numpy(d['targets'][:,-1]),variant);aux.backward();handle.remove()
        assert h.grad is None and z.grad is None and m.encoder.weight.grad is None
        assert counts=={'transition':3,'readout':2}
        assert m.readout.weight.grad is not None and torch.isfinite(m.readout.weight.grad).all()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.controller.parameters())


def test_gram_closure_and_future_label_independence():
    torch.set_num_threads(1);rng=np.random.default_rng(239023)
    h=torch.from_numpy(rng.normal(size=(32,3,2)).astype('float32'));u=torch.arange(32)%2
    for arm in ('REL','INV'):
        m=LearnedMemory(arm,239023)
        torch.testing.assert_close(gram(m.transition(h,u)),gram_transition(m,gram(h),u),atol=2e-6,rtol=0)
    m=LearnedMemory('REL',239023);d=generate(16,8,[239023,1],'MIX');changed=dict(d);changed['targets']=1-d['targets']
    raw,s=diagnostic(m,d);other,s2=diagnostic(m,changed)
    for key in ('states','z0','z_hold','z_flip2','d_example','d_rel_example'):np.testing.assert_array_equal(raw[key],other[key])
    assert abs(raw['d_example'].mean()-s['D'])<1e-12 and abs(raw['d_rel_example'].mean()-s['D_rel'])<1e-12
    _,_,a=loss_terms(m,d,'REL_CYCLE');_,_,b=loss_terms(m,changed,'REL_CYCLE');torch.testing.assert_close(a,b)


def fixture_rows():
    rows=[]
    for i,s in enumerate(SEEDS):
        for arm in ARMS:
            loss=(.15+.12*i) if arm=='REL_BASE' else .1 if arm=='REL_CYCLE' else .4
            ev={cell:{'future_bce':loss,'final_accuracy':(1. if i<3 else .5) if arm=='REL_BASE' and cell.startswith('LONG') else 1.,'accuracy_all':1.,'flip_accuracy':1.} for cell in CELLS}
            d=(i+1.) if arm=='REL_BASE' else .1 if arm=='REL_CYCLE' else 1.
            rows.append({'seed':s,'arm':arm,'evaluation':ev,'diagnostic':{'D':d,'D_rel':d,'current_accuracy':1.}})
    return rows


def test_preregistered_prediction_repair_and_zero_floor():
    a=decide(fixture_rows());assert a['decision']=='SUPPORTED' and a['P']['rho']==1.
    rows=fixture_rows()
    for r in rows:
        if r['seed'] in SEEDS[:4] and r['arm'] in ('REL_CYCLE','REL_EXTRA'):r['diagnostic']['D']=r['diagnostic']['D_rel']=0.
    a=decide(rows);assert a['M']['status']=='M_INCONCLUSIVE_FLOOR' and a['decision']=='PARTIALLY_SUPPORTED'
    assert not any(list(a['M']['improved_by_seed'].values())[:4])
    rows=fixture_rows()
    for r in rows:
        if r['seed'] in SEEDS[:2] and r['arm']=='REL_CYCLE':r['evaluation']['IID_U']['accuracy_all']=.9
    assert decide(rows)['substatus']=='LEARNING_GUARD_FAILURE'
    np.testing.assert_array_equal(average_ranks([1,2,2,4]),[1,2.5,2.5,4]);assert spearman([1,1,1],[1,2,3]) is None
    assert abs(spearman([1,2,3],[3,2,1])+1)<1e-12


def test_registered_resource_budget_and_pairing():
    from neurotopo.composition_repair import _resources
    m=LearnedMemory('REL',239023)
    for arm in ('REL_CYCLE','REL_EXTRA'):
        r=_resources(1000,m,arm);assert r['main_operation_tokens']==344000 and r['extra_transition_tokens']==192000 and r['extra_branch_readout_heads']==128000
    assert _resources(1000,m,'REL_BASE')['extra_transition_tokens']==0
    a=generate(16,8,[239023,10],'MIX');b=generate(16,8,[239023,11],'MIX')
    assert not np.array_equal(a['common'],b['common'])
