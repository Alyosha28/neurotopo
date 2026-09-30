import numpy as np
import torch
from neurotopo.equal_logit_future import _first_step_tangent,_extend
from neurotopo.learned_protected_pilot import LearnedMemory, generate, gram
from neurotopo.state_refresh_core import canonical_state


def test_gram_null_tangent_predicts_small_one_step_change():
    torch.set_num_threads(1)
    model=LearnedMemory('REL',259025)
    with torch.no_grad():
        model.readout.weight.copy_(torch.tensor([[1.,0.,2.]]))
        model.readout.bias.fill_(.1)
    original=np.array([[.4,.08],[.08,.3]])
    change=1e-4*np.array([[2.,0.],[0.,-1.]])
    h=torch.from_numpy(canonical_state(original)[0]).repeat(512,1,1)
    hp=torch.from_numpy(canonical_state(original+change)[0]).repeat(512,1,1)
    data=generate(512,256,[259025,91,1],'HOLD',np.pi,0.)
    audit=_first_step_tangent(model,h,hp,np.ones(512,bool),data)
    assert audit['available'].all()
    assert np.max(np.abs(audit['current_tangent']))<1e-7
    assert np.max(np.abs(audit['next_tangent']-audit['double_exact_next']))<2e-7
    with torch.no_grad():
        z=model.readout(gram(h)).squeeze(-1)
        zp=model.readout(gram(hp)).squeeze(-1)
    assert torch.max(torch.abs(z-zp)).item()<1e-6


def test_tangent_unavailable_is_nan_not_zero():
    model=LearnedMemory('REL',259025)
    h=torch.zeros(512,3,2)
    data=generate(512,256,[259025,91,2],'HOLD',np.pi,0.)
    audit=_first_step_tangent(model,h,h,np.ones(512,bool),data)
    assert not audit['available'].any()
    assert np.isnan(audit['next_tangent']).all()


def test_longer_suffix_keeps_frozen_prefix_and_oracle():
    for cell,mode in (('LONG_H','HOLD'),('LONG_U','UPDATE')):
        original=generate(512,256,[259025,91,3],mode,np.pi,0.)
        result=_extend(original,259025,cell)
        assert result['commands'].shape==(512,1024)
        for key in ('b0','commands','common','leak','targets'):
            prefix=result[key] if key=='b0' else result[key][:,:257] if key in ('common','targets') else result[key][:,:256]
            np.testing.assert_array_equal(prefix,original[key])
        np.testing.assert_array_equal(result['targets'],np.bitwise_xor.accumulate(
            np.column_stack((result['b0'],result['commands'])),axis=1))
        if mode=='HOLD':assert not result['commands'][:,256:].any()
