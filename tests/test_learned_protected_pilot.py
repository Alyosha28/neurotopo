"""Engineering inputs only: no experiment root or trained model."""
import json
from pathlib import Path
import numpy as np
import torch
from neurotopo.learned_protected_pilot import (LearnedMemory,radial,generate,rotations,tensors,
                                              metrics,state_hash,classify,length_at,validate_config)


def test_rotations_and_targets_local_randomness():
    np.random.seed(77);before=np.random.get_state()
    a=generate(16,8,[209002,0,0,0],'MIX');b=generate(16,8,[209002,0,0,0],'MIX')
    after=np.random.get_state();assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    assert all(np.array_equal(a[k],b[k]) for k in a)
    np.testing.assert_array_equal(a['targets'][:,1:],(a['b0'][:,None]+np.cumsum(a['commands'],axis=1))%2)
    np.testing.assert_allclose(a['common'].swapaxes(-1,-2)@a['common'],np.broadcast_to(np.eye(3),a['common'].shape),atol=2e-7,rtol=0)
    np.testing.assert_allclose(np.linalg.det(a['common']),1,atol=2e-7,rtol=0)
    assert np.sum(a['commands'][:,0]==1)==8


def test_initialization_parameter_accounting_and_radial_zero_gradient():
    ms=[LearnedMemory(a,209002) for a in ['RAW','INV','REL']]
    assert [sum(p.numel() for p in m.parameters()) for m in ms]==[502,406,340]
    assert len({state_hash(m,'encoder.') for m in ms})==1 and len({state_hash(m,'readout.') for m in ms})==1
    x=torch.zeros(2,3,2,requires_grad=True);radial(x).sum().backward()
    torch.testing.assert_close(x.grad,torch.full_like(x,1.5));assert torch.isfinite(x.grad).all()


def test_equivariance_and_invariance_in_their_defined_forms():
    torch.set_num_threads(1);r=np.random.default_rng(209002)
    h=torch.from_numpy(r.normal(size=(16,3,2)).astype('float32'));u=torch.arange(16)%2
    rot=torch.from_numpy(rotations(r,(16,),np.pi))
    rel=LearnedMemory('REL',209002);inv=LearnedMemory('INV',209002)
    torch.testing.assert_close(rel.transition(rot@h,u),rot@rel.transition(h,u),atol=2e-6,rtol=0)
    torch.testing.assert_close(inv.transition(rot@h,u),inv.transition(h,u),atol=2e-6,rtol=0)


def test_no_future_input_leakage_and_all_parts_receive_gradients():
    d=generate(16,4,[209002,1,0,0],'MIX');inputs=tensors(d)
    for arm in ['RAW','INV','REL']:
        model=LearnedMemory(arm,209002);full=model(*inputs)
        prefix=model(inputs[0],inputs[1][:,:2],inputs[2][:,:3],inputs[3][:,:2])
        torch.testing.assert_close(full[:,:3],prefix,atol=1e-7,rtol=0)
        torch.nn.functional.binary_cross_entropy_with_logits(full,torch.from_numpy(d['targets'])).backward()
        for name,module in [('encoder',model.encoder),('readout',model.readout),('controller',model.controller)]:
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters()),name


def test_pairing_metrics_and_pilot_branches():
    a=generate(16,8,[209002,2,0,0],'UPDATE',.4,0.)
    b=generate(16,8,[209002,2,0,0],'UPDATE',np.pi,.1)
    assert np.array_equal(a['commands'],b['commands']) and np.array_equal(a['b0'],b['b0'])
    assert metrics((a['targets']*2-1)*10,a)['flip_accuracy']==1.
    rows=[{'arm':arm,'updates':1000,'evaluation':{'IID_H':{'accuracy_all':.99},'IID_U':{'accuracy_all':.99,'flip_accuracy':.99}}} for arm in ['RAW','INV','REL']]
    rows[0]['evaluation']['IID_H']['accuracy_all']=.5
    assert classify(rows)==('SUPPORTED','SINGLE_SEED_LEARNABILITY_ONLY')
    rows[-1]['evaluation']['IID_U']['flip_accuracy']=.94
    assert classify(rows)[1]=='EXPLORATORY_LEARNING_UNRESOLVED'
    assert [length_at(s) for s in [1,125,126,250,251,500,501,1000]]==[1,1,2,2,4,4,8,8]


def test_config_route():
    from neurotopo.loop_worker import workflow_commands
    c=json.loads(Path('configs/exp0020_v2.json').read_text());validate_config(c)
    assert workflow_commands(Path('configs/exp0020_v2.json'),c)[0][2]=='neurotopo.learned_protected_pilot'
