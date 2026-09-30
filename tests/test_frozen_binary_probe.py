"""Small engineering checks on root189001; never scientific streams."""
import json
from pathlib import Path
import numpy as np
import torch
from neurotopo.frozen_binary_probe import (objective,fit,scaling,transform,metrics,classify,
                                         extract,standard_state_hash,validate_config)
from neurotopo.order_observation import ControlGRU,input_tensor


def test_gradient_hessian_against_finite_differences():
    r=np.random.default_rng(189001);x=np.column_stack((r.normal(size=(24,3)),np.ones(24)));y=r.integers(0,2,24);w=r.normal(size=4)
    _,g,h=objective(x,y,w);eps=1e-5
    for j in range(4):
        d=np.eye(4)[j]*eps
        np.testing.assert_allclose((objective(x,y,w+d)[0]-objective(x,y,w-d)[0])/(2*eps),g[j],atol=1e-9,rtol=0)
        np.testing.assert_allclose((objective(x,y,w+d)[1]-objective(x,y,w-d)[1])/(2*eps),h[:,j],atol=1e-9,rtol=0)


def test_known_signal_convergence_and_shuffle_negative_control():
    r=np.random.default_rng(189001);h=r.normal(size=(1024,5));v=r.normal(size=(1024,5));y=(h[:,0]>.0).astype(float);truth=(v[:,0]>0).astype(float)
    mean,scale=scaling(h);x=transform(h,mean,scale);test=transform(v,mean,scale)
    good=fit(x,y);bad=fit(x,r.permutation(y))
    assert good['converged'] and bad['converged']
    assert all(t['objective_after']<=t['objective_before'] for t in good['trace'])
    assert metrics(test@np.array(good['weights']),truth)['accuracy']>.95
    assert metrics(test@np.array(bad['weights']),truth)['accuracy']<.65


def test_scaling_uses_fit_only_and_constant_columns():
    h=np.array([[1.,2.],[3.,2.]]);m,s=scaling(h)
    np.testing.assert_array_equal(m,[2,2]);np.testing.assert_array_equal(s,[1,1])
    x=transform(np.array([[101.,200.]]),m,s);np.testing.assert_array_equal(x,[[99,198,1]])
    np.testing.assert_array_equal(m,[2,2])


def test_frozen_extraction_matches_registered_full_forward():
    torch.set_num_threads(1)
    model=ControlGRU('D64',189001);model.eval();model.requires_grad_(False)
    tokens=(np.arange(16*40).reshape(16,40)%5).astype(np.int8);before=standard_state_hash(model)
    hidden,z=extract(model,tokens)
    with torch.no_grad():expected=model(input_tensor(tokens))[:,-1].numpy()
    np.testing.assert_array_equal(z,expected);assert hidden.shape==(16,64)
    assert before==standard_state_hash(model) and all(p.grad is None for p in model.parameters())


def rows(acc=.5,ood=None):
    result=[]
    for seed in [17001,17002,17003]:
        for arm in ['D64','D16']:
            test={}
            for n in ['32','128']:
                a=acc if n=='32' or ood is None else ood
                test[n]={'true':{'accuracy':a,'positive_fraction':.5},'shuffle':{'accuracy':.5},'gap':a-.5,'parent':{'b_accuracy':.5,'j_accuracy':1.}}
            result.append({'seed':seed,'arm':arm,'fits':{'true':{'converged':True},'shuffle':{'converged':True}},'test':test})
    return result


def test_branches_and_anchors_do_not_gate():
    assert classify(rows())[1]=='NO_USEFUL_LINEAR_READOUT'
    assert classify(rows(.85))[0]=='SUPPORTED'
    assert classify(rows(.85,.6))[1]=='LENGTH_LIMITED_LINEAR_READOUT'
    assert classify(rows(.6))[1]=='LIMITED_LINEAR_SIGNAL'
    r=rows(.85);r[0]['test']['32']['shuffle']['accuracy']=.6
    assert classify(r)[1]=='CONTROL_FAILURE'
    r=rows();r[1]['test']['32']['parent']['j_accuracy']=0.
    assert classify(r)[1]=='NO_USEFUL_LINEAR_READOUT'
    r[0]['fits']['true']['converged']=False
    assert classify(r)[1]=='PROBE_OPTIMIZATION_FAILURE'


def test_config_and_worker_routing():
    from neurotopo.loop_worker import workflow_commands
    c=json.loads(Path('configs/exp0018_v1.json').read_text());validate_config(c)
    cmd,_,_=workflow_commands(Path('configs/exp0018_v1.json'),c)
    assert cmd[2]=='neurotopo.frozen_binary_probe'
