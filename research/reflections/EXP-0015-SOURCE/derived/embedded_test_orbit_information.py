"""Deterministic engineering tests: never use the registered scientific RNG roots."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from neurotopo.orbit_information import posterior, generate, bootstrap_means, classify, validate_config
from neurotopo.loop_worker import workflow_commands


def enumeration(y, states):
    result=np.zeros(states)
    for path in itertools.product(range(states), repeat=len(y)+1):
        probability=1/states
        for t, observation in enumerate(y):
            step=(path[t+1]-path[t]) % states
            transition={0:.7,1:.2,states-1:.1}.get(step,0.)
            probability *= transition*(.6 if path[t+1]==observation else .4/(states-1))
        result[path[-1]] += probability
    return result/result.sum()


def test_forward_matches_hidden_path_enumeration_and_suffix_conditioning():
    observations=np.array([[0,1,2],[3,3,1],[1,0,1]],dtype=np.int16)
    actual=posterior(observations,4)
    expected=np.stack([enumeration(row,4) for row in observations])
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-13)
    suffix=posterior(observations[:,-1:],4)
    np.testing.assert_allclose(suffix,np.stack([enumeration(row[-1:],4) for row in observations]),rtol=0,atol=1e-13)
    assert np.max(np.abs(actual-suffix))>.01


def test_prefix_does_not_depend_on_future_and_oracle_is_normalized():
    observations=np.array([[0,2,8,4],[1,3,4,5]],dtype=np.int16)
    prefix=posterior(observations[:,:2])
    observations[:,2:]=15
    np.testing.assert_array_equal(prefix,posterior(observations[:,:2]))
    np.testing.assert_allclose(prefix.sum(axis=1),1,atol=1e-13)


def test_engineering_generator_repeats_and_uses_only_legal_transitions():
    a=generate(17,5,1234);b=generate(17,5,1234)
    for key in a: np.testing.assert_array_equal(a[key],b[key])
    previous=np.concatenate([a['z0'][:,None],a['z'][:,:-1]],axis=1)
    assert set(np.unique((a['z']-previous)%16)) <= {0,1,15}
    assert a['y'].min()>=0 and a['y'].max()<16


def test_bootstrap_stream_is_block_invariant_and_columns_paired():
    values=np.arange(18,dtype=float).reshape(9,2)
    a=bootstrap_means(values,123,31,1);b=bootstrap_means(values,123,31,7)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(a[:,1]-a[:,0],1,atol=1e-14)


def test_fixed_information_thresholds_and_nan_fail_closed():
    good=dict(uniform_mean=.2,suffix1_lower=.05,suffix8_lower=.02)
    assert classify(good)[:2]==('SUPPORTED','TASK_INFORMATION_ONLY')
    for key in good:
        bad={**good,key:good[key]-1e-8}
        assert classify(bad)==('INCONCLUSIVE','TASK_INFORMATION_FAILURE',[key])
        assert classify({**good,key:float('nan')})[1]=='ENGINEERING_ABORT'


def test_frozen_config_rejects_task_threshold_and_budget_changes():
    original=json.loads(Path('configs/exp0015_v1.json').read_text())
    validate_config(original)
    for field,key,value in [('task','root',13),('information','suffix8_kl_ci_lower_min',0.),('resource','research_update_cap',1)]:
        altered=json.loads(json.dumps(original));altered[field][key]=value
        with pytest.raises(ValueError,match='Frozen'):validate_config(altered)


def test_worker_routes_zero_training_workflow_and_rejects_relative_override():
    cfg=json.loads(Path('configs/exp0015_v1.json').read_text())
    command,analysis,_=workflow_commands(Path('configs/exp0015_v1.json'),cfg)
    assert command[2]=='neurotopo.orbit_information'
    assert analysis[2]=='neurotopo.orbit_information_analysis'
    with pytest.raises(ValueError,match='absolute'):
        workflow_commands(Path('config.json'),cfg,Path('relative'))
