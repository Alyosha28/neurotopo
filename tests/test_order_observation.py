"""Engineering inputs only; no registered research stream is accessed."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from neurotopo.order_observation import (CELLS, SEEDS, ARMS, ControlGRU, generate, oracle,
                                        kl_scores, classify, validate_config)
from neurotopo.order_observation_reproduce import log_oracle, independent_kl


@pytest.mark.parametrize('cell', CELLS)
def test_oracle_independent_log_domain_and_observability(cell):
    data = generate(24, 12, cell, [179001, 0, CELLS.index(cell), 0])
    q = data['target']
    np.testing.assert_allclose(q.sum(-1), 1, rtol=0, atol=1e-14)
    np.testing.assert_allclose(q, log_oracle(data['tokens'], cell), rtol=0, atol=1e-14)
    assert np.all(q[np.arange(24), -1, data['latent_final']] > 0)
    if cell[1] == 'O':
        assert np.all((q > 0).sum(-1) == 1)
        assert np.array_equal(q[:, -1].argmax(-1), data['latent_final'])


def test_order_counterfactual_and_exact_ambiguous_sensor():
    pair = np.array([[0, 1], [1, 0]], dtype=np.int8)
    assert np.array_equal(oracle(pair, 'CO')[0, -1], oracle(pair, 'CO')[1, -1])
    assert not np.array_equal(oracle(pair, 'NO')[0, -1], oracle(pair, 'NO')[1, -1])
    q = oracle(np.array([[2, 4]]), 'NU')
    np.testing.assert_array_equal(q[0, 0], [0, .5, 0, .5, 0, 0])
    np.testing.assert_array_equal(q[0, 1], [0, 0, 0, 1, 0, 0])
    with pytest.raises(ValueError, match='Impossible'):
        oracle(np.array([[0, 4]]), 'CO')


def test_local_rng_determinism_and_prefix_no_future_leakage():
    np.random.seed(918)
    before = np.random.get_state()
    a = generate(12, 8, 'NU', [179001, 1, 3, 0])
    b = generate(12, 8, 'NU', [179001, 1, 3, 0])
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert all(np.array_equal(a[k], b[k]) for k in a)
    np.testing.assert_array_equal(oracle(a['tokens'][:, :5], 'NU'), a['target'][:, :5])


def test_parameters_initialization_and_no_candidate():
    for arm, expected in [('D16',1206),('D64',14022)]:
        a, b = ControlGRU(arm,19), ControlGRU(arm,19)
        assert sum(p.numel() for p in a.parameters()) == expected
        assert all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items())
    with pytest.raises(ValueError):
        ControlGRU('A2',19)


def records():
    return [dict(cell=c, seed=s, arm=a, status='COMPLETED', updates=750,
                 metrics={'32':{'kl_mean':.05}, '128':{'kl_mean':.1}})
            for c in CELLS for s in SEEDS for a in ARMS]


def test_decision_aggregation_boundaries_and_priority():
    rows = records()
    assert classify(rows)[:2] == ('SUPPORTED','CONTROL_PILOT_WINDOW')
    # v2: a low seed does not fail the cell mean gate.
    ds = [r for r in rows if r['cell']=='CO' and r['arm']=='D64']
    for r,x in zip(ds,[0.,.02,.04]):r['metrics']['128']['kl_mean']=x
    assert classify(rows)[:2] == ('SUPPORTED','CONTROL_PILOT_WINDOW')
    ds[-1]['metrics']['128']['kl_mean']=.039
    assert classify(rows)[1]=='CONTROL_CEILING'
    ds[0]['metrics']['32']['kl_mean']=.1
    assert classify(rows)[1]=='LEARNING_FAILURE'
    bad=deepcopy(rows);bad[-1]=bad[0]
    assert classify(bad)[1]=='ENGINEERING_ABORT'
    assert classify(rows[:-1])[1]=='ENGINEERING_ABORT'
    rows=records();rows[0]['metrics']['32']['kl_mean']=3.
    assert classify(rows)[1]=='CONTROL_PILOT_WINDOW'  # D16 does not gate.


def test_zero_probabilities_and_oracle_negative_control():
    q=np.array([[1,0,0,0,0,0],[.5,0,0,.5,0,0]])
    # Finite logits approximate exact zeros without modifying the oracle.
    z=np.full_like(q,-1000.);active=q>0;z[active]=np.log(q[active])
    np.testing.assert_allclose(kl_scores(q,z),0,atol=1e-14)
    np.testing.assert_allclose(kl_scores(q,z),independent_kl(q,z),atol=1e-14)
    assert (q[1]>0).sum()==2  # Correct posterior is ambiguous, not a shortcut.


def test_frozen_config_and_worker_route():
    from neurotopo.loop_worker import workflow_commands
    c=json.loads(Path('configs/exp0017_v2.json').read_text());validate_config(c)
    cmd,_,_=workflow_commands(Path('configs/exp0017_v2.json'),c)
    assert cmd[2]=='neurotopo.order_observation'
    c['training']['updates']=751
    with pytest.raises(ValueError,match='Frozen'):validate_config(c)


def test_internal_timeout_survives_worker_exit_wrapper():
    from neurotopo.order_observation import failure_status
    failure={'stage':'run','substatus':'ENGINEERING_ABORT'}
    assert failure_status(failure,{'error':"TimeoutError('budget')"})=='INCOMPLETE_BUDGET'
    assert failure_status(failure,{'error':"ValueError('invalid posterior')"})=='ENGINEERING_ABORT'
    assert failure_status({'stage':'analysis','substatus':'ENGINEERING_ABORT'}, {})=='ENGINEERING_ABORT'
