import json
from pathlib import Path
import numpy as np
import pytest
import torch
from neurotopo.co_paired_dynamics import (prefix_pools,paired_data,check_pairing,bit_equal,
                                        pair_curves,curve_summary,decision,SAMPLE_INDICES,visible,infer_branch,validate_config)
from neurotopo.order_observation import ControlGRU,oracle


def test_prefix_pool_endpoints_and_all_words_once():
    pools=prefix_pools();assert sum(len(p) for p in pools)==4096
    assert len({r.tobytes() for p in pools for r in p})==4096
    for state,pool in enumerate(pools):
        tok,q=visible(pool)
        assert np.all(q[:,-1]==state)
        assert np.array_equal(tok[:,4::5],3+(q[:,4::5]%3==0))


def test_deterministic_pairing_and_independent_oracle():
    a=paired_data(19,199001,n=6);b=paired_data(19,199001,n=6)
    assert all(np.array_equal(a[k],b[k]) for k in a);check_pairing(a)
    for branch in range(3):
        q=oracle(a['tokens'][branch],'CO')
        np.testing.assert_array_equal(q,np.eye(6)[a['states'][branch]])
    assert SAMPLE_INDICES.tolist()==[14,15,16,19,24,34,54,94,174]


def test_exact_equality_not_nearness_and_nonfinite_rejection():
    a=np.array([[[1.],[0.],[0.]]],dtype=np.float32);b=np.array([[[-1.],[0.],[0.]]],dtype=np.float32)
    result=pair_curves(a,b);assert result['equal'].tolist()==[[False,True,True]]
    s=curve_summary(result);assert s['new_merger_count']==1 and s['first_merge_relative_token']==[1]
    b=a.copy();b[0,0,0]=np.nextafter(np.float32(1),np.float32(2))
    assert not bit_equal(a,b)[0,0]
    assert not bit_equal(np.array([[0.]],dtype=np.float32),np.array([[-0.]],dtype=np.float32))[0]
    for value in [np.nan,np.inf]:
        bad=np.array([[[value]]],dtype=np.float32)
        with pytest.raises(FloatingPointError):bit_equal(bad,bad)
    with pytest.raises(AssertionError,match='revived'):
        pair_curves(np.array([[[0.],[1.]]]),np.array([[[0.],[2.]]]))


def test_fixed_full_batch_forward_duplicate():
    torch.set_num_threads(1)
    model=ControlGRU('D16',199001);model.eval();model.requires_grad_(False)
    tokens=np.tile(np.array([0,1,0,1,3],dtype=np.int8),(128,4))
    for dtype in [torch.float32,torch.float64]:
        model.to(dtype=dtype)
        a,z=infer_branch(model,tokens);b,w=infer_branch(model,tokens)
        assert bit_equal(a,b).all() and bit_equal(z,w).all()
        assert all(p.grad is None for p in model.parameters())


def test_decision_primary_only_and_initial_mergers_not_new():
    rows=[dict(seed=s,arm=a,precision=d,opposite_b=dict(initial_distinct_fraction=1.,terminal_merge_fraction_among_initial_distinct=.9,new_merger_count=346))
          for s in (17001,17002,17003) for a in ('D64','D16') for d in ('float32','float64')]
    assert decision(rows)==('SUPPORTED','FINITE_PRECISION_PAIR_COLLAPSE')
    for r in rows:
        r['opposite_b'].update(initial_distinct_fraction=0.,terminal_merge_fraction_among_initial_distinct=None,new_merger_count=0)
    assert decision(rows)[1]=='NO_NEW_EXACT_MERGER_OBSERVED'
    rows[0]['opposite_b']['new_merger_count']=1
    assert decision(rows)[1]=='PARTIAL_PAIR_COLLAPSE'
    assert decision(rows[:-1])[1]=='ENGINEERING_ABORT'


def test_frozen_config_and_route():
    from neurotopo.loop_worker import workflow_commands
    c=json.loads(Path('configs/exp0019_v2.json').read_text());validate_config(c)
    assert workflow_commands(Path('configs/exp0019_v2.json'),c)[0][2]=='neurotopo.co_paired_dynamics'
