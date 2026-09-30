"""Engineering-only tests for the EXP0021 one-shot intervention contract."""
import numpy as np
import torch
from neurotopo.learned_protected_pilot import LearnedMemory, generate, predict_with_gram
from neurotopo.protected_interventions import perturbations, intervention_rollout, intervention_metrics


def test_intervention_invariants_and_matching():
    rng = np.random.default_rng(209021)
    h = rng.normal(size=(128,3,2)).astype(np.float32)*.7
    p = perturbations(h,209021)
    g = h.swapaxes(1,2)@h
    for key in ('NONE','NORM_SHAM','ANGLE_SHAM'):
        v = p[key]['after']
        np.testing.assert_allclose(v.swapaxes(1,2)@v,g,atol=2e-6,rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(p['NORM']['after'],axis=1),1,atol=2e-7)
    np.testing.assert_allclose(np.linalg.norm(p['ANGLE']['after'],axis=1),np.linalg.norm(h,axis=1),atol=3e-7)
    mask = ~p['NORM_SHAM']['norm_sham_unreachable']
    assert np.max(np.abs(p['NORM_SHAM']['norm_sham_residual'][mask])) < 1e-10
    z = perturbations(np.zeros((4,3,2),np.float32),209021)
    for v in z.values():
        assert np.isfinite(v['after']).all() and not v['after'].any()
    tiny = perturbations(h*1e-6,209021)
    assert tiny['NORM_SHAM']['norm_sham_unreachable'].all()
    assert (tiny['NORM_SHAM']['norm_sham_residual'] < 0).all()


def test_none_exact_replay_t32_and_target_independence():
    torch.set_num_threads(1)
    data = generate(130,64,[209021,1], 'UPDATE',np.pi,0.)
    for arm in ('RAW','INV','REL'):
        model = LearnedMemory(arm,209021)
        z,g = predict_with_gram(model,data)
        effects = intervention_rollout(model,data,209021)
        np.testing.assert_array_equal(effects['NONE']['logits'],z[:,32:])
        np.testing.assert_array_equal(effects['NONE']['gram'],g[:,32:])
        changed = dict(data); changed['targets'] = 1-data['targets']
        other = intervention_rollout(model,changed,209021)
        for name in effects:
            np.testing.assert_array_equal(effects[name]['logits'],other[name]['logits'])
        scores = intervention_metrics(effects['NONE'],data)
        assert scores['immediate_accuracy'] == np.mean((z[:,32]>=0)==data['targets'][:,32])
        assert scores['suffix_accuracy'] == np.mean((z[:,33:]>=0)==data['targets'][:,33:])


def test_seed_isolation_and_reproducibility():
    a = generate(16,8,[209021,2], 'MIX')
    b = generate(16,8,[209021,3], 'MIX')
    assert not np.array_equal(a['common'],b['common'])
    np.testing.assert_array_equal(a['targets'][:,1:],(a['b0'][:,None]+np.cumsum(a['commands'],1))%2)
    h = np.ones((8,3,2),np.float32)
    a = perturbations(h,209021); b = perturbations(h,209021)
    for name in a:
        for key in a[name]: np.testing.assert_array_equal(a[name][key],b[name][key])


def test_registered_decisions_exclude_exploratory_seed():
    from copy import deepcopy
    from neurotopo.protected_replication_analysis import _decision
    tables=[dict(seed=seed,replication_pass=True,mechanism_pass=True,
                 iid_learnability={'INV':True,'REL':True},none_immediate_accuracy={'INV':1.,'REL':1.})
            for seed in [20002,20003,20004,20005,20001]]
    tables[-1]['replication_pass']=tables[-1]['mechanism_pass']=False
    assert _decision(tables)[:2]==('SUPPORTED','REPLICATION_AND_CROSSOVER')
    tables[0]['mechanism_pass']=False
    assert _decision(tables)[:2]==('PARTIALLY_SUPPORTED','REPLICATION_ONLY')
    tables[0]['replication_pass']=False
    assert _decision(tables)[0]=='FALSIFIED'
    tables[0]['iid_learnability']['REL']=False
    assert _decision(tables)[0]=='INCONCLUSIVE'


def test_monitor_resource_count_and_config():
    import json
    from pathlib import Path
    from neurotopo.protected_replication import _resource_at,validate_config
    validate_config(json.loads(Path('configs/exp0021_v1.json').read_text()))
    m=LearnedMemory('REL',209021)
    for step,expected in [(0,0),(125,125),(250,375),(500,1375),(1000,5375)]:
        r=_resource_at(step,m)
        assert r['operation_tokens']==64*expected and r['examples']==64*step
