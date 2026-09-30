"""Small deterministic engineering checks, never the EXP-0016 scientific roots."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from neurotopo.orbit_control_pilot import LearnedHMM,ControlGRU,project_cyclic,project_conjugate,build_model,tensor_hash,scores,classify,hmm_ceiling,validate_config
from neurotopo.orbit_information import posterior


def test_hmm_with_known_transition_matches_oracle_and_column_orientation():
    torch.set_num_threads(1)
    y=np.array([[0,1,1,2,3,3],[8,8,7,7,7,6]],dtype=np.int16)
    transition=torch.zeros(16,16)
    for j in range(16):transition[j,j]=.7;transition[(j+1)%16,j]=.2;transition[(j-1)%16,j]=.1
    model=LearnedHMM()
    with torch.no_grad():model.transition_logits.copy_(transition.log())
    q=model(F.one_hot(torch.from_numpy(y.astype(np.int64)),16).float()).exp().detach().numpy()
    np.testing.assert_allclose(q,posterior(y),rtol=0,atol=2e-6)


def test_exact_parameters_and_initialization_pairing_and_forbidden_candidate():
    for arm,expected in [('H',256),('B',1904),('U',1904),('D',16784)]:
        assert sum(p.numel() for p in build_model(arm,19).parameters())==expected
    assert tensor_hash(build_model('B',19))==tensor_hash(build_model('U',19))
    with pytest.raises(ValueError):build_model('A',19)


def test_conjugate_projector_and_penalty_gradient():
    w=torch.arange(256,dtype=torch.float64).reshape(16,16)/256
    projected=project_conjugate(w)
    torch.testing.assert_close(project_conjugate(projected),projected,rtol=0,atol=1e-13)
    p=torch.tensor([0,5,11,2,14,7,3,12,8,1,15,6,10,4,13,9]);q=torch.eye(16,dtype=torch.float64)[p]
    torch.testing.assert_close(projected,q@project_cyclic(q.T@w@q)@q.T,rtol=0,atol=1e-13)
    model=ControlGRU('B',19);loss=model.penalty();loss.backward()
    assert loss>0 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def record(arm,seed,iid=.05,ood=.1,upper=None):
    upper=ood+.001 if upper is None else upper
    return dict(arm=arm,seed=seed,status='COMPLETED',metrics={'32':dict(kl_mean=iid,ci=[iid-.001,iid+.001]),'64':dict(kl_mean=ood,ci=[ood-.001,upper])})


def test_staged_decision_failure_priority_and_no_training_after_ceiling():
    hs=[record('H',s,ood=.01) for s in [16001,16002,16003]]
    assert hmm_ceiling(hs) and classify(hs)==('INCONCLUSIVE','SIMPLE_ALGORITHM_CEILING')
    with pytest.raises(ValueError):classify(hs+[record('U',16001)])
    assert not hmm_ceiling([hs[0],hs[0],hs[0]])
    hs[0]['metrics']['64']['ci'][1]=float('nan')
    assert classify(hs)==('INCONCLUSIVE','ENGINEERING_ABORT')
    rows=[record(a,s) for a in ['H','B','U','D'] for s in [16001,16002,16003]]
    assert classify(rows)==('SUPPORTED','CONTROL_PILOT_WINDOW')
    rows[-1]['metrics']['32']['kl_mean']=.2
    assert classify(rows)==('INCONCLUSIVE','LEARNING_FAILURE')


def test_changed_config_rejected():
    c=json.loads(Path('configs/exp0016_v1.json').read_text());validate_config(c)
    c['training']['lr']=.003
    with pytest.raises(ValueError,match='Frozen'):validate_config(c)


def test_score_renormalizes_float32_log_probabilities():
    q=np.tile(np.arange(1,17,dtype=float),(2,1));q/=q.sum(axis=1,keepdims=True)
    np.testing.assert_allclose(scores(q,np.log(q).astype(np.float32)),0,rtol=0,atol=1e-10)


def test_independent_branch_recompute_detects_missing_or_forbidden_arms():
    from neurotopo.orbit_control_reproduce import decision
    hs=[record('H',s,ood=.01) for s in [16001,16002,16003]]
    assert decision(hs)==classify(hs)
    with pytest.raises(ValueError,match='GRU training'):
        decision(hs+[record('B',16001)])
    with pytest.raises(ValueError,match='Missing H'):
        decision(hs[:2])
    rows=[record(a,s) for a in ['H','B','U','D'] for s in [16001,16002,16003]]
    assert decision(rows)==classify(rows)
    rows[-1]['metrics']['64']['kl_mean']=.02
    assert decision(rows)==classify(rows)==('INCONCLUSIVE','CONTROL_CEILING')
