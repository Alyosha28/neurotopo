import numpy as np
import pytest
import torch

from neurotopo.models import RecurrentModel
from neurotopo.source_belief_task import (BAD_RELIABILITY, DISTRACTOR, EVIDENCE,
                                          GOOD_RELIABILITY, HYPOTHESES, INPUT_SIZE,
                                          QUERY, REGIMES, make_source_belief_batch)


def _independent_joint(x: torch.Tensor, stop: int) -> np.ndarray:
    results = []
    for sequence in x.numpy():
        probability = np.ones((HYPOTHESES, REGIMES), dtype=np.float64)
        for token in sequence[:stop]:
            if token[HYPOTHESES + REGIMES + EVIDENCE] == 0:
                continue
            cue = int(token[:HYPOTHESES].argmax())
            source = int(token[HYPOTHESES:HYPOTHESES + REGIMES].argmax())
            for h in range(HYPOTHESES):
                for r in range(REGIMES):
                    reliability = GOOD_RELIABILITY if source == r else BAD_RELIABILITY
                    probability[h, r] *= reliability if h == cue else (1 - reliability) / (HYPOTHESES - 1)
        results.append((probability / probability.sum()).reshape(-1))
    return np.stack(results)


def test_hidden_source_task_oracle_balance_determinism_and_queries():
    batch = make_source_belief_batch(120, np.random.default_rng(305))
    repeat = make_source_belief_batch(120, np.random.default_rng(305))
    assert torch.equal(batch.x, repeat.x)
    assert torch.equal(batch.truth_h, repeat.truth_h)
    assert torch.equal(batch.truth_r, repeat.truth_r)
    assert batch.x.shape == (120, 18, INPUT_SIZE)
    assert batch.pre_index == 8
    np.testing.assert_allclose(batch.pre_joint.numpy(), _independent_joint(batch.x, batch.pre_index), atol=4e-8)
    np.testing.assert_allclose(batch.final_joint.numpy(), _independent_joint(batch.x, 18), atol=4e-8)
    np.testing.assert_allclose(batch.final_joint.sum(1).numpy(), 1, atol=1e-7)
    sources = batch.evidence_sources.numpy()
    assert np.all((sources[:, :4] == 0).sum(1) == 2)
    assert np.all((sources[:, 4:] == 0).sum(1) == 2)
    for time in (batch.pre_index, 17):
        assert torch.all(batch.x[:, time, :HYPOTHESES + REGIMES] == 0)
        assert torch.all(batch.x[:, time, HYPOTHESES + REGIMES + QUERY] == 1)


def test_distractors_change_inputs_but_not_belief_targets():
    no_noise = make_source_belief_batch(64, np.random.default_rng(4), distractors_per_evidence=0)
    noise = make_source_belief_batch(64, np.random.default_rng(4), distractors_per_evidence=1)
    assert torch.equal(no_noise.evidence_cues, noise.evidence_cues)
    assert torch.equal(no_noise.evidence_sources, noise.evidence_sources)
    assert torch.equal(no_noise.pre_joint, noise.pre_joint)
    assert torch.equal(no_noise.final_joint, noise.final_joint)
    assert int(noise.x[:, :, HYPOTHESES + REGIMES + DISTRACTOR].sum()) == 64 * 8


def test_future_evidence_is_not_visible_at_pre_query():
    torch.manual_seed(2)
    model = RecurrentModel(INPUT_SIZE, 16, HYPOTHESES * REGIMES, groups=1, topology="dense")
    batch = make_source_belief_batch(16, np.random.default_rng(9))
    altered = batch.x.clone()
    altered[:, batch.pre_index + 1:] = torch.randn_like(altered[:, batch.pre_index + 1:])
    assert torch.equal(model(batch.x, batch.pre_index)[0], model(altered, batch.pre_index)[0])
    pre, final = model(batch.x, batch.pre_index)
    loss = -(batch.pre_joint * torch.log_softmax(pre, -1)).sum(-1).mean()
    loss += -(batch.final_joint * torch.log_softmax(final, -1)).sum(-1).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("examples,distractors", [(0, 1), (1, -1)])
def test_invalid_arguments(examples, distractors):
    with pytest.raises(ValueError):
        make_source_belief_batch(examples, np.random.default_rng(1),
                                 distractors_per_evidence=distractors)
