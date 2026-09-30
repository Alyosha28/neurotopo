import numpy as np
import pytest
import torch

from neurotopo.belief_task import (DISTRACTOR, EVIDENCE, HYPOTHESES, INPUT_SIZE,
                                   QUERY, RELIABILITIES, make_belief_batch)
from neurotopo.models import RecurrentModel


def _independent_oracle(x: torch.Tensor, stop: int) -> np.ndarray:
    answer = []
    for sequence in x.numpy():
        likelihood = np.ones(HYPOTHESES, dtype=np.float64)
        for token in sequence[:stop]:
            if token[HYPOTHESES + len(RELIABILITIES) + EVIDENCE] == 0:
                continue
            cue = int(np.argmax(token[:HYPOTHESES]))
            reliability = RELIABILITIES[int(np.argmax(token[HYPOTHESES:HYPOTHESES + len(RELIABILITIES)]))]
            likelihood *= np.array([reliability if hypothesis == cue else
                                    (1 - reliability) / (HYPOTHESES - 1)
                                    for hypothesis in range(HYPOTHESES)])
        answer.append(likelihood / likelihood.sum())
    return np.stack(answer)


def test_determinism_oracle_and_no_query_leakage():
    batch = make_belief_batch(100, np.random.default_rng(73))
    repeat = make_belief_batch(100, np.random.default_rng(73))
    assert torch.equal(batch.x, repeat.x)
    assert torch.equal(batch.truth, repeat.truth)
    assert batch.x.shape == (100, 14, INPUT_SIZE)
    assert batch.pre_index == 6
    np.testing.assert_allclose(batch.pre_posterior.numpy(), _independent_oracle(batch.x, batch.pre_index), atol=3e-8)
    np.testing.assert_allclose(batch.final_posterior.numpy(), _independent_oracle(batch.x, len(batch.x[0])), atol=3e-8)
    np.testing.assert_allclose(batch.pre_posterior.sum(1), 1, atol=1e-7)
    np.testing.assert_allclose(batch.final_posterior.sum(1), 1, atol=1e-7)
    assert torch.all(batch.x[:, batch.pre_index, :HYPOTHESES + len(RELIABILITIES)] == 0)
    assert torch.all(batch.x[:, -1, :HYPOTHESES + len(RELIABILITIES)] == 0)
    assert torch.all(batch.x[:, batch.pre_index, HYPOTHESES + len(RELIABILITIES) + QUERY] == 1)


def test_distractor_count_changes_input_not_posterior():
    batches = [make_belief_batch(80, np.random.default_rng(22), distractors_per_evidence=d)
               for d in (0, 1, 2)]
    for batch in batches[1:]:
        assert torch.equal(batch.truth, batches[0].truth)
        assert torch.equal(batch.evidence_cues, batches[0].evidence_cues)
        assert torch.equal(batch.evidence_reliabilities, batches[0].evidence_reliabilities)
        assert torch.equal(batch.pre_posterior, batches[0].pre_posterior)
        assert torch.equal(batch.final_posterior, batches[0].final_posterior)
    assert not torch.equal(batches[0].x, batches[1].x)
    distractor_tag = HYPOTHESES + len(RELIABILITIES) + DISTRACTOR
    assert int(batches[1].x[:, :, distractor_tag].sum().item()) == 80 * 6


def test_future_evidence_cannot_change_pre_query_model_output():
    torch.manual_seed(5)
    model = RecurrentModel(INPUT_SIZE, 16, HYPOTHESES, groups=4, topology="block")
    batch = make_belief_batch(10, np.random.default_rng(42))
    changed = batch.x.clone()
    changed[:, batch.pre_index + 1:] = torch.randn_like(changed[:, batch.pre_index + 1:])
    torch.testing.assert_close(model(batch.x, batch.pre_index)[0], model(changed, batch.pre_index)[0], rtol=0, atol=0)
    pre, final = model(batch.x, batch.pre_index)
    loss = -(batch.pre_posterior * torch.log_softmax(pre, -1)).sum(-1).mean()
    loss += -(batch.final_posterior * torch.log_softmax(final, -1)).sum(-1).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("examples,pre,late,distractors", [(0, 3, 3, 1), (1, 0, 3, 1), (1, 3, 0, 1), (1, 3, 3, -1)])
def test_invalid_generator_arguments(examples, pre, late, distractors):
    with pytest.raises(ValueError):
        make_belief_batch(examples, np.random.default_rng(1), pre_evidence=pre,
                          late_evidence=late, distractors_per_evidence=distractors)
