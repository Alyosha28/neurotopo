import copy

import numpy as np
import pytest
import torch

from neurotopo.study import schedule_delays, variant_batch
from neurotopo.task import DISTRACTOR, make_batch


PAYLOADS = ("noise", "full", "blank", "repeat", "id_only", "value_only")


def _assert_same_task(batch, original):
    assert torch.equal(batch.pre_target, original.pre_target)
    assert torch.equal(batch.final_target, original.final_target)
    assert batch.pre_index == original.pre_index
    assert torch.equal(batch.prior, original.prior)
    assert torch.equal(batch.selected, original.selected)
    assert torch.equal(batch.values, original.values)
    assert torch.equal(batch.x[:, :, 8:], original.x[:, :, 8:])


@pytest.mark.parametrize("payload", PAYLOADS)
def test_payload_variants_share_original_examples_labels_and_rng(payload):
    seed = 52
    rng = np.random.default_rng(seed)
    batch = variant_batch(10, 8, rng, payload)
    original = make_batch(10, 8, np.random.default_rng(seed))
    nuisance = original.x[:, :, 8 + DISTRACTOR].bool()

    _assert_same_task(batch, original)
    assert torch.equal(batch.x[~nuisance], original.x[~nuisance])
    assert torch.equal(batch.x[::2, :batch.pre_index + 1], batch.x[1::2, :batch.pre_index + 1])
    assert torch.equal(batch.x[::2, :, 8 + DISTRACTOR], batch.x[1::2, :, 8 + DISTRACTOR])

    # Paired examples differ in their future selector and answer, but their
    # transformed distractors (including post-evidence events) remain identical.
    assert not torch.equal(batch.selected[::2], batch.selected[1::2])
    assert not torch.equal(batch.final_target[::2], batch.final_target[1::2])
    left, right = batch.x[::2], batch.x[1::2]
    left_nuisance = left[:, :, 8 + DISTRACTOR].bool()
    right_nuisance = right[:, :, 8 + DISTRACTOR].bool()
    assert torch.equal(left_nuisance, right_nuisance)
    assert torch.equal(left[:, :, :8][left_nuisance], right[:, :, :8][right_nuisance])
    expected_rng = np.random.default_rng(seed)
    make_batch(10, 8, expected_rng)
    assert rng.bit_generator.state == expected_rng.bit_generator.state

    if payload in {"noise", "full"}:
        assert torch.equal(batch.x, original.x)
    elif payload == "blank":
        assert not batch.x[:, :, :8][nuisance].any()
    elif payload == "repeat":
        first_index = nuisance.to(torch.int64).argmax(dim=1)
        rows = torch.arange(batch.x.shape[0])
        first = batch.x[rows, first_index, :8]
        nuisance_payload = batch.x[:, :, :8][nuisance]
        expected = first[:, None, :].expand(-1, batch.x.shape[1], -1)[nuisance]
        assert torch.equal(nuisance_payload, expected)
        assert torch.equal(nuisance_payload.sum(dim=1), torch.full_like(nuisance_payload[:, 0], 2.0))
    elif payload == "id_only":
        assert torch.equal(batch.x[:, :, :4][nuisance], original.x[:, :, :4][nuisance])
        assert not batch.x[:, :, 4:8][nuisance].any()
        selected_payload = batch.x[:, :, :8][nuisance]
        assert torch.equal(selected_payload.sum(dim=1), torch.ones_like(selected_payload[:, 0]))
    elif payload == "value_only":
        assert not batch.x[:, :, :4][nuisance].any()
        assert torch.equal(batch.x[:, :, 4:8][nuisance], original.x[:, :, 4:8][nuisance])
        selected_payload = batch.x[:, :, :8][nuisance]
        assert torch.equal(selected_payload.sum(dim=1), torch.ones_like(selected_payload[:, 0]))


def test_all_payload_variants_leave_rng_at_same_state():
    states = []
    for payload in PAYLOADS:
        rng = np.random.default_rng(1234)
        variant_batch(6, 4, rng, payload)
        states.append(copy.deepcopy(rng.bit_generator.state))
    assert all(state == states[0] for state in states[1:])


def test_unknown_payload_is_rejected_without_consuming_rng():
    rng = np.random.default_rng(99)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError):
        variant_batch(2, 0, rng, "not_registered")
    assert rng.bit_generator.state == before


def test_schedule_boundaries_and_token_difference():
    curriculum = schedule_delays([[0, 2000], [2, 1000], [4, 1000], [8, 2000]])
    assert len(curriculum) == 6000
    assert [curriculum[i] for i in [1999, 2000, 2999, 3000, 3999, 4000, 5999]] == [0, 2, 2, 4, 4, 8, 8]
    assert sum(d + 10 for d in curriculum) == 82000
    assert sum(d + 10 for d in schedule_delays([[8,6000]])) == 108000
    with pytest.raises(ValueError):
        schedule_delays([[8,0]])
