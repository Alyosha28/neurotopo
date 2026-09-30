"""Probe construction checks that separate task evidence from late evidence."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from neurotopo.models import RecurrentModel
from neurotopo.probe import check_disjoint_prefixes, fit_ridge, hidden_at_pre_query, probe_scores, split_batch
from neurotopo.task import INPUT_SIZE


def test_hidden_state_uses_one_prefix_per_counterfactual_pair_and_no_future():
    torch.manual_seed(11)
    model = RecurrentModel(INPUT_SIZE, 64, 4, groups=4, topology="block")
    batch = split_batch(8, 4, 940000)
    hidden = hidden_at_pre_query(model, batch)
    assert hidden.shape == (8, 64)
    first, _ = model(batch.x[::2], batch.pre_index)
    assert torch.equal(model.readout(hidden), first)
    altered = batch.x.clone()
    altered[:, batch.pre_index + 1:] = torch.randn_like(altered[:, batch.pre_index + 1:])
    changed = type(batch)(altered, batch.pre_target, batch.final_target, batch.pre_index,
                          batch.prior, batch.selected, batch.values)
    assert torch.equal(hidden_at_pre_query(model, changed), hidden)


def test_ridge_recovers_candidate_labels_with_equal_head_capacity():
    rng = np.random.default_rng(3)
    labels = torch.from_numpy(np.stack([rng.permutation(4) for _ in range(128)]))
    states = torch.zeros((128, 64))
    states[:, :16] = F.one_hot(labels, num_classes=4).reshape(128, 16).float()
    prior = torch.from_numpy(rng.integers(4, size=128))
    mean, scale, coefficients = fit_ridge(states[:96], labels[:96], penalty=1, std_floor=.001)
    assert coefficients.shape == (65, 16) and coefficients.numel() == 1040
    assert torch.equal(mean, states[:96].double().mean(0))
    score = probe_scores(states[96:], labels[96:], prior[96:], mean, scale, coefficients)
    assert score["uncued_accuracy"] == score["prior_accuracy"] == 1.0
    assert score["all_four_correct"] == 1.0


def test_split_overlap_is_rejected():
    a = split_batch(8, 4, 900000)
    b = split_batch(8, 4, 910000)
    check_disjoint_prefixes({"train": a, "validation": b, "test": split_batch(8, 4, 920000)})
    with pytest.raises(ValueError, match="Duplicate counterfactual prefix"):
        check_disjoint_prefixes({"train": a, "validation": b, "test": a})
