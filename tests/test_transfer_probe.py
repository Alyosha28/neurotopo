"""Cross-delay readout controls on synthetic states, not scientific results."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from neurotopo.transfer_probe import fit_prior_head, score_head, validate_splits
from neurotopo.probe import split_batch


def test_same_scaler_exposes_mapping_shift_without_erasing_information():
    rng = np.random.default_rng(5)
    labels = torch.from_numpy(rng.integers(4, size=400))
    short = torch.zeros((400, 64))
    long = torch.zeros_like(short)
    short[:, :4] = F.one_hot(labels, 4).float()
    long[:, :4] = F.one_hot(3 - labels, 4).float()
    mean = short[:300].double().mean(0)
    scale = short[:300].double().std(0, unbiased=False).clamp_min(.001)
    short_head = fit_prior_head(short[:300], labels[:300], mean, scale, penalty=1)
    long_head = fit_prior_head(long[:300], labels[:300], mean, scale, penalty=1)
    assert short_head.shape == long_head.shape == (65, 4)
    assert score_head(long[300:], labels[300:], mean, scale, long_head)["accuracy"] == 1
    assert score_head(long[300:], labels[300:], mean, scale, short_head)["accuracy"] == 0


def test_duplicate_prefix_is_rejected_even_between_fit_streams():
    a = split_batch(8, 8, 930000)
    b = split_batch(8, 8, 935000)
    validate_splits({"A8": a, "B8": b, "V8": split_batch(8, 8, 940000)})
    with pytest.raises(ValueError, match="duplicate prefix"):
        validate_splits({"A8": a, "B8": a})
