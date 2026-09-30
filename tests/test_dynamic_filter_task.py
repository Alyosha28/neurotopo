import hashlib

import numpy as np
import pytest
import torch

from neurotopo.dynamic_filter_task import (EARLY_OBSERVATIONS, INPUT_SIZE, MODES,
                                          make_dynamic_filter_batch)
from neurotopo.models import RecurrentModel


def _brute_posterior(controls: np.ndarray, observations: np.ndarray, stop: int) -> np.ndarray:
    weights = []
    for theta in (-0.85, -0.30, 0.30, 0.85):
        state = np.zeros(2, dtype=np.float64)
        rotation = 0.9 * np.array(((np.cos(theta), -np.sin(theta)),
                                     (np.sin(theta), np.cos(theta))))
        error = 0.0
        for u, y in zip(controls[:stop], observations[:stop]):
            state = rotation @ state + np.asarray((1.0, 0.5)) * u
            error += ((y - np.dot((1.0, -0.3), state)) / 0.8) ** 2
        weights.append(np.exp(-0.5 * error))
    return np.asarray(weights) / sum(weights)


def test_exact_posterior_and_query_boundaries():
    batch = make_dynamic_filter_batch(24, 12, 771)
    assert batch.x.shape == (24, 14, INPUT_SIZE)
    assert batch.pre_index == EARLY_OBSERVATIONS
    assert torch.all(batch.x[:, 4, :3] == 0)
    assert torch.all(batch.x[:, -1, :3] == 0)
    assert torch.all(batch.x[:, [4, -1], 3] == 1)
    assert int(batch.x[:, :, 2].sum()) == 24 * 12
    assert int(batch.x[:, :, 3].sum()) == 24 * 2
    for t in range(12):
        position = t if t < 4 else t + 1
        np.testing.assert_allclose(batch.x[:, position, 0].numpy(),
                                   batch.controls[:, t].numpy(), atol=2e-7)
        np.testing.assert_allclose(batch.x[:, position, 1].numpy(),
                                   batch.observations[:, t].numpy(), atol=2e-7)
        assert torch.all(batch.x[:, position, 2] == 1)
        assert torch.all(batch.x[:, position, 3] == 0)
    for row in range(24):
        for stop in (4, 8, 12):
            expected = _brute_posterior(batch.controls[row].numpy(),
                                        batch.observations[row].numpy(), stop)
            np.testing.assert_allclose(batch.posterior_history[row, stop - 1].numpy(),
                                       expected, atol=1e-12)
    np.testing.assert_allclose(batch.pre_joint.numpy(), batch.posterior_history[:, 3].numpy(), atol=1e-7)
    np.testing.assert_allclose(batch.final_joint.numpy(), batch.posterior_history[:, -1].numpy(), atol=1e-7)
    assert set(batch.truth.unique().tolist()) <= set(range(MODES))


def test_independent_hidden_mode_and_noise_generation():
    examples, steps, seed = 24, 12, 771
    batch = make_dynamic_filter_batch(examples, steps, seed)
    h_seed, u_seed, noise_seed = np.random.SeedSequence(seed).spawn(3)
    h = np.random.default_rng(h_seed).integers(0, 4, size=examples)
    u = np.random.default_rng(u_seed).standard_normal((steps, examples))
    noise = np.random.default_rng(noise_seed).standard_normal((steps, examples))
    np.testing.assert_array_equal(batch.truth.numpy(), h)
    np.testing.assert_array_equal(batch.controls.numpy(), u.T)
    states = np.zeros((examples, 4, 2), dtype=np.float64)
    for t in range(steps):
        means = np.empty((examples, 4), dtype=np.float64)
        for mode, theta in enumerate((-0.85, -0.30, 0.30, 0.85)):
            rotation = 0.9 * np.array(((np.cos(theta), -np.sin(theta)),
                                       (np.sin(theta), np.cos(theta))))
            states[:, mode] = states[:, mode] @ rotation.T + u[t, :, None] * np.array((1.0, 0.5))
            means[:, mode] = states[:, mode] @ np.array((1.0, -0.3))
        expected_y = means[np.arange(examples), h] + 0.8 * noise[t]
        np.testing.assert_allclose(batch.observations[:, t].numpy(), expected_y, atol=1e-12)


def test_seed_determinism_and_cross_length_prefix_pairing():
    short = make_dynamic_filter_batch(60, 8, 9003)
    long = make_dynamic_filter_batch(60, 12, 9003)
    repeat = make_dynamic_filter_batch(60, 8, 9003)
    assert torch.equal(short.x, repeat.x)
    assert torch.equal(short.truth, long.truth)
    assert torch.equal(short.controls, long.controls[:, :8])
    assert torch.equal(short.observations, long.observations[:, :8])
    assert torch.equal(short.x[:, :9], long.x[:, :9])
    assert torch.equal(short.pre_joint, long.pre_joint)
    assert torch.equal(short.posterior_history, long.posterior_history[:, :8])
    assert torch.equal(short.final_joint, long.posterior_history[:, 7].to(torch.float32))
    assert hashlib.sha256(short.x.numpy().tobytes()).hexdigest() == "3f56b0e1e37dc1b5ffba93c53eae5ed9f80f106bd968069d6acc01f62103152c"
    assert hashlib.sha256(short.final_joint.numpy().tobytes()).hexdigest() == "304384a4da3956305da2dc0c031852206c3c1e6659d7c4b6a5ec2ee4a62f1f6f"


def test_model_cannot_read_future_and_gradients_are_finite():
    torch.manual_seed(2)
    model = RecurrentModel(INPUT_SIZE, 64, MODES, groups=1, topology="dense")
    batch = make_dynamic_filter_batch(16, 8, 71)
    altered = batch.x.clone()
    altered[:, batch.pre_index + 1:] = torch.randn_like(altered[:, batch.pre_index + 1:])
    assert torch.equal(model(batch.x, batch.pre_index)[0], model(altered, batch.pre_index)[0])
    pre, final = model(batch.x, batch.pre_index)
    loss = -(batch.pre_joint * torch.log_softmax(pre, -1)).sum(-1).mean()
    loss += -(batch.final_joint * torch.log_softmax(final, -1)).sum(-1).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("examples,observations,seed", [(0, 8, 1), (1, 3, 1), (1, 8, -1)])
def test_invalid_arguments(examples, observations, seed):
    with pytest.raises(ValueError):
        make_dynamic_filter_batch(examples, observations, seed)
