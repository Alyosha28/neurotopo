"""Engineering-only checks; no qualification, formal, or task-information roots."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from neurotopo.pfstyle_task import (MATRIX_ARRAY_SHA256, PRE_OBSERVATIONS,
                                    frozen_matrices, make_batch)


ENGINEERING_ROOT = 9_900_137


def _independent_oracle(u: np.ndarray, y: np.ndarray,
                        matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scalar mode/time/state loops separate from the production vectorization."""
    steps = len(u)
    log_weights = np.empty((steps, 4), dtype=np.float64)
    mode_means = np.empty((steps, 4), dtype=np.float64)
    for mode in range(4):
        state = [0.0] * 8
        log_probability = -np.log(4.0)
        for t in range(steps):
            next_state = []
            for row in range(8):
                value = sum(float(matrices[mode, row, col]) * state[col]
                            for col in range(8))
                next_state.append(value + (float(u[t]) if row == 0 else 0.0))
            state = next_state
            mode_means[t, mode] = state[0]
            residual = (float(y[t]) - state[0]) / 0.8
            log_probability += -0.5 * residual * residual - np.log(0.8) - 0.5 * np.log(2 * np.pi)
            log_weights[t, mode] = log_probability
    center = log_weights.max(axis=1, keepdims=True)
    weights = np.exp(log_weights - center)
    return weights / weights.sum(axis=1, keepdims=True), mode_means


def test_frozen_task_oracle_128_examples_two_lengths_and_prefix() -> None:
    matrices = frozen_matrices()
    assert matrices.shape == (4, 8, 8)
    assert len(MATRIX_ARRAY_SHA256) == 64
    short = make_batch(128, 16, ENGINEERING_ROOT)
    long = make_batch(128, 32, ENGINEERING_ROOT)
    assert short.pre_index == long.pre_index == PRE_OBSERVATIONS == 8
    assert short.x.shape == (128, 16, 2)
    assert long.x.shape == (128, 32, 2)
    assert short.pre_target.dtype == torch.float64
    assert short.posterior_history.shape == (128, 16, 4)
    for field in ("x", "controls", "observations", "posterior_history", "mode_means"):
        left = getattr(short, field)
        right = getattr(long, field)
        assert torch.equal(left, right[:, :16]), field
        assert left.numpy().tobytes() == right[:, :16].numpy().tobytes(), field
    assert torch.equal(short.truth, long.truth)
    assert torch.equal(short.pre_target, long.pre_target)
    assert torch.equal(short.final_target, long.posterior_history[:, 15])
    for batch in (short, long):
        assert batch.mode_means.shape == (128, batch.x.shape[1], 4)
        assert batch.mode_means.dtype == torch.float64
        for row in range(128):
            expected_posterior, expected_means = _independent_oracle(
                batch.controls[row].numpy(), batch.observations[row].numpy(), matrices)
            np.testing.assert_allclose(batch.posterior_history[row].numpy(),
                                       expected_posterior, atol=1e-6, rtol=0)
            np.testing.assert_allclose(batch.mode_means[row].numpy(),
                                       expected_means, atol=1e-12, rtol=0)
        torch.testing.assert_close(batch.pre_target, batch.posterior_history[:, 7])
        torch.testing.assert_close(batch.final_target, batch.posterior_history[:, -1])


def test_independent_substreams_replay_and_no_hidden_label_input() -> None:
    batch = make_batch(32, 16, ENGINEERING_ROOT)
    repeat = make_batch(32, 16, ENGINEERING_ROOT)
    other = make_batch(32, 16, ENGINEERING_ROOT + 1)
    assert torch.equal(batch.x, repeat.x)
    assert batch.x.numpy().tobytes() == repeat.x.numpy().tobytes()
    assert not torch.equal(batch.x, other.x)
    h_seed, u_seed, noise_seed = np.random.SeedSequence(ENGINEERING_ROOT).spawn(3)
    h = np.random.default_rng(h_seed).integers(0, 4, size=32)
    u = np.random.default_rng(u_seed).standard_normal((16, 32)).T
    noise = np.random.default_rng(noise_seed).standard_normal((16, 32)).T
    np.testing.assert_array_equal(batch.truth.numpy(), h)
    np.testing.assert_array_equal(batch.controls.numpy(), u)
    # Input channels are exactly the observed pair; truth is not a token/channel.
    np.testing.assert_array_equal(batch.x[:, :, 0].numpy(), u.astype(np.float32))
    np.testing.assert_array_equal(batch.x[:, :, 1].numpy(), batch.observations.numpy().astype(np.float32))
    assert batch.x.shape[-1] == 2  # mode_means and H are diagnostics, never model channels
    relabeled = replace(batch, truth=(batch.truth + 1) % 4)
    assert torch.equal(batch.x, relabeled.x)
    # Explicitly recover the third independent stream from the observed mixture.
    matrices = frozen_matrices()
    states = np.zeros((32, 4, 8), dtype=np.float64)
    for t in range(16):
        states = np.einsum("kij,nkj->nki", matrices, states, optimize=False)
        states[:, :, 0] += u[:, t, None]
        np.testing.assert_allclose(batch.mode_means[:, t].numpy(), states[:, :, 0], atol=1e-12, rtol=0)
        expected = states[np.arange(32), h, 0] + 0.8 * noise[:, t]
        np.testing.assert_allclose(batch.observations[:, t].numpy(), expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("examples,observations,root", [(0, 16, 1), (1, 8, 1),
                                                           (1, 24, 1), (1, 16, -1)])
def test_invalid_batch_arguments(examples: int, observations: int, root: int) -> None:
    with pytest.raises(ValueError):
        make_batch(examples, observations, root)
