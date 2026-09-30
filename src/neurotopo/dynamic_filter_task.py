"""Static-mode dynamical filtering task with an exact four-mode posterior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


MODES = 4
INPUT_SIZE = 4
EARLY_OBSERVATIONS = 4
THETA = (-0.85, -0.30, 0.30, 0.85)
DECAY = 0.9
CONTROL = (1.0, 0.5)
READOUT = (1.0, -0.3)
SIGMA = 0.8


@dataclass(frozen=True)
class DynamicFilterBatch:
    x: Tensor
    pre_joint: Tensor
    final_joint: Tensor
    pre_index: int
    truth: Tensor
    controls: Tensor
    observations: Tensor
    posterior_history: Tensor


def mode_matrices() -> np.ndarray:
    """Return the four fixed damped rotation matrices in float64."""
    matrices = []
    for theta in THETA:
        c, s = np.cos(theta), np.sin(theta)
        matrices.append(DECAY * np.array(((c, -s), (s, c)), dtype=np.float64))
    return np.stack(matrices)


def _posterior(log_weights: np.ndarray) -> np.ndarray:
    centered = log_weights - log_weights.max(axis=1, keepdims=True)
    weights = np.exp(centered)
    return weights / weights.sum(axis=1, keepdims=True)


def make_dynamic_filter_batch(examples: int, observations: int, seed: int) -> DynamicFilterBatch:
    """Generate paired-by-seed lengths using independent time-major RNG streams."""
    if examples < 1 or observations < EARLY_OBSERVATIONS or seed < 0:
        raise ValueError("examples >= 1, observations >= 4 and seed >= 0 required")
    mode_rng, control_rng, noise_rng = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(3)
    )
    truth = mode_rng.integers(0, MODES, size=examples)
    controls = control_rng.standard_normal((observations, examples))
    noise = noise_rng.standard_normal((observations, examples))

    matrices = mode_matrices()
    control_vector = np.asarray(CONTROL, dtype=np.float64)
    readout_vector = np.asarray(READOUT, dtype=np.float64)
    states = np.zeros((examples, MODES, 2), dtype=np.float64)
    log_weights = np.zeros((examples, MODES), dtype=np.float64)
    observed = np.empty((observations, examples), dtype=np.float64)
    history = np.empty((observations, examples, MODES), dtype=np.float64)
    rows = np.arange(examples)
    for t in range(observations):
        states = np.einsum("kij,nkj->nki", matrices, states) + controls[t, :, None, None] * control_vector
        means = np.einsum("nki,i->nk", states, readout_vector)
        observed[t] = means[rows, truth] + SIGMA * noise[t]
        log_weights -= 0.5 * ((observed[t, :, None] - means) / SIGMA) ** 2
        history[t] = _posterior(log_weights)

    pre_index = EARLY_OBSERVATIONS
    x = np.zeros((examples, observations + 2, INPUT_SIZE), dtype=np.float32)
    for t in range(observations):
        position = t if t < EARLY_OBSERVATIONS else t + 1
        x[:, position, 0] = controls[t]
        x[:, position, 1] = observed[t]
        x[:, position, 2] = 1.0
    x[:, pre_index, 3] = 1.0
    x[:, -1, 3] = 1.0
    return DynamicFilterBatch(
        x=torch.from_numpy(x),
        pre_joint=torch.from_numpy(history[EARLY_OBSERVATIONS - 1].astype(np.float32)),
        final_joint=torch.from_numpy(history[-1].astype(np.float32)),
        pre_index=pre_index,
        truth=torch.from_numpy(truth.astype(np.int64)),
        controls=torch.from_numpy(controls.T.copy()),
        observations=torch.from_numpy(observed.T.copy()),
        posterior_history=torch.from_numpy(history.transpose(1, 0, 2).copy()),
    )
