"""EXP-0011's fixed four-mode, eight-dimensional filtering task.

The frozen matrices are data, never regenerated here. Independent SeedSequence
children supply the hidden mode, shared control, and observation noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


MODES = 4
LATENT_DIMENSION = 8
INPUT_SIZE = 2
PRE_OBSERVATIONS = 8
OBSERVATION_SIGMA = 0.8
MATRIX_JSON_SHA256 = "e2f118379f3dde5a9ca32ce05fcf9865eb1167c50abe3812a10437d4ae3e36ec"
MATRIX_ARRAY_SHA256 = "7282b80d022f546d2993b6cee86ca9570b70bde0aad49510b8eac8354996b1d7"


@dataclass(frozen=True)
class PfstyleBatch:
    x: Tensor
    pre_target: Tensor
    final_target: Tensor
    posterior_history: Tensor
    truth: Tensor
    controls: Tensor
    observations: Tensor
    mode_means: Tensor
    pre_index: int


@lru_cache(maxsize=4)
def _load_matrices(path: str) -> np.ndarray:
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != MATRIX_JSON_SHA256:
        raise ValueError("EXP-0011 frozen matrix JSON SHA-256 mismatch")
    payload = json.loads(raw)
    matrices = np.asarray(payload["matrices"], dtype="<f8", order="C")
    if matrices.shape != (MODES, LATENT_DIMENSION, LATENT_DIMENSION):
        raise ValueError("EXP-0011 frozen matrix shape mismatch")
    if sha256(matrices.tobytes(order="C")).hexdigest() != MATRIX_ARRAY_SHA256:
        raise ValueError("EXP-0011 frozen matrix array SHA-256 mismatch")
    matrices.setflags(write=False)
    return matrices


def frozen_matrices(matrix_path: Path = Path("configs/exp0011_matrices.json")) -> np.ndarray:
    """Read-only copy of the registered float64 transition matrices."""
    matrices = _load_matrices(str(matrix_path.resolve())).copy()
    matrices.setflags(write=False)
    return matrices


def _posterior(log_weights: np.ndarray) -> np.ndarray:
    normalizer = np.logaddexp.reduce(log_weights, axis=1, keepdims=True)
    return np.exp(log_weights - normalizer)


def make_batch(examples: int, observations: int, root: int,
               matrix_path: Path = Path("configs/exp0011_matrices.json")) -> PfstyleBatch:
    """Generate an unscreened IID16 or OOD32 stream and exact mode posterior.

    Length is time-major in the random draws, so same-root 16 and 32 streams
    share H and the first 16 controls/noise/observations byte-for-byte.
    """
    if examples < 1 or observations not in (16, 32) or root < 0:
        raise ValueError("examples >= 1, observations in {16, 32}, root >= 0 required")
    matrices = _load_matrices(str(matrix_path.resolve()))
    h_seed, u_seed, noise_seed = np.random.SeedSequence(root).spawn(3)
    truth = np.random.default_rng(h_seed).integers(0, MODES, size=examples)
    controls = np.random.default_rng(u_seed).standard_normal((observations, examples))
    noise = np.random.default_rng(noise_seed).standard_normal((observations, examples))

    states = np.zeros((examples, MODES, LATENT_DIMENSION), dtype=np.float64)
    log_weights = np.full((examples, MODES), -np.log(MODES), dtype=np.float64)
    observed = np.empty((observations, examples), dtype=np.float64)
    conditional_means = np.empty((observations, examples, MODES), dtype=np.float64)
    history = np.empty((observations, examples, MODES), dtype=np.float64)
    rows = np.arange(examples)
    for t in range(observations):
        states = np.einsum("kij,nkj->nki", matrices, states, optimize=False)
        states[:, :, 0] += controls[t, :, None]
        means = states[:, :, 0]
        conditional_means[t] = means
        observed[t] = means[rows, truth] + OBSERVATION_SIGMA * noise[t]
        residual = (observed[t, :, None] - means) / OBSERVATION_SIGMA
        log_weights += -0.5 * residual * residual - np.log(OBSERVATION_SIGMA) - 0.5 * np.log(2 * np.pi)
        history[t] = _posterior(log_weights)

    x = np.stack((controls, observed), axis=-1).transpose(1, 0, 2).astype(np.float32)
    posterior = history.transpose(1, 0, 2).copy()
    return PfstyleBatch(
        x=torch.from_numpy(x),
        pre_target=torch.from_numpy(posterior[:, PRE_OBSERVATIONS - 1].copy()),
        final_target=torch.from_numpy(posterior[:, -1].copy()),
        posterior_history=torch.from_numpy(posterior),
        truth=torch.from_numpy(truth.astype(np.int64)),
        controls=torch.from_numpy(controls.T.copy()),
        observations=torch.from_numpy(observed.T.copy()),
        mode_means=torch.from_numpy(conditional_means.transpose(1, 0, 2).copy()),
        pre_index=PRE_OBSERVATIONS,
    )
