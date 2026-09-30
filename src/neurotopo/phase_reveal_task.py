"""Frozen EXP-0013/0014 phase-reveal generator and exact Gaussian oracles.

All times in this module are one-based; ``t`` is the number of observed frames.
Only ``x`` is intended for a model. The remaining batch fields are sealed data
and analytic targets for evaluation and independent verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


MATRIX_SHA256 = "8b5ed7d51b6178fa5469e9d0b54e25f502eb267017f6ba41db404f37d59f1b6c"
GAMMA = 0.13610245059775603
NOISE_STD = 0.8
STREAMS = 4
EARLY_QUERY = 8
TRUNCATION_STEP = 16


@dataclass(frozen=True)
class PhaseRevealBatch:
    x: NDArray[np.float32]
    oracle_pre: NDArray[np.float64]
    oracle_final: NDArray[np.float64]
    H: NDArray[np.int64]
    cue: NDArray[np.int64]
    u: NDArray[np.float64]
    eps: NDArray[np.float64]
    y: NDArray[np.float64]
    switch_step: int
    cue_reliability: float


def _load_matrices(path: str) -> tuple[NDArray[np.float64], float]:
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != MATRIX_SHA256:
        raise ValueError("phase-reveal matrix file differs from the frozen SHA-256")
    data = json.loads(raw)
    matrices = np.asarray(data["matrices"], dtype=np.float64)
    gamma = float(data["gamma"])
    if matrices.shape != (2, 6, 6) or not np.isfinite(matrices).all() or gamma != GAMMA:
        raise ValueError("invalid frozen phase-reveal matrices or gamma")
    matrices.setflags(write=False)
    return matrices, gamma


def _inputs(
    u: NDArray[np.float64], y: NDArray[np.float64], cue: NDArray[np.int64],
    t: int | None, switch_step: int, cue_reliability: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], int]:
    u = np.asarray(u, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cue = np.asarray(cue)
    if u.ndim != 3 or u.shape[2] != STREAMS or y.shape != u.shape:
        raise ValueError("u must be [B,T,4] and y [B,T,4]")
    if cue.shape != (u.shape[0],) or cue.dtype.kind not in "iu" or np.any(cue < 0) or np.any(cue >= STREAMS):
        raise ValueError("cue must contain one stream index 0..3 per example")
    if u.shape[0] < 1 or u.shape[1] < 1 or not np.isfinite(u).all() or not np.isfinite(y).all():
        raise ValueError("nonempty finite u/y required")
    if not isinstance(switch_step, int) or not 1 <= switch_step < u.shape[1]:
        raise ValueError("switch_step must lie within the sequence")
    if not np.isfinite(cue_reliability) or not 0 < cue_reliability < 1:
        raise ValueError("cue_reliability must lie strictly between zero and one")
    stop = u.shape[1] if t is None else t
    if not isinstance(stop, int) or not 1 <= stop <= u.shape[1]:
        raise ValueError("t must be a one-based prefix length")
    return u, y, cue.astype(np.int64, copy=False), stop


def _prior(cue: NDArray[np.int64], reliability: float) -> NDArray[np.float64]:
    prior = np.full((cue.size, STREAMS), (1.0 - reliability) / 3.0, dtype=np.float64)
    prior[np.arange(cue.size), cue] = reliability
    return prior


def _softmax(log_weights: NDArray[np.float64]) -> NDArray[np.float64]:
    if not np.isfinite(log_weights).all():
        raise FloatingPointError("non-finite phase-reveal log likelihood")
    shifted = log_weights - np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(shifted)
    probabilities = weights / np.sum(weights, axis=1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("non-finite phase-reveal posterior")
    return probabilities


def cue_only_posterior(
    cue: NDArray[np.int64], cue_reliability: float = 0.75,
) -> NDArray[np.float64]:
    """Posterior from the discrete cue alone."""
    cue = np.asarray(cue)
    if cue.ndim != 1 or cue.dtype.kind not in "iu" or np.any(cue < 0) or np.any(cue >= 4):
        raise ValueError("cue must be a one-dimensional array of indices 0..3")
    if not np.isfinite(cue_reliability) or not 0 < cue_reliability < 1:
        raise ValueError("invalid cue reliability")
    return _prior(cue.astype(np.int64), cue_reliability)


def _means(u: NDArray[np.float64], matrices: NDArray[np.float64], gamma: float) -> NDArray[np.float64]:
    states = np.zeros((2, u.shape[0], STREAMS, 6), dtype=np.float64)
    means = np.empty((2, u.shape[0], u.shape[1], STREAMS), dtype=np.float64)
    for s in range(u.shape[1]):
        states = np.einsum("aij,abkj->abki", matrices, states) + u[:, s, :][None, :, :, None] * np.eye(6)[0]
        means[:, :, s, :] = gamma * states[..., 0]
    return means


def full_posterior(
    u: NDArray[np.float64], y: NDArray[np.float64], cue: NDArray[np.int64],
    matrix_path: str | Path, *, t: int | None = None, switch_step: int = 16,
    cue_reliability: float = 0.75,
) -> NDArray[np.float64]:
    """Exact four-H posterior conditional on all observed input history."""
    u, y, cue, stop = _inputs(u, y, cue, t, switch_step, cue_reliability)
    prior = _prior(cue, cue_reliability)
    if stop <= switch_step:
        return prior
    matrices, gamma = _load_matrices(str(matrix_path))
    means = _means(u[:, :stop], matrices, gamma)
    residual = y[:, switch_step:stop][None] - means[:, :, switch_step:stop]
    per_type = np.sum(residual * residual, axis=2) / (2.0 * NOISE_STD**2)
    return _softmax(np.log(prior) + per_type[1] - per_type[0])


def last_frame_posterior(
    u: NDArray[np.float64], y: NDArray[np.float64], cue: NDArray[np.int64],
    matrix_path: str | Path, *, t: int | None = None, switch_step: int = 16,
    cue_reliability: float = 0.75,
) -> NDArray[np.float64]:
    """Exact posterior using only cue and the last frame's u/y (earlier u marginalized)."""
    u, y, cue, stop = _inputs(u, y, cue, t, switch_step, cue_reliability)
    prior = _prior(cue, cue_reliability)
    if stop <= switch_step:
        return prior
    matrices, gamma = _load_matrices(str(matrix_path))
    variance = np.full(2, NOISE_STD**2, dtype=np.float64)
    for a in range(2):
        power = np.eye(6, dtype=np.float64)
        for _ in range(1, stop):
            power = power @ matrices[a]
            variance[a] += gamma**2 * power[0, 0] ** 2
    residual = y[:, stop - 1, :] - gamma * u[:, stop - 1, :]
    log_density = -0.5 * (np.log(variance)[:, None, None] + residual[None] ** 2 / variance[:, None, None])
    return _softmax(np.log(prior) + log_density[0] - log_density[1])


def _truncated_parameters(
    u: NDArray[np.float64], matrix: NDArray[np.float64], gamma: float,
    stop: int, switch_step: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean and Cholesky factor after integrating the first 16 controls."""
    # Cov(z_16) follows directly from the 16 independent standard-normal inputs.
    e1 = np.eye(6, dtype=np.float64)[0]
    covariance_z = np.zeros((6, 6), dtype=np.float64)
    power = np.eye(6, dtype=np.float64)
    for _ in range(TRUNCATION_STEP):
        direction = power @ e1
        covariance_z += np.outer(direction, direction)
        power = matrix @ power
    state = np.zeros((u.shape[0], STREAMS, 6), dtype=np.float64)
    state_rows = []
    predicted = []
    power = np.eye(6, dtype=np.float64)
    for s in range(TRUNCATION_STEP, stop):
        state = state @ matrix.T + u[:, s, :, None] * e1
        power = power @ matrix
        if s + 1 > switch_step:
            predicted.append(gamma * state[..., 0])
            state_rows.append(power[0].copy())
    mean = np.stack(predicted, axis=1)
    L = np.stack(state_rows, axis=0)
    cov = NOISE_STD**2 * np.eye(L.shape[0]) + gamma**2 * L @ covariance_z @ L.T
    return mean, np.linalg.cholesky(cov)


def truncated_prehistory_posterior(
    u: NDArray[np.float64], y: NDArray[np.float64], cue: NDArray[np.int64],
    matrix_path: str | Path, *, t: int | None = None, switch_step: int = 16,
    cue_reliability: float = 0.75,
) -> NDArray[np.float64]:
    """Exact posterior after marginalizing u1:16; retain later u/y and cue."""
    u, y, cue, stop = _inputs(u, y, cue, t, switch_step, cue_reliability)
    if switch_step < TRUNCATION_STEP:
        raise ValueError("strict u1:16 marginalization requires switch_step >= 16")
    prior = _prior(cue, cue_reliability)
    if stop <= switch_step:
        return prior
    matrices, gamma = _load_matrices(str(matrix_path))
    log_density = []
    for a in range(2):
        mean, chol = _truncated_parameters(u, matrices[a], gamma, stop, switch_step)
        residual = (y[:, switch_step:stop, :] - mean).transpose(1, 0, 2).reshape(stop - switch_step, -1)
        solved = np.linalg.solve(chol, residual)
        quadratic = np.sum(solved * solved, axis=0).reshape(u.shape[0], STREAMS)
        log_density.append(-0.5 * quadratic - np.log(np.diag(chol)).sum())
    return _softmax(np.log(prior) + log_density[0] - log_density[1])


def make_batch(
    batch_size: int, length: int, root: int, matrix_path: str | Path, *,
    cue_reliability: float = 0.75, switch_step: int = 16,
) -> PhaseRevealBatch:
    """Generate one frozen four-stream batch using H/cue/U/eps child RNGs in order."""
    if not isinstance(batch_size, int) or batch_size < 1 or not isinstance(length, int) or length <= switch_step:
        raise ValueError("batch_size >= 1 and length > switch_step required")
    if not isinstance(root, int) or root < 0:
        raise ValueError("root must be a nonnegative integer")
    if not isinstance(switch_step, int) or switch_step < TRUNCATION_STEP:
        raise ValueError("switch_step >= 16 required")
    if not np.isfinite(cue_reliability) or not 0 < cue_reliability < 1:
        raise ValueError("invalid cue reliability")
    matrices, gamma = _load_matrices(str(matrix_path))
    h_seed, cue_seed, u_seed, eps_seed = np.random.SeedSequence(root).spawn(4)
    h_rng, cue_rng, u_rng, eps_rng = (np.random.Generator(np.random.PCG64(seed)) for seed in (h_seed, cue_seed, u_seed, eps_seed))
    H = h_rng.integers(0, STREAMS, size=batch_size, dtype=np.int64)
    correct = cue_rng.uniform(size=batch_size) < cue_reliability
    wrong_index = cue_rng.integers(0, STREAMS - 1, size=batch_size)
    cue = np.where(correct, H, (H + 1 + wrong_index) % STREAMS).astype(np.int64)
    u = u_rng.standard_normal((batch_size, length, STREAMS))
    eps = eps_rng.standard_normal((batch_size, length, STREAMS))
    means = _means(u, matrices, gamma)
    y = NOISE_STD * eps
    active = np.arange(length) >= switch_step
    target_means = np.where(H[:, None, None] == np.arange(STREAMS)[None, None, :], means[0], means[1])
    y[:, active, :] += target_means[:, active, :]
    x = np.empty((batch_size, length, 16), dtype=np.float32)
    x[:, :, 0::4] = u.astype(np.float32)
    x[:, :, 1::4] = y.astype(np.float32)
    x[:, :, 2::4] = (cue[:, None, None] == np.arange(STREAMS)[None, None, :]).astype(np.float32)
    x[:, :, 3::4] = active[None, :, None].astype(np.float32)
    pre = full_posterior(u, y, cue, matrix_path, t=EARLY_QUERY, switch_step=switch_step, cue_reliability=cue_reliability)
    final = full_posterior(u, y, cue, matrix_path, switch_step=switch_step, cue_reliability=cue_reliability)
    return PhaseRevealBatch(x, pre, final, H, cue, u, eps, y, switch_step, cue_reliability)


def validate_oracles(
    batch: PhaseRevealBatch, matrix_path: str | Path, *, t: int | None = None,
    atol: float = 1e-10,
) -> dict[str, float | str]:
    """Independent four-H enumeration and explicit covariance check on a small batch.

    This is an engineering entry point, not a task-information or control result.
    The explicit covariance reference has cubic cost in the selected prefix.
    """
    if batch.u.shape[0] > 64:
        raise ValueError("independent oracle validation is limited to 64 examples")
    stop = batch.u.shape[1] if t is None else t
    u, y, cue, stop = _inputs(batch.u, batch.y, batch.cue, stop, batch.switch_step, batch.cue_reliability)
    matrices, gamma = _load_matrices(str(matrix_path))
    prior = cue_only_posterior(cue, batch.cue_reliability)
    # Reference full likelihood: independently enumerate every H and every stream.
    ref_full = np.empty_like(prior)
    for b in range(u.shape[0]):
        costs = np.zeros(4)
        for h in range(4):
            for k in range(4):
                state = np.zeros(6)
                for s in range(stop):
                    state = matrices[0 if h == k else 1] @ state + np.eye(6)[0] * u[b, s, k]
                    if s + 1 > batch.switch_step:
                        costs[h] += (y[b, s, k] - gamma * state[0]) ** 2 / (2 * NOISE_STD**2)
        ref_full[b] = _softmax((np.log(prior[b]) - costs)[None])[0]
    got_full = full_posterior(u, y, cue, matrix_path, t=stop, switch_step=batch.switch_step, cue_reliability=batch.cue_reliability)
    errors = {"full": float(np.max(np.abs(ref_full - got_full)))}
    output_arrays = {"full": got_full}
    if stop <= batch.switch_step:
        ref_last = prior
    else:
        # Compute the last-frame variance directly from every omitted input's
        # coefficient, without calling the production variance loop.
        variances = []
        for matrix in matrices:
            coefficients = np.array([np.linalg.matrix_power(matrix, stop - 1 - q)[0, 0] for q in range(stop - 1)])
            variances.append(NOISE_STD**2 + gamma**2 * np.dot(coefficients, coefficients))
        last_residual = y[:, stop - 1] - gamma * u[:, stop - 1]
        ref_last = _softmax(np.log(prior) - 0.5 * (np.log(variances[0]) + last_residual**2 / variances[0])
                            + 0.5 * (np.log(variances[1]) + last_residual**2 / variances[1]))
    got_last = last_frame_posterior(u, y, cue, matrix_path, t=stop, switch_step=batch.switch_step, cue_reliability=batch.cue_reliability)
    errors["last_frame"] = float(np.max(np.abs(ref_last - got_last)))
    output_arrays["last_frame"] = got_last
    if stop > batch.switch_step:
        # Separate reference: build each retained y's coefficients on all early u,
        # then marginalize them through C C^T without the production P/L formula.
        likelihoods = []
        for a in range(2):
            matrix = matrices[a]
            rows = []
            mean = np.empty((u.shape[0], stop - batch.switch_step, 4))
            for i, s in enumerate(range(batch.switch_step, stop)):
                powers = [np.linalg.matrix_power(matrix, s - q)[0, 0] for q in range(s + 1)]
                rows.append(np.asarray(powers[:TRUNCATION_STEP]))
                mean[:, i] = gamma * np.einsum("q,bqk->bk", powers[TRUNCATION_STEP:], u[:, TRUNCATION_STEP:s + 1])
            C = gamma * np.stack(rows)
            covariance = NOISE_STD**2 * np.eye(len(rows)) + C @ C.T
            explicit_chol = np.linalg.cholesky(covariance)
            logdet = 2.0 * np.log(np.diag(explicit_chol)).sum()
            residual = (y[:, batch.switch_step:stop] - mean).transpose(1, 0, 2).reshape(len(rows), -1)
            solved = np.linalg.solve(covariance, residual)
            likelihoods.append((-0.5 * (np.sum(residual * solved, axis=0) + logdet)).reshape(u.shape[0], 4))
        ref_trunc = _softmax(np.log(prior) + likelihoods[0] - likelihoods[1])
        got_trunc = truncated_prehistory_posterior(u, y, cue, matrix_path, t=stop, switch_step=batch.switch_step, cue_reliability=batch.cue_reliability)
        errors["truncated"] = float(np.max(np.abs(ref_trunc - got_trunc)))
        output_arrays["truncated"] = got_trunc
    if any(not np.isfinite(error) or error > atol for error in errors.values()):
        raise AssertionError(f"phase-reveal oracle independent validation failed: {errors}")
    source_hasher = sha256()
    for array in (u, y, cue, batch.H, batch.eps[:, :stop], batch.x[:, :stop]):
        source_hasher.update(np.asarray(array.shape, dtype="<i8").tobytes())
        source_hasher.update(np.ascontiguousarray(array).tobytes())
    source_hasher.update(np.asarray([batch.switch_step], dtype="<i8").tobytes())
    source_hasher.update(np.asarray([batch.cue_reliability], dtype="<f8").tobytes())
    report: dict[str, float | str] = dict(errors)
    report["source_sha256"] = source_hasher.hexdigest()
    report["matrix_sha256"] = MATRIX_SHA256
    for name, array in output_arrays.items():
        report[f"{name}_sha256"] = sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()
    return report
