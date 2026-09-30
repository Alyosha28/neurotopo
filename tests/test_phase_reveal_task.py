"""Engineering-only checks for the frozen phase-reveal generator and oracles."""

import json
from pathlib import Path

import numpy as np
import pytest

from neurotopo.phase_reveal_task import (
    GAMMA, NOISE_STD, cue_only_posterior, full_posterior,
    last_frame_posterior, make_batch, truncated_prehistory_posterior,
    validate_oracles,
)


MATRICES = Path(__file__).resolve().parents[1] / "configs" / "exp0013_matrices.json"


@pytest.mark.parametrize("switch,reliability", [(16, 0.75), (20, 0.60)])
def test_rng_contract_phase_features_and_early_cue(switch: int, reliability: float) -> None:
    batch = make_batch(5, 48, 42, MATRICES, cue_reliability=reliability, switch_step=switch)
    children = np.random.SeedSequence(42).spawn(4)
    h_rng, c_rng, u_rng, e_rng = (np.random.Generator(np.random.PCG64(child)) for child in children)
    H = h_rng.integers(0, 4, size=5, dtype=np.int64)
    correct = c_rng.uniform(size=5) < reliability
    wrong = c_rng.integers(0, 3, size=5)
    cue = np.where(correct, H, (H + 1 + wrong) % 4)
    np.testing.assert_array_equal(batch.H, H)
    np.testing.assert_array_equal(batch.cue, cue)
    np.testing.assert_array_equal(batch.u, u_rng.standard_normal((5, 48, 4)))
    np.testing.assert_array_equal(batch.eps, e_rng.standard_normal((5, 48, 4)))
    np.testing.assert_allclose(batch.y[:, :switch], NOISE_STD * batch.eps[:, :switch], rtol=0, atol=0)
    np.testing.assert_array_equal(batch.x[:, :, 0::4], batch.u.astype(np.float32))
    np.testing.assert_array_equal(batch.x[:, :, 1::4], batch.y.astype(np.float32))
    np.testing.assert_array_equal(batch.x[:, :, 2::4], np.broadcast_to((cue[:, None, None] == np.arange(4)).astype(np.float32), (5, 48, 4)))
    np.testing.assert_array_equal(batch.x[:, :, 3::4], np.broadcast_to((np.arange(48) >= switch)[None, :, None], (5, 48, 4)))
    assert batch.x.dtype == np.float32
    assert batch.oracle_pre.dtype == np.float64
    np.testing.assert_array_equal(batch.oracle_pre, cue_only_posterior(cue, reliability))
    np.testing.assert_array_equal(full_posterior(batch.u, batch.y, cue, MATRICES, t=switch, switch_step=switch, cue_reliability=reliability), batch.oracle_pre)
    assert batch.x.shape == (5, 48, 16)
    # The model-facing tensor has no hidden H, matrix, gamma, or oracle channel.
    assert batch.x.shape[-1] == 4 * 4
    repeat = make_batch(5, 48, 42, MATRICES, cue_reliability=reliability, switch_step=switch)
    for field in ("x", "oracle_pre", "oracle_final", "H", "cue", "u", "eps", "y"):
        np.testing.assert_array_equal(getattr(batch, field), getattr(repeat, field))
    changed_u, changed_y = batch.u.copy(), batch.y.copy()
    changed_u[:, 8:] += 100
    changed_y[:, 8:] += 100
    np.testing.assert_array_equal(
        full_posterior(changed_u, changed_y, cue, MATRICES, t=8, switch_step=switch, cue_reliability=reliability),
        batch.oracle_pre,
    )
    np.testing.assert_array_equal(
        full_posterior(changed_u, changed_y, cue, MATRICES, t=8, switch_step=switch, cue_reliability=reliability),
        cue_only_posterior(cue, reliability),
    )
    changed_u, changed_y = batch.u.copy(), batch.y.copy()
    changed_u[:, 24:] += 100
    changed_y[:, 24:] += 100
    np.testing.assert_array_equal(
        full_posterior(changed_u, changed_y, cue, MATRICES, t=24, switch_step=switch, cue_reliability=reliability),
        full_posterior(batch.u, batch.y, cue, MATRICES, t=24, switch_step=switch, cue_reliability=reliability),
    )


def test_independent_four_h_enumeration_and_separate_validation() -> None:
    batch = make_batch(3, 24, 42, MATRICES)
    matrix_data = json.loads(MATRICES.read_text())
    matrices = np.asarray(matrix_data["matrices"])
    for b in range(3):
        for k in range(4):
            state = np.zeros(6)
            for s in range(24):
                state = matrices[0 if k == batch.H[b] else 1] @ state
                state[0] += batch.u[b, s, k]
                expected_y = NOISE_STD * batch.eps[b, s, k]
                if s >= 16:
                    expected_y += GAMMA * state[0]
                np.testing.assert_allclose(batch.y[b, s, k], expected_y, atol=1e-12)
    for stop in (8, 17, 24):
        expected = []
        for b in range(3):
            weights = []
            for h in range(4):
                loglik = np.log(0.75 if h == batch.cue[b] else 0.25 / 3)
                for k in range(4):
                    state = np.zeros(6)
                    for s in range(stop):
                        state = matrices[0 if k == h else 1] @ state
                        state[0] += batch.u[b, s, k]
                        if s >= 16:
                            loglik -= (batch.y[b, s, k] - GAMMA * state[0]) ** 2 / (2 * NOISE_STD**2)
                weights.append(loglik)
            weights = np.exp(np.asarray(weights) - max(weights))
            expected.append(weights / weights.sum())
        actual = full_posterior(batch.u, batch.y, batch.cue, MATRICES, t=stop)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
        report = validate_oracles(batch, MATRICES, t=stop)
        assert max(report[name] for name in ("full", "last_frame") if name in report) <= 1e-10
        assert len(report["source_sha256"]) == 64
        assert len(report["full_sha256"]) == 64
    np.testing.assert_allclose(batch.oracle_final, expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize("switch", [16, 20])
def test_truncated_covariance_against_direct_early_input_marginal(switch: int) -> None:
    batch = make_batch(2, 24, 42, MATRICES, switch_step=switch)
    matrices = np.asarray(json.loads(MATRICES.read_text())["matrices"])
    prior = cue_only_posterior(batch.cue)
    log_densities = []
    for matrix in matrices:
        # Independently construct the observation-by-early-input loading matrix.
        C = np.zeros((24 - switch, 16))
        mean = np.zeros((2, 24 - switch, 4))
        for row, s in enumerate(range(switch, 24)):
            for q in range(16):
                C[row, q] = GAMMA * np.linalg.matrix_power(matrix, s - q)[0, 0]
            for q in range(16, s + 1):
                mean[:, row] += GAMMA * np.linalg.matrix_power(matrix, s - q)[0, 0] * batch.u[:, q]
        cov = C @ C.T + NOISE_STD**2 * np.eye(24 - switch)
        sign, logdet = np.linalg.slogdet(cov)
        assert sign == 1
        residual = (batch.y[:, switch:] - mean).transpose(1, 0, 2).reshape(24 - switch, -1)
        value = -0.5 * (np.sum(residual * np.linalg.solve(cov, residual), axis=0) + logdet)
        log_densities.append(value.reshape(2, 4))
    weights = np.log(prior) + log_densities[0] - log_densities[1]
    weights = np.exp(weights - weights.max(axis=1, keepdims=True))
    expected = weights / weights.sum(axis=1, keepdims=True)
    actual = truncated_prehistory_posterior(batch.u, batch.y, batch.cue, MATRICES, switch_step=switch)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    assert validate_oracles(batch, MATRICES)["truncated"] <= 1e-10
    # Early u is marginalized: this oracle is invariant to the supplied u1:16.
    changed = batch.u.copy()
    changed[:, :16] += 1000
    np.testing.assert_allclose(truncated_prehistory_posterior(changed, batch.y, batch.cue, MATRICES, switch_step=switch), actual, atol=1e-12)


def test_last_frame_true_marginal_and_history_invariance() -> None:
    batch = make_batch(2, 24, 42, MATRICES)
    matrices = np.asarray(json.loads(MATRICES.read_text())["matrices"])
    variances = []
    for matrix in matrices:
        coeffs = [np.linalg.matrix_power(matrix, lag)[0, 0] for lag in range(1, 24)]
        variances.append(NOISE_STD**2 + GAMMA**2 * np.sum(np.square(coeffs)))
    residual = batch.y[:, -1] - GAMMA * batch.u[:, -1]
    logp = np.log(cue_only_posterior(batch.cue))
    for a, sign in ((0, 1), (1, -1)):
        logp += sign * -0.5 * (np.log(variances[a]) + residual**2 / variances[a])
    expected = np.exp(logp - logp.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    actual = last_frame_posterior(batch.u, batch.y, batch.cue, MATRICES)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    u, y = batch.u.copy(), batch.y.copy()
    u[:, :-1] += 100
    y[:, :-1] -= 100
    np.testing.assert_array_equal(last_frame_posterior(u, y, batch.cue, MATRICES), actual)


@pytest.mark.parametrize("switch,reliability", [(16, 0.75), (20, 0.60)])
def test_full_length_independent_oracles_and_phase_boundary(switch: int, reliability: float) -> None:
    batch = make_batch(3, 48, 42, MATRICES,
                       cue_reliability=reliability, switch_step=switch)
    check = validate_oracles(batch, MATRICES)
    assert max(check[name] for name in ("full", "last_frame", "truncated")) < 1e-10
    assert np.all(batch.x[:, switch - 1, 3::4] == 0)
    assert np.all(batch.x[:, switch, 3::4] == 1)


def test_input_and_nonfinite_guards(tmp_path: Path) -> None:
    batch = make_batch(2, 24, 42, MATRICES)
    (tmp_path / "matrix.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen SHA"):
        make_batch(2, 24, 42, tmp_path / "matrix.json")
    for oracle in (full_posterior, last_frame_posterior, truncated_prehistory_posterior):
        broken = batch.y.copy()
        broken[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            oracle(batch.u, broken, batch.cue, MATRICES)
    with pytest.raises(ValueError, match="switch_step"):
        make_batch(2, 24, 42, MATRICES, switch_step=15)
    with pytest.raises(ValueError, match="cue"):
        cue_only_posterior(np.array([4]))
