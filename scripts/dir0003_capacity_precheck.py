"""DIR-0003 analytical precheck: timescale organisation at matched total memory capacity.

No training, no scientific endpoint, CPU-only, deterministic float64.

Purpose (see research/design/dir0003_new_question_predesign_v0.md):
  * compute the exact population memory-capacity curve MC_k of a linear leaky reservoir,
    MC_k = g_k^T C^{-1} g_k  with  g_k = E[x_t u_{t-k}],  C = E[x_t x_t^T],
  * check that closed form against a direct finite-sample simulation (self-check),
  * build a *declared timescale ladder* core and a *homogeneous* core whose TOTAL
    memory capacity is matched to it by bisection on the homogeneous decay,
  * locate the predicted crossover lag k* where the two capacity curves intersect,
  * add a third rung with the ladder's spectrum but random eigenvectors (same spectrum,
    different organisation) to show that the spectrum alone does not fix the curve.

Outputs JSON only; nothing is trained and no hypothesis is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BOOTSTRAP_SEED = 20260924


def input_weights(n: int, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Fixed +-1/sqrt(n) input vector, shared by every arm."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=n) / np.sqrt(n)


def solve_covariance(cov: np.ndarray, g: np.ndarray) -> tuple[float, float, float]:
    """Return g^T C^{-1} g plus the condition number and the relative residual.

    Identical decays with identical input weights make C rank deficient (the units are
    redundant), so the pseudo-inverse is used and the rank deficiency is reported rather
    than silently ignored.
    """
    cond = float(np.linalg.cond(cov))
    if cond < 1e12:
        w = np.linalg.solve(cov, g)
    else:
        w = np.linalg.pinv(cov, rcond=1e-12) @ g
    value = float(g @ w)
    residual = float(np.linalg.norm(cov @ w - g) / max(np.linalg.norm(g), 1e-300))
    return value, cond, residual


def ridge_covariance(cov: np.ndarray, ridge_rel: float) -> np.ndarray:
    """Add a declared relative ridge, matching the ridge readout used by the experiment."""
    if ridge_rel <= 0.0:
        return cov
    return cov + ridge_rel * float(np.mean(np.diag(cov))) * np.eye(cov.shape[0])


def capacity_diagonal(decays: np.ndarray, b: np.ndarray, k_max: int,
                      diagnostics: dict[str, float] | None = None,
                      ridge_rel: float = 0.0) -> np.ndarray:
    """MC_k for x_t = diag(decays) x_{t-1} + b u_t, i.i.d. unit-variance u.

    ridge_rel > 0 gives the population optimum of a ridge readout; the experiment's readout
    is ridge, so the frozen prediction uses the same functional.
    """
    cov = ridge_covariance(np.outer(b, b) / (1.0 - np.outer(decays, decays)), ridge_rel)
    curve = np.empty(k_max)
    for k in range(1, k_max + 1):
        g = b * decays**k
        value, cond, residual = solve_covariance(cov, g)
        curve[k - 1] = value
        if diagnostics is not None:
            diagnostics["max_condition_number"] = max(diagnostics.get("max_condition_number", 0.0), cond)
            diagnostics["max_relative_residual"] = max(diagnostics.get("max_relative_residual", 0.0), residual)
    return curve


def capacity_coupled(w: np.ndarray, b: np.ndarray, k_max: int,
                     ridge_rel: float = 0.0) -> np.ndarray:
    """MC_k for x_t = W x_{t-1} + b u_t via the discrete Lyapunov equation."""
    n = w.shape[0]
    eye = np.eye(n * n)
    cov = np.linalg.solve(eye - np.kron(w, w), np.outer(b, b).reshape(-1)).reshape(n, n)
    cov = ridge_covariance(cov, ridge_rel)
    curve = np.empty(k_max)
    g = b.copy()
    for k in range(1, k_max + 1):
        g = w @ g
        curve[k - 1] = float(g @ np.linalg.solve(cov, g))
    return curve


def simulate_diagonal(decays: np.ndarray, b: np.ndarray, steps: int, k_max: int,
                      seed: int, ridge_rel: float, test_fraction: float = 0.5) -> np.ndarray:
    """Held-out ridge R^2 per lag, using the same functional as the frozen prediction.

    The readout is fitted on the first part of the stream and scored on the second, so the
    check tests generalisation of the same ridge functional rather than in-sample fit.
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(steps + k_max)
    x = np.zeros(decays.size)
    states = np.empty((steps, decays.size))
    for t in range(steps):
        x = decays * x + b * u[k_max + t]
        states[t] = x
    split = int(steps * (1.0 - test_fraction))
    curve = np.empty(k_max)
    for k in range(1, k_max + 1):
        target = u[k_max - k: k_max - k + steps]
        train_x, train_y = states[:split], target[:split]
        test_x, test_y = states[split:], target[split:]
        gram = train_x.T @ train_x
        gram = gram + ridge_rel * float(np.mean(np.diag(gram))) * np.eye(gram.shape[0])
        coef = np.linalg.solve(gram, train_x.T @ train_y)
        residual = test_y - test_x @ coef
        curve[k - 1] = 1.0 - float(residual @ residual) / float(test_y @ test_y)
    return curve


def log_uniform_band(low: float, high: float, n: int) -> np.ndarray:
    """n distinct decays, log-uniform on [low, high]; identical decays are redundant."""
    return np.exp(np.linspace(np.log(low), np.log(high), n))


def match_narrow_band(decays_wide: np.ndarray, b: np.ndarray, k_max: int, spread: float,
                      ridge_rel: float, lo: float = 0.3, hi: float = 1.0 - 1e-4,
                      tol: float = 1e-12,
                      scan_points: int = 12) -> tuple[float, float, list[dict[str, float]]]:
    """Bisect the narrow band centre so its total MC matches the wide band's total."""
    target = float(capacity_diagonal(decays_wide, b, k_max, ridge_rel=ridge_rel).sum())

    def total(centre: float) -> float:
        band = log_uniform_band(centre / spread, min(centre * spread, 1.0 - 1e-9), decays_wide.size)
        return float(capacity_diagonal(band, b, k_max, ridge_rel=ridge_rel).sum())

    scan = [{"centre": float(c), "total": total(float(c))}
            for c in np.linspace(lo, hi, scan_points)]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    centre = 0.5 * (lo + hi)
    return centre, total(centre) / target, scan


def crossover(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    """First interpolated lag where a - b changes sign, or nan when it does not."""
    diff = curve_a - curve_b
    sign_change = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0]
    if sign_change.size == 0:
        return float("nan")
    i = int(sign_change[0])
    d0, d1 = diff[i], diff[i + 1]
    return float((i + 1) + d0 / (d0 - d1))


def orthogonal(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--rho", type=float, default=0.99)
    parser.add_argument("--k-max", type=int, default=200)
    parser.add_argument("--sim-steps", type=int, default=400_000)
    parser.add_argument("--spread", type=float, default=1.05)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    n, blocks = args.units, args.blocks
    if n % blocks:
        parser.error("units must be divisible by blocks")
    b = input_weights(n)

    # declared wide timescale ladder: n distinct decays, log-uniform from rho to rho**(1/2**(blocks-1))
    decays_ladder = log_uniform_band(args.rho, args.rho ** (1.0 / 2.0 ** (blocks - 1)), n)
    diag_ladder: dict[str, float] = {}
    curve_ladder = capacity_diagonal(decays_ladder, b, args.k_max, diag_ladder, args.ridge)
    centre, total_ratio, centre_scan = match_narrow_band(
        decays_ladder, b, args.k_max, args.spread, args.ridge)
    decays_narrow = log_uniform_band(centre / args.spread, min(centre * args.spread, 1.0 - 1e-9), n)
    diag_narrow: dict[str, float] = {}
    curve_narrow = capacity_diagonal(decays_narrow, b, args.k_max, diag_narrow, args.ridge)

    # spread ladder: every rung is matched on TOTAL memory capacity to the wide band
    spread_ladder = []
    for spread in (1.02, 1.05, 1.15, 1.40, 2.00):
        c_rung, ratio_rung, _ = match_narrow_band(decays_ladder, b, args.k_max, spread, args.ridge)
        band_rung = log_uniform_band(c_rung / spread, min(c_rung * spread, 1.0 - 1e-9), n)
        curve_rung = capacity_diagonal(band_rung, b, args.k_max, ridge_rel=args.ridge)
        k_rung = crossover(curve_ladder, curve_rung)
        diff = curve_ladder - curve_rung
        spread_ladder.append({
            "spread": spread,
            "matched_centre_decay": c_rung,
            "total_ratio_to_wide": ratio_rung,
            "crossover_lag": k_rung,
            "max_gap_narrow_favoured_lag_le_10": float(np.max(-diff[:10])),
            "max_gap_wide_favoured_lag_ge_25": float(np.max(diff[24:100])),
            "timescale_first": float(1.0 / (1.0 - band_rung[0])),
            "timescale_last": float(1.0 / (1.0 - band_rung[-1])),
        })

    q = orthogonal(n, BOOTSTRAP_SEED + 1)
    w_rot = q @ np.diag(decays_ladder) @ q.T
    curve_rot = capacity_coupled(w_rot, b, args.k_max, args.ridge)

    # self-checks: closed form against finite-sample least squares
    single = 0.97
    closed_single = capacity_diagonal(np.array([single]), np.array([1.0 / np.sqrt(n)]), 40,
                                      ridge_rel=args.ridge)
    sim_single = simulate_diagonal(np.array([single]), np.array([1.0 / np.sqrt(n)]),
                                   args.sim_steps, 40, BOOTSTRAP_SEED + 2, args.ridge)
    sim_bank = simulate_diagonal(decays_ladder, b, args.sim_steps, 40, BOOTSTRAP_SEED + 3, args.ridge)
    analytic_single_identity = float(single ** 2 * (1.0 - single ** 80))

    def report(curve: np.ndarray) -> dict[str, float]:
        k_star = crossover(curve_ladder, curve_narrow)
        index = int(round(k_star)) if np.isfinite(k_star) else 50
        index = min(max(index, 1), curve.size)
        return {
            "total": float(curve.sum()),
            "mc_lag1": float(curve[0]),
            "mc_at_predicted_crossover": float(curve[index - 1]),
            "mc_lag50": float(curve[49]),
            "mc_lag200": float(curve[199]),
        }

    k_wide_narrow = crossover(curve_ladder, curve_narrow)
    k_wide_rot = crossover(curve_ladder, curve_rot)
    result = {
        "schema": "dir0003_capacity_precheck_v1",
        "generated_by": "scripts/dir0003_capacity_precheck.py",
        "config": {
            "units": n, "blocks": blocks, "rho": args.rho, "k_max": args.k_max,
            "sim_steps": args.sim_steps, "bootstrap_seed": BOOTSTRAP_SEED,
            "narrow_band_spread": args.spread, "ridge_relative": args.ridge,
        },
        "declared_wide_band": {
            "decay_first": float(decays_ladder[0]), "decay_last": float(decays_ladder[-1]),
            "timescale_first": float(1.0 / (1.0 - decays_ladder[0])),
            "timescale_last": float(1.0 / (1.0 - decays_ladder[-1])),
        },
        "matched_narrow_band": {
            "centre_decay": centre,
            "decay_first": float(decays_narrow[0]), "decay_last": float(decays_narrow[-1]),
            "timescale_first": float(1.0 / (1.0 - decays_narrow[0])),
            "timescale_last": float(1.0 / (1.0 - decays_narrow[-1])),
            "total_ratio_to_wide": total_ratio,
        },
        "spectral_radius": {"wide_band": float(decays_ladder.max()),
                            "narrow_band_matched": float(decays_narrow.max()),
                            "rotated_same_spectrum": float(np.abs(np.linalg.eigvals(w_rot)).max())},
        "predicted_crossover_lag": {
            "wide_vs_narrow": k_wide_narrow,
            "wide_vs_rotated": k_wide_rot,
        },
        "capacity_summary": {
            "wide_band": report(curve_ladder),
            "narrow_band_matched": report(curve_narrow),
            "rotated_same_spectrum": report(curve_rot),
        },
        "narrow_band_centre_scan": centre_scan,
        "spread_ladder": spread_ladder,
        "numerical_diagnostics": {
            "wide_band": diag_ladder,
            "narrow_band_matched": diag_narrow,
        },
        "self_checks": {
            "single_mode_algebraic_identity": analytic_single_identity,
            "single_mode_closed_form_total": float(closed_single.sum()),
            "single_mode_simulated_total": float(sim_single.sum()),
            "single_mode_max_abs_error": float(np.max(np.abs(closed_single - sim_single))),
            "bank_max_abs_error_lag1_40": float(np.max(np.abs(curve_ladder[:40] - sim_bank))),
        },
        "curves": {
            "wide_band": curve_ladder.tolist(),
            "narrow_band_matched": curve_narrow.tolist(),
            "rotated_same_spectrum": curve_rot.tolist(),
        },
        "claims_not_made": [
            "no task was trained and no architecture comparison was run",
            "these numbers are population quantities of a frozen linear core, not capability evidence",
            "total-capacity matching here is a construction for a future design, not a Gate 1 result",
        ],
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "total_ratio": total_ratio,
        "matched_narrow_centre_decay": centre,
        "crossover_wide_vs_narrow": k_wide_narrow,
        "crossover_wide_vs_rotated": k_wide_rot,
        "totals": {k: v["total"] for k, v in result["capacity_summary"].items()},
        "self_checks": result["self_checks"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
