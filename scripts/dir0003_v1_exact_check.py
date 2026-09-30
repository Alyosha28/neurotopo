"""DIR-0003 v1 feasibility: exact untruncated capacity matching under the task input law.

Three fixes over v0 are implemented and checked here:
  1. the total memory capacity is an EXACT infinite-horizon sum (closed form over
     geometric series), so no horizon truncation is used anywhere;
  2. the matching and the prediction are computed under the TASK input law (AR(1) with a
     declared coefficient), not under an i.i.d. probe law;
  3. the manipulated factor is the unit ALLOCATION across three fixed timescale bands,
     with the total capacity held fixed, so the family is one-dimensional by construction
     and every by-product (mean timescale, spectral radius) is a disclosed consequence.

No training. Deterministic float64. Population quantities of frozen linear cores only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

SEED = 20260924
BANDS = ((0.20, 0.40), (0.60, 0.75), (0.90, 0.97))  # short / mid / long decay edges
GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48)


def band_decays(edges: tuple[float, float], count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0)
    return np.exp(np.linspace(np.log(edges[0]), np.log(edges[1]), count))


def allocation_decays(counts: tuple[int, int, int]) -> np.ndarray:
    return np.concatenate([band_decays(edge, count) for edge, count in zip(BANDS, counts)])


def input_weights(n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.choice([-1.0, 1.0], size=n) / np.sqrt(n)


def covariance(decays: np.ndarray, b: np.ndarray, phi: float) -> np.ndarray:
    outer = np.outer(decays, decays)
    base = np.outer(b, b) / (1.0 - outer)
    if phi == 0.0:
        return base
    return base * (1.0 - outer * phi**2) / np.outer(1.0 - decays * phi, 1.0 - decays * phi)


def response_terms(decays: np.ndarray, phi: float) -> tuple[np.ndarray, np.ndarray]:
    """g_i(k)/b_i = A_i(k) + B_i(k) with A_i(k) = (phi^k - lam^k)/(phi - lam), B_i(k) = lam^k/(1-lam*phi)."""
    if phi == 0.0:
        zeros = np.zeros_like(decays)
        return zeros, decays
    near = np.abs(decays - phi) < 1e-9
    safe = np.where(near, phi + 1e-3, decays)
    a_coef = 1.0 / (phi - safe)
    b_coef = 1.0 / (1.0 - safe * phi)
    return np.where(near, 0.0, a_coef), b_coef


def response_matrix(decays: np.ndarray, phi: float, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (phi^k - lam^k)/(phi-lam) and lam^k with the lambda -> phi limit handled."""
    lam_k = decays**k
    if phi == 0.0:
        return np.zeros_like(decays), lam_k
    phi_k = phi**k
    diff = phi - decays
    near = np.abs(diff) < 1e-12
    safe = np.where(near, phi + 1e-3, decays)
    first = (phi_k - safe**k) / (phi - safe)
    first = np.where(near, k * phi ** max(k - 1, 0), first)
    return first, lam_k


def cross_covariance(decays: np.ndarray, b: np.ndarray, phi: float, k: int) -> np.ndarray:
    first, lam_k = response_matrix(decays, phi, k)
    second = lam_k / (1.0 - decays * phi) if phi > 0 else np.zeros_like(decays)
    return b * (first + second)


def gram_exact(decays: np.ndarray, phi: float) -> np.ndarray:
    """Exact sum over k >= 1 of (g_i(k)/b_i)(g_j(k)/b_j) in closed form."""
    lam_i = decays[:, None]
    lam_j = decays[None, :]

    def geo(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x * y / (1.0 - x * y)

    if phi == 0.0:
        return geo(lam_i, lam_j)
    a_row, b_row = response_terms(decays, phi)
    a_i, b_i = a_row[:, None], b_row[:, None]
    a_j, b_j = a_row[None, :], b_row[None, :]
    aa = (geo(np.full_like(lam_i, phi), np.full_like(lam_j, phi))
          - geo(np.full_like(lam_i, phi), lam_j)
          - geo(lam_i, np.full_like(lam_j, phi))
          + geo(lam_i, lam_j))
    aa = aa * a_i * a_j
    ab = (geo(np.full_like(lam_i, phi), lam_j) - geo(lam_i, lam_j)) * a_i * b_j
    ba = (geo(lam_i, np.full_like(lam_j, phi)) - geo(lam_i, lam_j)) * b_i * a_j
    bb = geo(lam_i, lam_j) * b_i * b_j
    return aa + ab + ba + bb


def capacity_curve(decays: np.ndarray, b: np.ndarray, phi: float, ridge_rel: float,
                   k_max: int) -> np.ndarray:
    cov = covariance(decays, b, phi)
    cov = cov + ridge_rel * float(np.mean(np.diag(cov))) * np.eye(cov.shape[0])
    inverse = np.linalg.inv(cov)
    return np.array([float(cross_covariance(decays, b, phi, k) @ inverse
                           @ cross_covariance(decays, b, phi, k))
                     for k in range(1, k_max + 1)])


def total_exact(decays: np.ndarray, b: np.ndarray, phi: float, ridge_rel: float) -> float:
    n = decays.size
    cov = covariance(decays, b, phi)
    cov = cov + ridge_rel * float(np.mean(np.diag(cov))) * np.eye(n)
    gram = gram_exact(decays, phi) * np.outer(b, b)
    return float(np.sum(np.linalg.inv(cov) * gram.T))


def first_flip(diff: np.ndarray, grid: tuple[int, ...]) -> int | None:
    """First grid lag where the sign changes between ADJACENT GRID POINTS.

    diff is indexed by lag - 1, so grid points must be looked up explicitly; comparing
    index-adjacent entries instead would silently restrict the scan to lags 1..14.
    """
    values = [diff[k - 1] for k in grid]
    signs = [np.sign(v) for v in values]
    for i in range(len(grid) - 1):
        if signs[i] * signs[i + 1] < 0:
            return int(grid[i + 1])
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    n = args.units
    b = input_weights(n)

    # self-check 1: exact closed-form gram against direct summation
    probe = allocation_decays((8, 8, 8))
    gram_closed = gram_exact(probe, args.phi)
    direct = np.zeros_like(gram_closed)
    a_coef, b_coef = response_terms(probe, args.phi)
    for k in range(1, 20001):
        first, lam_k = response_matrix(probe, args.phi, k)
        second = lam_k / (1.0 - probe * args.phi)
        g = first + second
        direct += np.outer(g, g)
    gram_error = float(np.max(np.abs(gram_closed - direct)))

    # self-check 2: phi = 0 must reproduce the independent v0 implementation's h=4000 value
    wide = np.exp(np.linspace(np.log(0.99), np.log(0.99 ** (1 / 2**3)), n))
    total_zero_phi = total_exact(wide, b, 0.0, args.ridge)

    # ladder: allocations with a fixed unit total, matched on the EXACT task-law total
    reference = (n // 3, n // 3, n - 2 * (n // 3))
    reference_total = total_exact(allocation_decays(reference), b, args.phi, args.ridge)
    by_long: dict[int, tuple[tuple[int, int, int], float, float]] = {}
    for n1 in range(n + 1):
        for n2 in range(n - n1 + 1):
            n3 = n - n1 - n2
            counts = (n1, n2, n3)
            total = total_exact(allocation_decays(counts), b, args.phi, args.ridge)
            ratio = total / reference_total
            if abs(ratio - 1.0) > args.tolerance:
                continue
            best = by_long.get(n3)
            if best is None or abs(ratio - 1.0) < abs(best[2] - 1.0):
                by_long[n3] = (counts, total, ratio, len(allocation_decays(counts)))

    ladder = []
    curves = {}
    for n3 in sorted(by_long):
        counts, total, ratio, _ = by_long[n3]
        decays = allocation_decays(counts)
        curve = capacity_curve(decays, b, args.phi, args.ridge, 49)
        curves[n3] = curve
        ladder.append({"counts": list(counts), "total": total, "total_ratio": ratio,
                       "mean_decay": float(np.mean(decays)),
                       "spectral_radius": float(decays.max())})

    pairs = []
    order = sorted(curves)
    low_arm, high_arm = order[0], order[-1]
    for rung in ladder:
        n3 = rung["counts"][2]
        if n3 not in (low_arm, high_arm):
            continue
    if len(order) >= 2:
        diff = curves[order[0]] - curves[order[-1]]
        pairs.append({
            "pair": [list(by_long[order[0]][0]), list(by_long[order[-1]][0])],
            "delta_total_ratio": by_long[order[-1]][2] - by_long[order[0]][2],
            "first_flip_grid_lag": first_flip(diff, GRID),
            "gap_at_grid_lag_1": float(diff[0]),
            "gap_at_grid_lag_6": float(diff[5]),
            "gap_at_grid_lag_24": float(diff[23]),
            "max_gap_first_arm_k_le_6": float(np.max(diff[:6])),
            "max_gap_second_arm_k_ge_20": float(np.max(-diff[19:33])),
        })

    # ridge sensitivity of the predicted flip lag on the extreme pair
    ridge_rows = []
    if len(order) >= 2:
        a_decays = allocation_decays(by_long[order[0]][0])
        z_decays = allocation_decays(by_long[order[-1]][0])
        for ridge in (0.003, 0.01, 0.03):
            ca = capacity_curve(a_decays, b, args.phi, ridge, 49)
            cz = capacity_curve(z_decays, b, args.phi, ridge, 49)
            ridge_rows.append({"ridge_relative": ridge,
                               "first_flip_grid_lag": first_flip(ca - cz, GRID),
                               "total_ratio": total_exact(a_decays, b, args.phi, ridge)
                               / total_exact(z_decays, b, args.phi, ridge)})

    report = {
        "schema": "dir0003_v1_exact_check_v1",
        "generated_by": "scripts/dir0003_v1_exact_check.py",
        "config": {"units": n, "phi": args.phi, "ridge_relative": args.ridge,
                   "tolerance": args.tolerance, "bands": BANDS, "grid": GRID,
                   "bootstrap_seed": SEED},
        "self_checks": {
            "gram_closed_form_vs_direct_sum_max_abs_error": gram_error,
            "phi_zero_total_wide_band_should_match_v0_h4000_3_699": total_zero_phi,
        },
        "reference_allocation": list(reference),
        "reference_total": reference_total,
        "matched_rungs": ladder,
        "extreme_pair": pairs,
        "ridge_sensitivity": ridge_rows,
        "claims_not_made": ["no training", "population quantities of frozen linear cores only",
                            "matched rungs are a construction for a future design, not evidence"],
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "file_sha256": hashlib.sha256((payload + "\n").encode()).hexdigest(),
        "self_checks": report["self_checks"],
        "reference": {"counts": list(reference), "total": reference_total,
                      "phi": args.phi},
        "matched_rung_count": len(ladder),
        "rungs": [{"counts": r["counts"], "ratio": round(r["total_ratio"], 6),
                   "mean_decay": round(r["mean_decay"], 4)} for r in ladder][:12],
        "extreme_pair": pairs,
        "ridge_sensitivity": ridge_rows,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
