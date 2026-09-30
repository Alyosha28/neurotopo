"""DIR-0003 AR(1) transfer check: does probe-distribution capacity matching survive?

The frozen precheck (`dir0003_capacity_precheck.json`) matches TOTAL memory capacity
under an i.i.d. input, while the pre-design's tasks use an AR(1) input. This script
computes the same analytic capacity functional under AR(1) input and reports whether
the matching and the predicted crossover survive the distribution change.

No training, deterministic, CPU-only. Output is a separate JSON; the frozen packet's
precheck file is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 20260924


def log_uniform_band(low: float, high: float, n: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(low), np.log(high), n))


def input_weights(n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.choice([-1.0, 1.0], size=n) / np.sqrt(n)


def covariance(decays: np.ndarray, b: np.ndarray, phi: float) -> np.ndarray:
    outer = np.outer(decays, decays)
    base = np.outer(b, b) / (1.0 - outer)
    if phi == 0.0:
        return base
    numerator = 1.0 - outer * phi**2
    denom = np.outer(1.0 - decays * phi, 1.0 - decays * phi)
    return base * numerator / denom


def cross_covariance(decays: np.ndarray, b: np.ndarray, phi: float, k: int) -> np.ndarray:
    if phi == 0.0:
        return b * decays**k
    with np.errstate(divide="ignore", invalid="ignore"):
        geometric = np.where(
            np.abs(decays - phi) < 1e-15, k * phi ** (k - 1), (1.0 - (decays / phi) ** k) / (1.0 - decays / phi)
        )
    first = phi**k * geometric
    second = decays**k / (1.0 - decays * phi)
    return b * (first + second)


def capacity(decays: np.ndarray, b: np.ndarray, k_max: int, phi: float,
             ridge_rel: float) -> np.ndarray:
    cov = covariance(decays, b, phi)
    cov = cov + ridge_rel * float(np.mean(np.diag(cov))) * np.eye(cov.shape[0])
    curve = np.empty(k_max)
    for k in range(1, k_max + 1):
        g = cross_covariance(decays, b, phi, k)
        curve[k - 1] = float(g @ np.linalg.solve(cov, g))
    return curve


def crossing(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    flips = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0]
    if flips.size == 0:
        return float("nan")
    i = int(flips[0])
    return float((i + 1) + diff[i] / (diff[i] - diff[i + 1]))


def match_centre(wide: np.ndarray, b: np.ndarray, k_max: int, spread: float, ridge_rel: float,
                 target: float, lo: float = 0.3, hi: float = 1.0 - 1e-4) -> tuple[float, float]:
    def total(centre: float) -> float:
        band = log_uniform_band(centre / spread, min(centre * spread, 1.0 - 1e-9), wide.size)
        return float(capacity(band, b, k_max, 0.0, ridge_rel).sum())

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < target:
            lo = mid
        else:
            hi = mid
    centre = 0.5 * (lo + hi)
    return centre, total(centre)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--rho", type=float, default=0.99)
    parser.add_argument("--k-max", type=int, default=200)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--phi", type=float, default=0.5, help="AR(1) coefficient of the task input")
    parser.add_argument("--spread", type=float, default=1.15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    n = args.units
    b = input_weights(n)
    wide = log_uniform_band(args.rho, args.rho ** (1.0 / 2.0 ** (args.blocks - 1)), n)

    # probe-matched construction: centre solved under the i.i.d. probe distribution
    target_probe = float(capacity(wide, b, args.k_max, 0.0, args.ridge).sum())
    centre_probe, _ = match_centre(wide, b, args.k_max, args.spread, args.ridge, target_probe)

    report: dict[str, object] = {
        "schema": "dir0003_ar1_transfer_check_v1",
        "generated_by": "scripts/dir0003_ar1_transfer_check.py",
        "config": {"units": n, "blocks": args.blocks, "rho": args.rho, "k_max": args.k_max,
                   "ridge_relative": args.ridge, "spread": args.spread, "phi": args.phi},
        "probe_matched_centre": centre_probe,
        "per_distribution": {},
        "claims_not_made": ["no training", "population quantities of a frozen linear core only"],
    }
    for phi in (0.0, args.phi):
        band = log_uniform_band(centre_probe / args.spread,
                                min(centre_probe * args.spread, 1.0 - 1e-9), n)
        wide_curve = capacity(wide, b, args.k_max, phi, args.ridge)
        band_curve = capacity(band, b, args.k_max, phi, args.ridge)
        report["per_distribution"][f"phi={phi}"] = {
            "wide_total": float(wide_curve.sum()),
            "probe_matched_band_total": float(band_curve.sum()),
            "total_ratio": float(band_curve.sum() / wide_curve.sum()),
            "crossover_lag": crossing(wide_curve, band_curve),
            "max_gap_band_favoured_k_le_10": float(np.max(band_curve[:10] - wide_curve[:10])),
            "max_gap_wide_favoured_k_ge_25": float(np.max(wide_curve[24:100] - band_curve[24:100])),
        }

    # is a *task-distribution* matched band possible at all, and does its crossover move?
    target_task = float(capacity(wide, b, args.k_max, args.phi, args.ridge).sum())
    centre_task, achieved_task = match_centre(wide, b, args.k_max, args.spread, args.ridge, target_task)
    band_task = log_uniform_band(centre_task / args.spread,
                                 min(centre_task * args.spread, 1.0 - 1e-9), n)
    task_wide = capacity(wide, b, args.k_max, args.phi, args.ridge)
    task_band = capacity(band_task, b, args.k_max, args.phi, args.ridge)
    probe_band = capacity(band_task, b, args.k_max, 0.0, args.ridge)
    probe_wide = capacity(wide, b, args.k_max, 0.0, args.ridge)
    report["task_distribution_matched_alternative"] = {
        "matched_centre": centre_task,
        "task_distribution_total_ratio": float(achieved_task / target_task),
        "crossover_lag_under_task_distribution": crossing(task_wide, task_band),
        "probe_distribution_total_ratio_of_same_band": float(probe_band.sum() / probe_wide.sum()),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                      **report["per_distribution"],
                      "task_matched_alternative": report["task_distribution_matched_alternative"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
