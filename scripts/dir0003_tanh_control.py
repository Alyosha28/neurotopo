"""DIR-0003 R2 control: does the linear-core conclusion survive a static nonlinearity?

Re-entry condition (3) of decision0075 requires the tanh control to be COMPUTED, not
asserted. The tanh core keeps the same timescale-allocation organisation and the same
AR(1) input law, and replaces the linear response by an elementwise tanh:

    x_t = lambda * x_{t-1} + (1 - lambda) * tanh(s * b * u_t)

so the organisation (the decay allocation) is unchanged while the readout becomes a
nonlinear functional of the input history. Total and per-lag capacities are estimated by
held-out ridge regression (train/test split), with the design Gram factorised once and
reused across lags. No training of the core; no capability claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "v1", Path(__file__).with_name("dir0003_v1_exact_check.py"))
V1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V1)

GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48)


def simulate(decays: np.ndarray, b: np.ndarray, phi: float, steps: int, gain: float,
             seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(steps + 400) * np.sqrt(1.0 - phi**2)
    u = np.empty_like(eps)
    u[0] = eps[0]
    for t in range(1, u.size):
        u[t] = phi * u[t - 1] + eps[t]
    x = np.zeros(decays.size)
    states = np.empty((steps, decays.size))
    for t in range(steps):
        x = decays * x + (1.0 - decays) * np.tanh(gain * b * u[400 + t])
        states[t] = x
    return states, u  # keep the warm-up prefix: lag-k targets need u[400 - k + t]


def capacity(states: np.ndarray, u: np.ndarray, k_max: int, ridge_rel: float,
             test_fraction: float = 0.5) -> np.ndarray:
    steps = states.shape[0]
    split = int(steps * (1.0 - test_fraction))
    gram = states[:split].T @ states[:split]
    gram = gram + ridge_rel * float(np.mean(np.diag(gram))) * np.eye(gram.shape[0])
    inverse = np.linalg.inv(gram)
    curve = np.empty(k_max)
    for k in range(1, k_max + 1):
        target = u[400 - k: 400 - k + steps]
        coefficient = inverse @ (states[:split].T @ target[:split])
        residual = target[split:] - states[split:] @ coefficient
        curve[k - 1] = 1.0 - float(residual @ residual) / float(target[split:] @ target[split:])
    return curve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--k-max", type=int, default=200)
    parser.add_argument("--pairs", type=int, default=2, help="number of seed pairs to test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    n = args.units
    rows = []
    for seed_index in range(args.pairs):
        rng = np.random.default_rng(V1.SEED + seed_index)
        b = rng.standard_normal(n) / np.sqrt(n)
        allocations = [(49, 11, 4), (21, 21, 22)]  # linear-case matched pair, same seeds
        curves = {}
        for counts in allocations:
            decays = V1.allocation_decays(tuple(counts))
            states, u = simulate(decays, b, args.phi, args.steps, args.gain, V1.SEED + 700 + seed_index)
            for ridge in (0.01, 0.03):
                curve = capacity(states, u, args.k_max, ridge)
                curves[(counts, ridge)] = curve
        for ridge in (0.01, 0.03):
            a = curves[(allocations[0], ridge)]
            z = curves[(allocations[1], ridge)]
            diff = a - z
            own = curves[(allocations[0], 0.01)] - curves[(allocations[0], 0.03)]
            rows.append({
                "seed_index": seed_index, "ridge_relative": ridge, "gain": args.gain,
                "total_ratio": float(a.sum() / z.sum()),
                "tail_change_200_vs_100": float(a[100:].sum() / max(a.sum(), 1e-12)),
                "max_abs_contrast_on_grid": float(max(abs(diff[k - 1]) for k in GRID)),
                "contrast_long_lag_ge_20": float(max(abs(diff[k - 1]) for k in GRID if k >= 20)),
                "own_curve_ridge_band": float(np.max(np.abs(own))),
            })
    report = {
        "schema": "dir0003_tanh_control_v1",
        "generated_by": "scripts/dir0003_tanh_control.py",
        "config": {"units": n, "phi": args.phi, "steps": args.steps, "gain": args.gain,
                   "k_max": args.k_max, "pairs": args.pairs, "grid": GRID,
                   "allocations": [[49, 11, 4], [21, 21, 22]]},
        "rows": rows,
        "claims_not_made": ["held-out ridge estimates on a frozen nonlinear core",
                            "no training, no capability claim, no Gate 1 claim"],
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "file_sha256": hashlib.sha256((payload + "\n").encode()).hexdigest(),
                      "rows": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
