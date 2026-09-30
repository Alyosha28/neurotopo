"""DIR-0004 independent second implementation of the ladder and the affordance screen.

The first implementation (`dir0004_precision_check.py`, `dir0004_affordance_screen.py`)
obtains every number by *stepping the recursion* and *scanning* a bit grid. This script
recomputes the same quantities in closed form instead, so agreement is evidence that the
published numbers are not artefacts of step count, scan order or floating-point drift.

Hidden coordinate. Storage quantisation adds c = q^2/12 per step with q = 2^-b. The first
implementation applies *predict then re-store* inside one loop iteration and records
var[t] after t+1 such cycles, i.e.
    var[t] = c (lam^{2(t+2)} - 1) / (lam^2 - 1),
and returns the crossing step T* = t+1 for the first t with var[t] > eps^2. The closed
form below counts cycles explicitly (`cycles` = number of predict/re-store cycles), so the
two conventions are written down instead of being left implicit -- the first version of
this crosscheck silently used one cycle fewer and produced an apparent off-by-one.

Observed coordinate. The Kalman step caps the variance at sigma^2, so
    var_{t+1} = f(lam^2 var_t) + c,   f(v) = sigma^2 v / (sigma^2 + v),
whose fixed point solves the quadratic
    lam^2 v^2 + (sigma^2 - lam^2 sigma^2 - c lam^2) v - c sigma^2 = 0,
i.e. the required bits are bounded in T and are read off the positive root.

No training, no samples, CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def hidden_variance(lam: float, bits: int, cycles: int) -> float:
    """Error variance after `cycles` predict-and-re-store cycles (cycles >= 1)."""
    c = (2.0 ** (-bits)) ** 2 / 12.0
    return c * (lam ** (2 * (cycles + 1)) - 1.0) / (lam ** 2 - 1.0)


def hidden_required_bits(lam: float, T: int, eps: float) -> int:
    # The first implementation tests var[T-1] = c (lam^{2(T+1)} - 1)/(lam^2 - 1) < eps^2,
    # i.e. 2b > log2((lam^{2(T+1)} - 1) / (12 eps^2 (lam^2 - 1))).
    value = (lam ** (2 * (T + 1)) - 1.0) / (12.0 * eps ** 2 * (lam ** 2 - 1.0))
    return int(np.ceil(0.5 * np.log2(value) - 1e-12))


def hidden_crossing(lam: float, bits: int, eps: float, cycles_max: int = 4000) -> float:
    """First 1-based cycle index whose recorded variance exceeds eps^2."""
    for cycles in range(1, cycles_max + 1):
        if hidden_variance(lam, bits, cycles) > eps ** 2:
            return float(cycles)
    return float("nan")


def observed_fixed_point(lam: float, bits: int, sigma: float) -> float:
    c = (2.0 ** (-bits)) ** 2 / 12.0
    a = lam ** 2
    b = sigma ** 2 - lam ** 2 * sigma ** 2 - c * lam ** 2
    cc = -c * sigma ** 2
    disc = b * b - 4 * a * cc
    if disc < 0:
        raise SystemExit("observed fixed point has no real root")
    return float((-b + np.sqrt(disc)) / (2 * a))


def observed_required_bits(lam: float, sigma: float, eps: float, bits_max: int = 200) -> int:
    for bits in range(1, bits_max + 1):
        if observed_fixed_point(lam, bits, sigma) < eps ** 2:
            return bits
    raise SystemExit("observed bit search exhausted")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigma", type=float, default=1e-6)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--lam-u", type=float, default=2.0)
    ap.add_argument("--screen", type=Path,
                    default=Path("research/design/dir0004_affordance_screen.json"))
    ap.add_argument("--ladder", type=Path,
                    default=Path("research/design/dir0004_precision_check.json"))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)

    screen = json.loads(a.screen.read_text(encoding="utf-8"))
    rows = screen["conditions"]["hidden"]["rows"]
    hidden_deltas = {row["T"]: hidden_required_bits(a.lam_u, row["T"], a.eps) - row["min_bits"]
                     for row in rows}
    observed_rows = screen["conditions"]["observed"]["rows"]
    closed_observed = observed_required_bits(a.lam_u, a.sigma, a.eps)
    observed_deltas = {row["T"]: closed_observed - row["min_bits"] for row in observed_rows}

    # The ladder's own regime (eps = 1e-6, no observation of the expanding mode).
    ladder = json.loads(a.ladder.read_text(encoding="utf-8"))
    ladder_eps = ladder["declared"]["eps"]
    crossings = {}
    for bits in (32, 63):
        crossings[bits] = {
            "closed_form": hidden_crossing(a.lam_u, bits, ladder_eps),
            "reported": next(row["crossing_T"] for row in ladder["rows"]
                             if row["bits_expanding"] == bits),
        }

    out = {
        "schema": "dir0004_screen_crosscheck_v1",
        "method": "closed form (this script) versus step recursion plus bit scan (first implementation)",
        "config": {"sigma": a.sigma, "eps": a.eps, "lam_u": a.lam_u,
                   "screen": str(a.screen), "ladder": str(a.ladder)},
        "sources": {
            "screen_sha256": hashlib.sha256(a.screen.read_bytes()).hexdigest(),
            "ladder_sha256": hashlib.sha256(a.ladder.read_bytes()).hexdigest(),
        },
        "hidden_required_bits_delta": hidden_deltas,
        "observed_required_bits": {"closed_form": closed_observed, "deltas": observed_deltas},
        "ladder_crossings": crossings,
        "self_checks": {},
    }
    out["self_checks"] = {
        "hidden_required_bits_agree_exactly": all(v == 0 for v in hidden_deltas.values()),
        "observed_required_bits_agree_exactly": all(v == 0 for v in observed_deltas.values()),
        "ladder_crossings_agree_exactly": all(
            v["closed_form"] == v["reported"] for v in crossings.values()),
        "observation_floor_below_eps_squared": bool(
            observed_fixed_point(a.lam_u, 200, a.sigma) < a.eps ** 2),
        "crosscheck_valid": bool(
            all(v == 0 for v in hidden_deltas.values())
            and all(v == 0 for v in observed_deltas.values())
            and all(v["closed_form"] == v["reported"] for v in crossings.values())),
    }
    payload = json.dumps(out, indent=2, sort_keys=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"output": str(a.output),
                      "sha256": hashlib.sha256((payload + "\n").encode()).hexdigest(),
                      "ladder_crossings": crossings,
                      "self_checks": out["self_checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
