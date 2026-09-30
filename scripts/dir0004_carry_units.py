"""Does a second float32 unit buy a second mantissa's worth of horizon?

The arithmetic-floor measurement shows a single float32 state saturating at its own
mantissa. DIR-0004's predicted T* = 45 needs ~63 stored bits, so the design implicitly
assumes several float32 units can carry one scalar. Whether they can depends only on how
the update treats them, and this script isolates that question with no training:

  exact24      one integer 24-bit code, exact shift            -> 1 mantissa
  exact48      two integer 24-bit codes, exact carry/shift     -> 2 mantissas
  f32_single   one float32 state, float32 arithmetic           -> 1 mantissa
  f32_recombi  two float32 units, but the update recombines them into one float32 value
               before applying the map (the generic situation when the units are mixed by
               ordinary float32 arithmetic)                   -> 1 mantissa, not 2

The doubling map z -> 2z mod 1 is used because it is topologically conjugate to the tent
map with the same Lyapunov exponent ln 2 and its exact multi-unit update is a pure shift,
which isolates the arithmetic effect from the folding arithmetic. The claim under test is
narrow and stated in advance: if f32_recombi matches exact24 rather than exact48, then
extra stored bits above the mantissa are unusable unless the mechanism declares an exact
renormalisation, which is a different mechanism from "expanding/contracting partition".

No learning, no samples, no training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

T_MAX = 200
EPS = 1e-3
Z0 = 0.3141592653589793


def steps_to_eps(reference: np.ndarray, trajectory: np.ndarray) -> float:
    idx = np.where(np.abs(trajectory - reference) > EPS)[0]
    return float(idx[0]) if idx.size else float("nan")


def reference_trajectory(steps: int) -> np.ndarray:
    z = Z0
    out = np.empty(steps)
    for t in range(steps):
        z = (2.0 * z) % 1.0
        out[t] = z
    return out


def exact_code(scales: int, units: int, steps: int) -> np.ndarray:
    """Integer code with `scales` bits per unit over `units` units (total bits = scales*units)."""
    modulus = 1 << (scales * units)
    m = int(round(Z0 * modulus)) % modulus
    out = np.empty(steps)
    for t in range(steps):
        m = (2 * m) % modulus
        out[t] = m / modulus
    return out


def f32_single(steps: int) -> np.ndarray:
    z = np.float32(Z0)
    out = np.empty(steps)
    for t in range(steps):
        z = np.float32((np.float32(2.0) * z) % np.float32(1.0))
        out[t] = float(z)
    return out


def f32_recombined(steps: int) -> np.ndarray:
    """Two float32 units, but each update recombines them into one float32 value first."""
    scale = np.float32(2.0) ** np.float32(-24.0)
    hi = np.float32(Z0)
    lo = np.float32(0.0)
    out = np.empty(steps)
    for t in range(steps):
        combined = np.float32(hi + np.float32(lo * scale))   # rounding happens here
        z = np.float32((np.float32(2.0) * combined) % np.float32(1.0))
        hi, lo = z, np.float32(0.0)
        out[t] = float(z)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)

    ref = reference_trajectory(T_MAX)
    arms = {
        "exact24": exact_code(24, 1, T_MAX),
        "exact48": exact_code(24, 2, T_MAX),
        "f32_single": f32_single(T_MAX),
        "f32_recombined": f32_recombined(T_MAX),
    }
    steps = {name: steps_to_eps(ref, traj) for name, traj in arms.items()}
    out = {
        "schema": "dir0004_carry_units_v1",
        "config": {"eps": EPS, "t_max": T_MAX, "z0": Z0, "map": "z -> 2z mod 1"},
        "steps_to_eps": steps,
        "self_checks": {},
    }
    out["self_checks"] = {
        "two_units_help_under_exact_shift": bool(steps["exact48"] >= steps["exact24"] + 15),
        "two_units_help_nothing_under_float32_recombination": bool(
            abs(steps["f32_recombined"] - steps["f32_single"]) <= 2.0),
        "recombined_matches_single_not_double": bool(
            abs(steps["f32_recombined"] - steps["f32_single"]) <= 2.0
            and steps["f32_recombined"] < steps["exact48"] - 10),
        "exact24_matches_f32_single": bool(abs(steps["exact24"] - steps["f32_single"]) <= 2.0),
    }
    out["self_checks"]["claim_supported"] = bool(
        out["self_checks"]["two_units_help_under_exact_shift"]
        and out["self_checks"]["recombined_matches_single_not_double"])
    payload = json.dumps(out, indent=2, sort_keys=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"output": str(a.output),
                      "sha256": hashlib.sha256((payload + "\n").encode()).hexdigest(),
                      **out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
