"""DIR-0004 arithmetic floor: does the storage-precision law survive finite-precision arithmetic?

The verified law ("one extra stored bit buys one extra step") is derived and machine-checked
under *exact arithmetic plus storage quantisation*: the variance recursion is evaluated in
float64 and only the stored state carries a q^2/12 penalty. A real arm computes its update
in finite precision, so the question this script answers without any training is:

    at which stored-precision budget does the *arithmetic* of the arm, rather than its
    storage, become the binding constraint?

Three measurements on the tent map F(z) = 2z for z < 1/2 else 2 - 2z (Lyapunov exponent
ln 2, i.e. the predicted slope is exactly one step per bit):

  A  float64 arithmetic, b-bit quantised start  -> expect steps ~ b + const.
  B  float32 arithmetic, b-bit quantised start  -> identical while float32 can hold the
     start and its rounding noise stays below the start error.
  C  storage requested beyond the float32 mantissa (b > 24): the start itself rounds, so
     the achievable horizon saturates even though the *allocation* asked for more bits.

Reported: steps to eps per budget, the fitted slope, and the budget at which B separates
from A. No learning, no samples, no training.
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


def tent(z: np.floating) -> np.floating:
    return np.where(z < 0.5, 2.0 * z, 2.0 - 2.0 * z).astype(z.dtype)


def quantise(z: float, bits: int) -> float:
    """Round z to `bits` fractional bits (round to nearest)."""
    scale = 2.0 ** bits
    return float(np.round(z * scale) / scale)


def steps_to_eps(reference: np.ndarray, trajectory: np.ndarray) -> float:
    idx = np.where(np.abs(trajectory - reference) > EPS)[0]
    return float(idx[0]) if idx.size else float("nan")


def run(dtype, z_start: float, steps: int = T_MAX) -> np.ndarray:
    z = dtype(z_start)
    out = np.empty(steps, dtype=np.float64)
    for t in range(steps):
        z = tent(z)
        out[t] = float(z)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bits", type=int, nargs="+",
                    default=[4, 8, 12, 16, 20, 22, 24, 26, 28, 32, 40, 48])
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)

    reference = run(np.float64, Z0)          # unquantised start, float64 arithmetic
    rows = []
    for bits in a.bits:
        z_q = quantise(Z0, bits)
        f64 = steps_to_eps(reference, run(np.float64, z_q))
        f32 = steps_to_eps(reference, run(np.float32, z_q))
        start_error_representable = float(np.float32(z_q)) == z_q
        rows.append({
            "bits": bits,
            "float64_steps": f64,
            "float32_steps": f32,
            "float32_start_exactly_representable": start_error_representable,
            "float32_effective_start_bits": 24 if not start_error_representable else bits,
        })

    def slope(rows_subset, key):
        xs = np.array([r["bits"] for r in rows_subset], dtype=float)
        ys = np.array([r[key] for r in rows_subset], dtype=float)
        ok = ~np.isnan(ys)
        return float(np.polyfit(xs[ok], ys[ok], 1)[0]) if ok.sum() >= 2 else float("nan")

    within = [r for r in rows if r["bits"] <= 24]
    beyond = [r for r in rows if r["bits"] > 24]
    out = {
        "schema": "dir0004_arithmetic_floor_v1",
        "config": {"eps": EPS, "t_max": T_MAX, "z0": Z0, "bits": a.bits},
        "rows": rows,
        "fits": {
            "float64_slope_steps_per_bit_bits_le_24": slope(within, "float64_steps"),
            "float32_slope_steps_per_bit_bits_le_24": slope(within, "float32_steps"),
            "float64_slope_over_all_bits": slope(rows, "float64_steps"),
            "float32_slope_over_all_bits": slope(rows, "float32_steps"),
        },
        "self_checks": {},
    }
    f32_within = [r["float32_steps"] for r in within]
    f32_beyond = [r["float32_steps"] for r in beyond]
    out["self_checks"] = {
        "float64_slope_near_one_within_float32_mantissa": bool(
            abs(out["fits"]["float64_slope_steps_per_bit_bits_le_24"] - 1.0) <= 0.15),
        "float32_slope_near_one_within_own_mantissa": bool(
            abs(out["fits"]["float32_slope_steps_per_bit_bits_le_24"] - 1.0) <= 0.15),
        "float32_saturates_beyond_mantissa": bool(
            f32_beyond and (max(f32_beyond) - min(f32_beyond)) <= 2.0),
        "float32_horizon_gap_beyond_mantissa_steps": float(
            np.nanmax([r["float64_steps"] for r in beyond], initial=np.nan)
            - np.nanmax(f32_beyond, initial=np.nan)) if f32_beyond else None,
    }
    payload = json.dumps(out, indent=2, sort_keys=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"output": str(a.output),
                      "sha256": hashlib.sha256((payload + "\n").encode()).hexdigest(),
                      "rows": rows, "self_checks": out["self_checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
