"""DIR-0003 specification curve: allocation contrasts at exactly matched total capacity.

Implements re-entry conditions (2) and (3) of decision0075:
  * every reported contrast is recomputed per readout ridge and per input-weight seed,
    so no single-point extremum can serve as a verdict input;
  * ALL total-capacity-matched allocations are enumerated, and an R1 control is added
    that also matches the FIRST MOMENT of the log-timescale distribution, so the
    residual freedom is a higher moment rather than the mean.

Population quantities of frozen linear cores under an AR(1) task law. No training.
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
LOG_TAU = lambda d: np.log(1.0 / (1.0 - d))  # noqa: E731


def first_flip(diff: np.ndarray) -> int | None:
    values = [diff[k - 1] for k in GRID]
    signs = [np.sign(v) for v in values]
    for i in range(len(GRID) - 1):
        if signs[i] * signs[i + 1] < 0:
            return int(GRID[i + 1])
    return None


def enumerate_rungs(b: np.ndarray, phi: float, ridge: float, tolerance: float) -> list[dict]:
    n = b.size
    reference = (n // 3, n // 3, n - 2 * (n // 3))
    ref_decays = V1.allocation_decays(reference)
    ref_total = V1.total_exact(ref_decays, b, phi, ridge)
    ref_l1 = float(np.mean(LOG_TAU(ref_decays)))
    ref_l2 = float(np.std(LOG_TAU(ref_decays)))
    rungs = []
    for n1 in range(n + 1):
        for n2 in range(n - n1 + 1):
            counts = (n1, n2, n - n1 - n2)
            decays = V1.allocation_decays(counts)
            total = V1.total_exact(decays, b, phi, ridge)
            ratio = total / ref_total
            if abs(ratio - 1.0) > tolerance:
                continue
            log_tau = LOG_TAU(decays)
            rungs.append({"counts": list(counts), "total_ratio": ratio,
                          "moment1": float(np.mean(log_tau)), "moment2": float(np.std(log_tau)),
                          "decays": decays})
    return [{"reference": list(reference), "ref_total": ref_total, "ref_moment1": ref_l1,
             "ref_moment2": ref_l2}, rungs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--moment1-tolerance", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--bands", type=str, default="",
                        help="semicolon-separated lo,hi pairs overriding V1.BANDS")
    parser.add_argument("--label", type=str, default="primary")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.bands:
        parsed = tuple(tuple(float(x) for x in part.split(",")) for part in args.bands.split(";"))
        if len(parsed) != 3 or any(len(b) != 2 for b in parsed):
            parser.error("--bands needs three lo,hi pairs")
        V1.BANDS = parsed

    n = args.units
    rows = []
    for seed_index in range(args.seeds):
        # Gaussian weights: the +-1/ sqrt(n) family is invariant under sign flips of b
        # (C -> D C D and g -> D g leave g^T C^-1 g unchanged), so it spans an EMPTY
        # specification dimension. A Gaussian draw breaks that symmetry.
        rng = np.random.default_rng(V1.SEED + seed_index)
        b = rng.standard_normal(n) / np.sqrt(n)
        for ridge in (0.003, 0.01, 0.03):
            meta, rungs = enumerate_rungs(b, args.phi, ridge, args.tolerance)
            if not rungs:
                continue
            ref_m1, ref_m2 = meta["ref_moment1"], meta["ref_moment2"]
            # C1: total matched only, maximise second-moment contrast
            c1 = max(rungs, key=lambda r: abs(r["moment2"] - ref_m2))
            # C2 (R1 control): total AND first moment matched, maximise second-moment contrast
            pool = [r for r in rungs if abs(r["moment1"] - ref_m1) <= args.moment1_tolerance]
            c2 = max(pool, key=lambda r: abs(r["moment2"] - ref_m2)) if pool else None
            ref_decays = V1.allocation_decays(tuple(meta["reference"]))

            def contrast(rung: dict | None) -> dict | None:
                if rung is None:
                    return None
                decays = rung["decays"]
                ref_curve = V1.capacity_curve(ref_decays, b, args.phi, ridge, 49)
                arm_curve = V1.capacity_curve(decays, b, args.phi, ridge, 49)
                own = V1.capacity_curve(ref_decays, b, args.phi, 0.01, 49)
                own_other = V1.capacity_curve(ref_decays, b, args.phi, 0.03, 49)
                diff = arm_curve - ref_curve
                return {
                    "counts": rung["counts"], "total_ratio": rung["total_ratio"],
                    "moment1": rung["moment1"], "moment2": rung["moment2"],
                    "max_abs_contrast_on_grid": float(max(abs(diff[k - 1]) for k in GRID)),
                    "contrast_long_lag_ge_20": float(max(abs(diff[k - 1]) for k in GRID if k >= 20)),
                    "first_flip_grid_lag": first_flip(diff),
                    "own_curve_ridge_band": float(np.max(np.abs(own - own_other))),
                }

            rows.append({"seed_index": seed_index, "ridge_relative": ridge,
                         "matched_rung_count": len(rungs),
                         "r1_pool_size": len(pool),
                         "C1_total_matched_only": contrast(c1),
                         "C2_total_and_first_moment_matched": contrast(c2)})

    def collect(path: str, key: str) -> list[float]:
        out = []
        for row in rows:
            block = row[path]
            if block is not None:
                out.append(block[key])
        return out

    def summarize(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        arr = np.array(values)
        return {"n": int(arr.size), "median": float(np.median(arr)),
                "min": float(arr.min()), "max": float(arr.max()),
                "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90))}

    flips = [row["C1_total_matched_only"]["first_flip_grid_lag"]
             for row in rows if row["C1_total_matched_only"] is not None]
    report = {
        "schema": "dir0003_spec_curve_v1",
        "generated_by": "scripts/dir0003_spec_curve.py",
        "config": {"label": args.label, "bands": [list(b) for b in V1.BANDS],
                   "units": n, "phi": args.phi, "tolerance": args.tolerance,
                   "moment1_tolerance": args.moment1_tolerance, "seeds": args.seeds, "input_weight_law": "gaussian(0, 1/N)",
                   "ridges": [0.003, 0.01, 0.03], "grid": GRID},
        "rows": [{k: v for k, v in row.items()} for row in rows],
        "summary": {
            "C1_max_abs_contrast_on_grid": summarize(collect("C1_total_matched_only", "max_abs_contrast_on_grid")),
            "C1_contrast_long_lag_ge_20": summarize(collect("C1_total_matched_only", "contrast_long_lag_ge_20")),
            "C1_own_curve_ridge_band": summarize(collect("C1_total_matched_only", "own_curve_ridge_band")),
            "C2_max_abs_contrast_on_grid": summarize(collect("C2_total_and_first_moment_matched", "max_abs_contrast_on_grid")),
            "C2_contrast_long_lag_ge_20": summarize(collect("C2_total_and_first_moment_matched", "contrast_long_lag_ge_20")),
            "C1_first_flip_grid_lags": sorted({int(f) for f in flips if f is not None}),
        },
        "claims_not_made": [
            "no training was run and no capability claim is made",
            "this is a field-level population study of frozen linear cores",
            "Gate 1 is NOT claimed or advanced by this artifact",
        ],
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "file_sha256": hashlib.sha256((payload + "\n").encode()).hexdigest(),
                      "rows": len(rows), "summary": report["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
