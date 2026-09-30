"""DIR-0003 R2 (matched): nested re-matching under a static nonlinearity.

The first R2 run reused a pair matched under the LINEAR law and found it unmatched under
tanh (total ratio 0.82-0.88), which is a statement about matching portability rather than
a matched control. This script performs the nested search the objective requires: for each
seed it estimates the tanh-core total capacity of a ladder of allocations, keeps the one
closest to the reference, and then reports the contrast curve against the artifact band
measured in the same run. Held-out ridge estimates; no training of the core.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("v1", HERE / "dir0003_v1_exact_check.py")
V1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V1)
SPEC2 = importlib.util.spec_from_file_location("tanh", HERE / "dir0003_tanh_control.py")
TANH = importlib.util.module_from_spec(SPEC2)
SPEC2.loader.exec_module(TANH)

GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48)
LADDER_N3 = (0, 8, 16, 24, 32, 40, 48)


def linear_matched_n2(b: np.ndarray, n3: int, phi: float, ridge: float, n: int) -> int:
    """Pick n2 so that the LINEAR total for (n-n3-n2, n2, n3) is closest to the reference."""
    reference = (n // 3, n // 3, n - 2 * (n // 3))
    target = V1.total_exact(V1.allocation_decays(reference), b, phi, ridge)
    best, best_gap = 0, np.inf
    for n2 in range(0, n - n3 + 1):
        decays = V1.allocation_decays((n - n3 - n2, n2, n3))
        gap = abs(V1.total_exact(decays, b, phi, ridge) / target - 1.0)
        if gap < best_gap:
            best, best_gap = n2, gap
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--k-max", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    n = args.units
    reference = (n // 3, n // 3, n - 2 * (n // 3))
    rows = []
    for seed_index in range(args.seeds):
        rng = np.random.default_rng(V1.SEED + seed_index)
        b = rng.standard_normal(n) / np.sqrt(n)
        cache: dict[tuple, np.ndarray] = {}

        def tanh_curve(counts: tuple[int, int, int], ridge: float) -> np.ndarray:
            key = (counts, ridge)
            if key not in cache:
                decays = V1.allocation_decays(counts)
                states, u = TANH.simulate(decays, b, args.phi, args.steps, args.gain,
                                          V1.SEED + 900 + seed_index)
                cache[key] = TANH.capacity(states, u, args.k_max, ridge)
            return cache[key]

        ref_curve = {r: tanh_curve(reference, r) for r in (0.01, 0.03)}
        ref_total = float(ref_curve[0.01].sum())
        candidates = []
        for n3 in LADDER_N3:
            n2 = linear_matched_n2(b, n3, args.phi, 0.01, n)
            counts = (n - n3 - n2, n2, n3)
            curve = tanh_curve(counts, 0.01)
            linear_decays = V1.allocation_decays(counts)
            linear_total = V1.total_exact(linear_decays, b, args.phi, 0.01)
            linear_ref = V1.total_exact(V1.allocation_decays(reference), b, args.phi, 0.01)
            candidates.append({
                "counts": list(counts),
                "tanh_total_ratio": float(curve.sum() / ref_total),
                "linear_total_ratio": float(linear_total / linear_ref),
            })
        best = min(candidates, key=lambda c: abs(c["tanh_total_ratio"] - 1.0))
        counts = tuple(best["counts"])
        diff = tanh_curve(counts, 0.01) - ref_curve[0.01]
        own = ref_curve[0.01] - ref_curve[0.03]
        rows.append({
            "seed_index": seed_index,
            "reference": list(reference),
            "linear_matched_pair_for_this_seed": {
                "note": "the pair used in the first R2 run",
                "linear_total_ratio": float(V1.total_exact(
                    V1.allocation_decays((49, 11, 4)), b, args.phi, 0.01)
                    / V1.total_exact(V1.allocation_decays(reference), b, args.phi, 0.01)),
                "tanh_total_ratio": float(tanh_curve((49, 11, 4), 0.01).sum() / ref_total),
            },
            "nested_tanh_matched": best,
            "all_candidates": candidates,
            "contrast_max_on_grid": float(max(abs(diff[k - 1]) for k in GRID)),
            "contrast_long_lag_ge_20": float(max(abs(diff[k - 1]) for k in GRID if k >= 20)),
            "artifact_band": float(np.max(np.abs(own))),
            "tail_fraction_beyond_lag_100": float(ref_curve[0.01][100:].sum() / ref_total),
        })
    report = {
        "schema": "dir0003_tanh_matched_v1",
        "generated_by": "scripts/dir0003_tanh_matched.py",
        "config": {"units": n, "phi": args.phi, "steps": args.steps, "gain": args.gain,
                   "k_max": args.k_max, "seeds": args.seeds, "grid": GRID,
                   "ladder_n3": list(LADDER_N3)},
        "rows": rows,
        "claims_not_made": ["held-out ridge estimates on a frozen nonlinear core",
                            "truncated lag sum is reported with its tail fraction",
                            "no training, no capability claim, no Gate 1 claim"],
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "file_sha256": hashlib.sha256((payload + "\n").encode()).hexdigest(),
                      "rows": [{k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in row.items() if k != "all_candidates"}
                               for row in rows]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
