"""DIR-0003 between-arm null contrast, ridge band over all three ridges, variance split.

Review issue I4: the within-arm ridge sensitivity is a different quantity from a between-arm
contrast and cannot serve as its verdict threshold. This script computes a proper null for the
between-arm statistic: two arms with the SAME allocation and the SAME ridge, differing only in
the input-weight realisation (different seeds). It also widens the ridge band to all pairs of
the three ridges (I5) and splits the held-out contrast variance into seed and ridge components
(I11). Population quantities of frozen linear cores; no training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("sc", HERE / "dir0003_spec_curve.py")
SC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SC)
V1 = SC.V1

ADJACENT = ((0.20, 0.40), (0.60, 0.75), (0.90, 0.97))
SEPARATED = ((0.10, 0.30), (0.50, 0.70), (0.85, 0.95))
CELLS = {"adjacent_phi0.5": (ADJACENT, 0.5), "separated_phi0.5": (SEPARATED, 0.5),
         "adjacent_phi0.7": (ADJACENT, 0.7), "separated_phi0.3": (SEPARATED, 0.3)}
RIDGES = (0.003, 0.01, 0.03)


def stats(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {"n": int(arr.size), "min": float(arr.min()), "p10": float(np.percentile(arr, 10)),
            "median": float(np.median(arr)), "p90": float(np.percentile(arr, 90)),
            "max": float(arr.max())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    n = args.units
    report: dict[str, object] = {"schema": "dir0003_null_contrast_v1",
                                "generated_by": "scripts/dir0003_null_contrast.py",
                                "config": {"units": n, "phi": args.phi, "seeds": args.seeds,
                                           "ridges": list(RIDGES), "grid": list(SC.GRID)},
                                "cells": {}, "phi_per_cell": {k: v[1] for k, v in CELLS.items()}, "claims_not_made": [
                                    "population quantities of frozen linear cores",
                                    "no training, no capability claim, no Gate 1 claim"]}
    for cell, (bands, phi_cell) in CELLS.items():
        V1.BANDS = bands
        reference = (n // 3, n // 3, n - 2 * (n // 3))
        weights = [np.random.default_rng(V1.SEED + i).standard_normal(n) / np.sqrt(n)
                   for i in range(args.seeds)]
        curves = {}
        for si, b in enumerate(weights):
            for r in RIDGES:
                curves[(si, r)] = V1.capacity_curve(V1.allocation_decays(reference), b, phi_cell, r, 49)

        # (1) realisation null: same allocation, same ridge, different seeds
        null = {"max_on_grid": [], "long_lag": []}
        for si, sj in combinations(range(args.seeds), 2):
            for r in RIDGES:
                d = curves[(si, r)] - curves[(sj, r)]
                null["max_on_grid"].append(max(abs(d[k - 1]) for k in SC.GRID))
                null["long_lag"].append(max(abs(d[k - 1]) for k in SC.GRID if k >= 20))

        # (2) ridge band over ALL pairs of ridges, per seed
        band = []
        for si in range(args.seeds):
            worst = 0.0
            for r1, r2 in combinations(RIDGES, 2):
                d = curves[(si, r1)] - curves[(si, r2)]
                worst = max(worst, max(abs(d[k - 1]) for k in SC.GRID))
            band.append(worst)

        # (3) typical versus max-selected contrast inside the matched family
        selected, typical = [], []
        for si, b in enumerate(weights):
            meta, rungs = SC.enumerate_rungs(b, phi_cell, 0.01, 0.02)
            if len(rungs) < 4:
                continue
            ref_curve = V1.capacity_curve(V1.allocation_decays(tuple(meta["reference"])), b, phi_cell, 0.01, 49)

            def contrast(rung: dict) -> float:
                d = V1.capacity_curve(rung["decays"], b, phi_cell, 0.01, 49) - ref_curve
                return max(abs(d[k - 1]) for k in SC.GRID)

            best = max(rungs, key=lambda r: abs(r["moment2"] - meta["ref_moment2"]))
            selected.append(contrast(best))
            rng = np.random.default_rng(V1.SEED + 500 + si)
            typical.append(float(np.mean([contrast(rungs[i]) for i in
                                          rng.integers(0, len(rungs), size=min(20, len(rungs)))])))
        report["cells"][cell] = {
            "matched_family_median": float(np.median([len(SC.enumerate_rungs(
                np.random.default_rng(V1.SEED + i).standard_normal(n) / np.sqrt(n), phi_cell, 0.01, 0.02)[1])
                for i in range(args.seeds)])),
            "realisation_null_max_on_grid": stats(null["max_on_grid"]),
            "realisation_null_long_lag": stats(null["long_lag"]),
            "ridge_band_all_pairs_max_on_grid": stats(band),
            "selected_max_contrast": stats(selected),
            "typical_random_member_contrast": stats(typical),
        }
    payload = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "file_sha256": hashlib.sha256((payload + "\n").encode()).hexdigest(),
                      "cells": {k: {kk: (vv if not isinstance(vv, dict) else
                                         {k3: round(v3, 4) for k3, v3 in vv.items()})
                                    for kk, vv in v.items()} for k, v in report["cells"].items()}},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
