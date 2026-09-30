"""DIR-0003 diagnostic: does the matched-total contrast track the first moment?

A single matched scalar leaves the mean log-timescale free. This script asks, cell by cell,
how much of the between-allocation contrast is explained by the first-moment deviation
(|dm1|) versus the residual higher-moment deviation (|dm2|), and how large the matched
family and the R1 (total + first moment matched) pool actually are. Population quantities
of frozen linear cores; no training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("sc", HERE / "dir0003_spec_curve.py")
SC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SC)
V1 = SC.V1

CELLS = {
    "adjacent_phi0.5": ((0.20, 0.40), (0.60, 0.75), (0.90, 0.97)),
    "separated_phi0.5": ((0.10, 0.30), (0.50, 0.70), (0.85, 0.95)),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--moment1-tolerance", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    n = args.units
    rows = []
    for cell, bands in CELLS.items():
        V1.BANDS = bands
        for seed_index in range(args.seeds):
            b = np.random.default_rng(V1.SEED + seed_index).standard_normal(n) / np.sqrt(n)
            for ridge in (0.01, 0.03):
                meta, rungs = SC.enumerate_rungs(b, args.phi, ridge, args.tolerance)
                if len(rungs) < 4:
                    rows.append({"cell": cell, "seed_index": seed_index, "ridge_relative": ridge,
                                 "matched_family": len(rungs), "r1_pool": 0, "note": "family too small"})
                    continue
                ref1, ref2 = meta["ref_moment1"], meta["ref_moment2"]
                ref_curve = V1.capacity_curve(V1.allocation_decays(tuple(meta["reference"])),
                                              b, args.phi, ridge, 49)

                def contrast(rung: dict) -> float:
                    d = V1.capacity_curve(rung["decays"], b, args.phi, ridge, 49) - ref_curve
                    return max(abs(d[k - 1]) for k in SC.GRID)

                dm1 = np.array([r["moment1"] - ref1 for r in rungs])
                dm2 = np.array([r["moment2"] - ref2 for r in rungs])
                con = np.array([contrast(r) for r in rungs])
                pool = [i for i in range(len(rungs)) if abs(dm1[i]) <= args.moment1_tolerance]
                design = np.column_stack([np.abs(dm1), np.abs(dm2), np.ones(len(rungs))])
                coef, *_ = np.linalg.lstsq(design, con, rcond=None)
                predicted = design @ coef
                r_squared = 1.0 - float(((con - predicted) ** 2).sum() / ((con - con.mean()) ** 2).sum())
                rows.append({
                    "cell": cell, "seed_index": seed_index, "ridge_relative": ridge,
                    "matched_family": len(rungs), "r1_pool": len(pool),
                    "family_dm1_range": [float(dm1.min()), float(dm1.max())],
                    "family_dm2_range": [float(dm2.min()), float(dm2.max())],
                    "ols_abs_dm1_coef": float(coef[0]), "ols_abs_dm2_coef": float(coef[1]),
                    "ols_intercept": float(coef[2]), "ols_r_squared": r_squared,
                    "max_contrast_in_family": float(con.max()),
                    "max_contrast_in_r1_pool": float(con[pool].max()) if pool else None,
                })
    summary = {}
    for cell in CELLS:
        subset = [r for r in rows if r["cell"] == cell and "ols_r_squared" in r]
        if not subset:
            continue
        summary[cell] = {
            "n_cells": len(subset),
            "matched_family_median": float(np.median([r["matched_family"] for r in subset])),
            "r1_pool_median": float(np.median([r["r1_pool"] for r in subset])),
            "ols_dm1_coef_median": float(np.median([r["ols_abs_dm1_coef"] for r in subset])),
            "ols_dm2_coef_median": float(np.median([r["ols_abs_dm2_coef"] for r in subset])),
            "ols_r2_median": float(np.median([r["ols_r_squared"] for r in subset])),
            "max_contrast_family_median": float(np.median([r["max_contrast_in_family"] for r in subset])),
            "max_contrast_r1pool_median": float(np.median(
                [r["max_contrast_in_r1_pool"] for r in subset if r["max_contrast_in_r1_pool"] is not None])),
        }
    report = {"schema": "dir0003_moment_diagnostic_v1",
              "generated_by": "scripts/dir0003_moment_diagnostic.py",
              "config": {"units": n, "phi": args.phi, "tolerance": args.tolerance,
                         "moment1_tolerance": args.moment1_tolerance, "seeds": args.seeds,
                         "cells": {k: [list(b) for b in v] for k, v in CELLS.items()}},
              "rows": rows, "summary": summary,
              "claims_not_made": ["population quantities of frozen linear cores",
                                  "no training, no capability claim, no Gate 1 claim"]}
    payload = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "file_sha256": hashlib.sha256((payload + "\n").encode()).hexdigest(),
                      "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
