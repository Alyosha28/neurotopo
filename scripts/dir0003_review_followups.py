"""DIR-0003 review follow-ups: (I9) same selection rule in the nonlinear control,
(I11) split of the held-out contrast variance into seed and ridge components.

I9: the linear specification selects the member maximising the log-timescale-spread contrast
among total-matched allocations. The nonlinear control previously selected the member
closest in total capacity, which is a different rule. Here the nonlinear control selects
among tanh-total-matched candidates by the same second-moment criterion.

I11: with seeds x ridges and no replication, a two-way decomposition without interaction
attributes variance to seed, to ridge, and to the remainder.
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
LOG_TAU = lambda d: np.log(1.0 / (1.0 - d))  # noqa: E731
CELLS = {"adjacent_phi0.5": ("research/design/dir0003_spec_curve_primary8.json"),
         "separated_phi0.5": ("research/design/dir0003_spec_curve_heldout_bands.json")}


def variance_split(rows: list[dict], key: str) -> dict:
    seeds = sorted({r["seed_index"] for r in rows})
    ridges = sorted({r["ridge_relative"] for r in rows})
    table = np.full((len(seeds), len(ridges)), np.nan)
    for r in rows:
        table[seeds.index(r["seed_index"]), ridges.index(r["ridge_relative"])] = r[key]
    grand = np.nanmean(table)
    seed_means = np.nanmean(table, axis=1)
    ridge_means = np.nanmean(table, axis=0)
    ss_total = float(np.nansum((table - grand) ** 2))
    ss_seed = float(len(ridges) * np.nansum((seed_means - grand) ** 2))
    ss_ridge = float(len(seeds) * np.nansum((ridge_means - grand) ** 2))
    ss_resid = ss_total - ss_seed - ss_ridge
    share = lambda x: float(x / ss_total) if ss_total > 0 else float("nan")  # noqa: E731
    return {"ss_total": ss_total, "share_seed": share(ss_seed), "share_ridge": share(ss_ridge),
            "share_remainder": share(ss_resid),
            "seed_means": [float(v) for v in seed_means], "ridge_means": [float(v) for v in ridge_means],
            "ridges": ridges}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--tanh-tolerance", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    n = args.units
    report: dict[str, object] = {"schema": "dir0003_review_followups_v1",
                                 "generated_by": "scripts/dir0003_review_followups.py",
                                 "i11_variance_split": {}, "i9_tanh_same_rule": []}
    for cell, path in CELLS.items():
        rows = json.loads(Path(path).read_text())["rows"]
        report["i11_variance_split"][cell] = {
            "max_on_grid": variance_split(rows, "C1_total_matched_only.max_abs_contrast_on_grid")
            if False else None}
        flat = [{"seed_index": r["seed_index"], "ridge_relative": r["ridge_relative"],
                 "grid": r["C1_total_matched_only"]["max_abs_contrast_on_grid"],
                 "long": r["C1_total_matched_only"]["contrast_long_lag_ge_20"]} for r in rows]
        report["i11_variance_split"][cell] = {
            "max_on_grid": variance_split(flat, "grid"),
            "long_lag": variance_split(flat, "long"),
            "note": "two-way decomposition without interaction; remainder absorbs seed x ridge",
        }

    for seed_index in range(args.seeds):
        b = np.random.default_rng(V1.SEED + seed_index).standard_normal(n) / np.sqrt(n)
        reference = (n // 3, n // 3, n - 2 * (n // 3))
        cache: dict = {}

        def tanh_curve(counts, ridge):
            key = (counts, ridge)
            if key not in cache:
                states, u = TANH.simulate(V1.allocation_decays(counts), b, args.phi, args.steps,
                                          1.0, V1.SEED + 900 + seed_index)
                cache[key] = TANH.capacity(states, u, 200, ridge)
            return cache[key]

        ref_curve = tanh_curve(reference, 0.01)
        ref_total = float(ref_curve.sum())
        ref_m2 = float(np.std(LOG_TAU(V1.allocation_decays(reference))))
        candidates = []
        for n3 in LADDER_N3:
            n2 = 0
            best_gap, target = np.inf, None
            for cand in range(0, n - n3 + 1):
                decays = V1.allocation_decays((n - n3 - cand, cand, n3))
                gap = abs(V1.total_exact(decays, b, args.phi, 0.01)
                          / V1.total_exact(V1.allocation_decays(reference), b, args.phi, 0.01) - 1.0)
                if gap < best_gap:
                    best_gap, n2 = gap, cand
            counts = (n - n3 - n2, n2, n3)
            curve = tanh_curve(counts, 0.01)
            diff = curve - ref_curve
            candidates.append({
                "counts": list(counts),
                "tanh_total_ratio": float(curve.sum() / ref_total),
                "moment2": float(np.std(LOG_TAU(V1.allocation_decays(counts)))),
                "moment2_deviation": abs(float(np.std(LOG_TAU(V1.allocation_decays(counts)))) - ref_m2),
                "contrast_max_on_grid": float(max(abs(diff[k - 1]) for k in GRID)),
                "contrast_long_lag_ge_20": float(max(abs(diff[k - 1]) for k in GRID if k >= 20)),
            })
        qualified = [c for c in candidates if abs(c["tanh_total_ratio"] - 1.0) <= args.tanh_tolerance]
        chosen = max(qualified, key=lambda c: c["moment2_deviation"]) if qualified else None
        report["i9_tanh_same_rule"].append({
            "seed_index": seed_index, "n_qualified": len(qualified), "candidates": candidates,
            "chosen_by_moment2_rule": chosen,
            "artifact_band_ridge_band": float(np.max(np.abs(ref_curve - tanh_curve(reference, 0.03)))),
        })
    payload = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "file_sha256": hashlib.sha256((payload + "\n").encode()).hexdigest(),
                      "i11": {k: {kk: (vv if not isinstance(vv, dict) else
                                        {k3: (round(v3, 3) if isinstance(v3, float) else v3)
                                         for k3, v3 in vv.items() if k3.startswith("share")})
                                   for kk, vv in v.items()}
                              for k, v in report["i11_variance_split"].items()},
                      "i9": [{"seed": r["seed_index"], "qualified": r["n_qualified"],
                              "chosen": r["chosen_by_moment2_rule"]} for r in report["i9_tanh_same_rule"]]},
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
