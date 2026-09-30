"""Post-hoc robustness report for the single near-miss EXP-0028 condition (N5, sigma=0.4, T=64).

Registered after the reflection identified that the frozen negative verdict for the scan hinges on N5's
2-seed hold mean 0.5989 vs the pre-registered 0.60 floor. This report adds three pre-specified extra
seeds and a bootstrap over the 1024-example evaluation stream. It does NOT change the frozen verdict or
grant any gate: the registered rule and its 2-seed result stand as recorded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neurotopo.arch0001_core import condition_streams, evaluate, fit, op_estimate, param_count

SEEDS = [28101, 28102, 28103, 28104, 28105]
CONDITION = dict(id="N5_T64_leak0_s40", length=64, leak=0.0, sigma=0.4)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/exp0028_v1.json").read_text())
    streams = condition_streams(cfg, CONDITION)
    per_seed = []
    records = {}
    for seed in SEEDS:
        archived = root / "artifacts/EXP-0028/v1/scan" / f"{CONDITION['id']}_seed{seed}.json"
        if archived.exists():
            record = json.loads(archived.read_text())
            per_seed.append(dict(seed=seed, source="archived", flip=record["primary"]["flip_accuracy"],
                                 hold=record["primary"]["hold_accuracy"]))
            continue
        run = fit("INV_COMP", seed, CONDITION, cfg, cfg["training"]["updates"], width=32, blocks=1)
        parts = evaluate(run["model"], streams["primary"])
        logits = parts["logits"].numpy().astype(np.float64)
        target = streams["primary"]["targets"]
        u = streams["primary"]["commands"]
        correct = (logits >= 0) == (target >= .5)
        records[seed] = correct
        per_seed.append(dict(seed=seed, source="posthoc",
                             flip=float(correct[:, 1:][u == 1].mean()),
                             hold=float(correct[:, 1:][u == 0].mean())))
    flip = np.array([r["flip"] for r in per_seed])
    hold = np.array([r["hold"] for r in per_seed])
    rng = np.random.default_rng(20260929)
    bootstrap = {}
    if records:
        seed0 = sorted(records)[0]
        u = streams["primary"]["commands"]
        correct = records[seed0]
        rows = np.arange(correct.shape[0])
        flips, holds = [], []
        for _ in range(4000):
            pick = rng.choice(rows, size=len(rows), replace=True)
            flips.append(float(correct[pick][:, 1:][u[pick] == 1].mean()))
            holds.append(float(correct[pick][:, 1:][u[pick] == 0].mean()))
        bootstrap = dict(seed=seed0, flip_ci95=[float(np.percentile(flips, 2.5)), float(np.percentile(flips, 97.5))],
                         hold_ci95=[float(np.percentile(holds, 2.5)), float(np.percentile(holds, 97.5))],
                         draws=4000)
    summary = dict(condition=CONDITION, seeds=SEEDS, per_seed=per_seed,
                   flip_mean=float(flip.mean()), flip_min=float(flip.min()), flip_max=float(flip.max()),
                   hold_mean=float(hold.mean()), hold_min=float(hold.min()), hold_max=float(hold.max()),
                   bootstrap_example_resampling=bootstrap,
                   frozen_rule="flip mean in [.60,.95], every seed in [.55,.97], hold mean >= .60",
                   frozen_two_seed_result=dict(flip_mean=0.62175, hold_mean=0.5989347999427532, qualified=False),
                   interpretation=("Added seeds and example bootstrap are post-hoc robustness only; the frozen "
                                    "two-seed scan verdict and EXP-0028 decision are unchanged."),
                   candidate_inference_ops=op_estimate("CAND")["inference"],
                   control_parameters=param_count("INV_COMP", 32, 1))
    path = root / "research/evidence/EXP-0028/posthoc_sensitivity.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"}, ensure_ascii=False))
    for r in per_seed:
        print(r)


if __name__ == "__main__":
    main()
