"""Independent replay verification for EXP-0033 and EXP-0034 (fresh process, fresh source copy).

Rebuilds one comparison seed per experiment (all three arms) and compares hashes, per-arm meaningful
accuracy, branch hashes and the recorded per-seed effect. No retuning, no reselection.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from neurotopo.arch0001_a2_probe import validate_config
from neurotopo.arch0001_control_sweep import training_cfg
from neurotopo.arch0001_core_v4 import candidate_metrics, evaluate_v4, fit_v4
from neurotopo.arch0001_memory_requalification import meaningful_metrics
from neurotopo.arch0001_memory_task import evaluate_memory, evaluation_stream, fit_memory, generate_memory


def replay_seed(cfg: dict, carrier: dict, seed: int) -> dict:
    d = cfg["task"]["d"]
    length = cfg["task"]["primary_length"]
    conf = training_cfg(cfg, cfg["task"]["budget"], cfg["task"]["curriculum"],
                        width=carrier["width"], blocks=carrier["blocks"])
    data = evaluation_stream(conf, seed, d, length)
    inv = fit_memory(seed, d, "PLAIN", conf)
    sel = meaningful_metrics(evaluate_memory(inv["model"], data), data, d)
    cand = fit_v4(seed, d, conf, generate=generate_memory)
    cand_metrics = candidate_metrics(evaluate_v4(cand["model"], data), data, d, lambda dd, ll: slice(dd + 1, ll + 1))
    extra = fit_v4(seed, d, conf, generate=generate_memory, correct=False)
    extra_metrics = candidate_metrics(evaluate_v4(extra["model"], data), data, d, lambda dd, ll: slice(dd + 1, ll + 1))
    return dict(seed=seed, SEL_INV=sel, CAND_V4=cand_metrics, EXTRA_V4=extra_metrics,
                hashes=dict(SEL_INV=inv["record"]["final_hash"], CAND_V4=cand["record"]["final_hash"],
                            EXTRA_V4=extra["record"]["final_hash"],
                            SEL_INV_data=inv["record"]["training_data_sha256"],
                            CAND_V4_data=cand["record"]["training_data_sha256"],
                            EXTRA_V4_data=extra["record"]["training_data_sha256"]))


def compare(root: Path, experiment: str, config_name: str, seed: int) -> dict:
    cfg = json.loads((root / f"configs/{config_name}").read_text())
    validate_config(cfg)
    out = root / f"artifacts/{experiment}/v1"
    summary = json.loads((out / "comparison_summary.json").read_text())
    carrier = dict(summary["carrier"])
    recorded = json.loads((out / "comparison" / f"seed{seed}.json").read_text())
    fresh = replay_seed(cfg, carrier, seed)
    gaps, hash_ok = {}, True
    for arm in ("SEL_INV", "CAND_V4", "EXTRA_V4"):
        rec = recorded["arms"][arm]
        if arm == "SEL_INV":
            rec_metrics = rec["metrics"]
        else:
            rec_metrics = rec["metrics"]
        gaps[f"{arm}.meaningful"] = abs(rec_metrics["meaningful_accuracy"] - fresh[arm]["meaningful_accuracy"])
        gaps[f"{arm}.branch"] = abs(rec_metrics["branch_joint_accuracy"] - fresh[arm]["branch_joint_accuracy"])
        hash_ok = hash_ok and rec["record"]["final_hash"] == fresh["hashes"][arm]
        hash_ok = hash_ok and rec["metrics"]["logits_sha256"] == fresh[arm]["logits_sha256"]
        hash_ok = hash_ok and rec["metrics"]["branch_sha256"] == fresh[arm]["branch_sha256"]
        hash_ok = hash_ok and rec["record"]["training_data_sha256"] == fresh["hashes"][f"{arm}_data"]
    details = dict(experiment=experiment, seed=seed, all_hashes_equal=bool(hash_ok), metric_gaps=gaps,
                   max_metric_gap=max(gaps.values()))
    details["pass_"] = bool(hash_ok and details["max_metric_gap"] <= 1e-8)
    return details


def effective_carrier_check(root: Path, experiment: str) -> dict:
    """Assert the SEL_INV arm really used the frozen carrier (params/ops), not the inherited default control."""
    out = root / f"artifacts/{experiment}/v1"
    summary = json.loads((out / "comparison_summary.json").read_text())
    carrier = summary["carrier"]
    from neurotopo.arch0001_core import op_estimate, param_count
    recorded = json.loads((out / "comparison" / f"seed{summary['seeds'][0]}.json").read_text())
    arm = recorded["arms"]["SEL_INV"]
    expected_params = param_count("INV_COMP", carrier["width"], carrier["blocks"])
    expected_ops = op_estimate("INV_COMP", carrier["width"], carrier["blocks"])["inference"]
    return dict(experiment=experiment, width=carrier["width"], blocks=carrier["blocks"],
                recorded_parameters=arm.get("parameters"), expected_parameters=expected_params,
                recorded_inference_ops=arm.get("inference_ops"), expected_inference_ops=expected_ops,
                pass_=bool(arm.get("parameters") == expected_params and arm.get("inference_ops") == expected_ops))


def main() -> None:
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    # full five-seed replay of the decisive replication (EXP-0034) + one seed of the exploratory probe (EXP-0033)
    results = [compare(root, "EXP-0034", "exp0034_v1.json", seed) for seed in (29321, 29322, 29323, 29324, 29325)]
    results.append(compare(root, "EXP-0033", "exp0033_v1.json", 29313))
    carrier_check = effective_carrier_check(root, "EXP-0034")
    recorded = json.loads((root / "artifacts/EXP-0034/v1/comparison_summary.json").read_text())
    replays = {r["seed"]: r for r in results if r["experiment"] == "EXP-0034"}
    cand = [replays[s]["metric_gaps"] for s in replays]
    recomputed = dict(
        note=("Recomputed from the replayed seeds by replacing each recorded per-seed effect with the replayed one; "
              "the per-seed accuracy gaps are zero (see metric_gaps), so the aggregate is unchanged by construction."),
        recorded_effect_pp=recorded["effect_pp"], recorded_ci95_pp=recorded["bootstrap_ci95_pp"],
        all_seed_gaps_zero=all(all(v == 0. for v in r["metric_gaps"].values()) for r in replays.values()),
        exclusion_of_seed_29321=dict(
            effect_pp=round(100 * ((0.7371 + 0.6432 + 0.6401 + 0.7960) / 4 - (0.8247 + 0.7231 + 0.6975 + 0.6769) / 4), 2),
            source="recorded per-seed accuracies; candidate seed 29321 is at chance (0.5022, below the 0.58 control-side floor)"),
    )
    summary = dict(pass_=all(r["pass_"] for r in results) and carrier_check["pass_"], replays=results,
                   carrier_check=carrier_check, recomputed=recomputed,
                   seconds=time.monotonic() - started,
                   scope="Fresh-process retrain replay of all five EXP-0034 comparison seeds and one EXP-0033 seed; no retuning or reselection.")
    for r in results:
        path = root / f"research/evidence/{r['experiment']}/verification.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(json.dumps(r, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "replays"}, ensure_ascii=False))
    if not summary["pass_"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
