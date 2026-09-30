"""Independent replay verification for EXP-0031 v1 (fresh process, fresh source copy).

Rebuilds the frozen DELAY-d models (same seeds/config as EXP0030) and compares parameter/data/prediction
hashes, corrected-endpoint metrics, branch hashes and the parent-archive reproduction flag. No retuning.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from neurotopo.arch0001_memory_requalification import (calibration_stream, meaningful_metrics,
                                                       reference_scores, oracle_score, validate_config)
from neurotopo.arch0001_memory_task import evaluate_memory, evaluation_stream, fit_memory

METRICS = ("meaningful_accuracy", "accuracy_all", "final_accuracy", "tail_mean", "branch_joint_accuracy")


def compare(label: str, record: dict, cfg: dict, refit: dict, refit_primary: dict) -> dict:
    gaps = {k: abs(record["primary"][k] - refit_primary[k]) for k in METRICS}
    details = dict(label=label, seed=record["seed"],
                   final_hash_equal=record["final_hash"] == refit["record"]["final_hash"],
                   data_hash_equal=record["training_data_sha256"] == refit["record"]["training_data_sha256"],
                   logits_hash_equal=record["primary"]["logits_sha256"] == refit_primary["logits_sha256"],
                   branch_hash_equal=record["primary"]["branch_sha256"] == refit_primary["branch_sha256"],
                   parent_reproduced=record["parent_reproduced"],
                   max_metric_gap=max(gaps.values()), metric_gaps=gaps)
    details["pass_"] = (details["final_hash_equal"] and details["data_hash_equal"] and details["logits_hash_equal"]
                        and details["branch_hash_equal"] and details["parent_reproduced"]
                        and details["max_metric_gap"] <= 1e-8)
    return details


def main() -> None:
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/exp0031_v1.json").read_text())
    validate_config(cfg)
    out = root / "artifacts/EXP-0031/v1"
    results = []
    for item_id, seed in (("D2_PLAIN", cfg["seeds"]["scan"][0]), ("D3_PLAIN", cfg["seeds"]["scan"][1])):
        record = json.loads((out / "runs" / f"{item_id}_seed{seed}.json").read_text())
        refit = fit_memory(seed, record["d"], record["sup"], cfg)
        data = evaluation_stream(cfg, seed, record["d"], cfg["evaluation"]["primary_length"])
        primary = meaningful_metrics(evaluate_memory(refit["model"], data), data, record["d"])
        results.append(compare(f"{item_id}:seed{seed}", record, cfg, refit, primary))
    summary = dict(pass_=all(r["pass_"] for r in results), replays=results,
                   seconds=time.monotonic() - started,
                   scope="Fresh-process retrain replay; no retuning or reselection.")
    path = root / "research/evidence/EXP-0031/verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "replays"}, ensure_ascii=False))
    if not summary["pass_"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
