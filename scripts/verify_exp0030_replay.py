"""Independent replay verification for EXP-0030 v1 (fresh process, fresh source copy).

Rebuilds archived runs from the frozen config and compares parameter/data/prediction hashes and every
metric against the archived records. It never retunes, reselects seeds or overwrites original artifacts.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from neurotopo.arch0001_memory_task import evaluate_memory, evaluation_stream, fit_memory, memory_metrics, validate_config

METRICS = ("accuracy_all", "final_accuracy", "initial_accuracy", "tail_accuracy", "branch_joint_accuracy")


def compare(label: str, record: dict, refit: dict) -> dict:
    gaps = {}
    hash_equal = True
    for name in [key for key in record if key == "primary" or key.startswith("secondary_")]:
        for key in METRICS:
            gaps[f"{name}.{key}"] = abs(record[name][key] - refit[name][key])
        hash_equal = (hash_equal and record[name]["logits_sha256"] == refit[name]["logits_sha256"]
                      and record[name]["branch_sha256"] == refit[name]["branch_sha256"])
    details = dict(label=label, seed=record["seed"],
                   final_hash_equal=record["final_hash"] == refit["record"]["final_hash"],
                   data_hash_equal=record["training_data_sha256"] == refit["record"]["training_data_sha256"],
                   logits_hash_equal=bool(hash_equal),
                   max_metric_gap=max(gaps.values()) if gaps else 0., metric_gaps=gaps)
    details["pass_"] = (details["final_hash_equal"] and details["data_hash_equal"] and details["logits_hash_equal"]
                        and details["max_metric_gap"] <= 1e-8)
    return details


def main() -> None:
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/exp0030_v1.json").read_text())
    validate_config(cfg)
    out = root / "artifacts/EXP-0030/v1"
    scan = json.loads((out / "scan_summary.json").read_text())
    confirm = json.loads((out / "confirm_summary.json").read_text())
    qualified = scan["qualified"]
    targets = [(f"D{qualified['d']}_{qualified['sup']}", cfg["seeds"]["scan"][0]),
               (f"D{qualified['d']}_{qualified['sup']}", cfg["seeds"]["confirm"][-1])]

    results = []
    for item_id, seed in targets:
        record = json.loads((out / "runs" / f"{item_id}_seed{seed}.json").read_text())
        refit = fit_memory(seed, record["d"], record["sup"], cfg)
        refit_metrics = dict(primary=memory_metrics(evaluate_memory(refit["model"], evaluation_stream(cfg, seed, record["d"], cfg["evaluation"]["primary_length"])),
                                                 evaluation_stream(cfg, seed, record["d"], cfg["evaluation"]["primary_length"])))
        for length in cfg["evaluation"]["secondary_lengths"]:
            data = evaluation_stream(cfg, seed, record["d"], length)
            refit_metrics[f"secondary_{length}"] = memory_metrics(evaluate_memory(refit["model"], data), data)
        results.append(compare(f"{item_id}:seed{seed}", record, dict(record=refit["record"], **refit_metrics)))

    summary = dict(pass_=all(r["pass_"] for r in results), replays=results,
                   qualified=qualified, confirm_reproduced=confirm["reproduced"],
                   seconds=time.monotonic() - started,
                   scope="Fresh-process retrain replay; no retuning or reselection.")
    path = root / "research/evidence/EXP-0030/verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "replays"}, ensure_ascii=False))
    if not summary["pass_"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
