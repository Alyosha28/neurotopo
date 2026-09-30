"""Independent replay verification for EXP-0029 v1 (fresh process, fresh source copy).

Rebuilds archived runs from the frozen config and compares parameter/data/prediction hashes and every
metric against the archived records. It never retunes, reselects seeds or overwrites original artifacts.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from neurotopo.arch0001_core import fit, metrics_for, model_tag
from neurotopo.arch0001_learning_qualification import stream_for, validate_config

METRICS = ("accuracy_all", "flip_accuracy", "hold_accuracy", "response_joint_accuracy",
           "current_ok_next_response_bad", "bce")


def compare(label: str, record: dict, refit: dict) -> dict:
    gaps = {}
    hash_equal = True
    for name in record["cells"]:
        for key in METRICS:
            a, b = record["cells"][name].get(key), refit["cells"][name].get(key)
            if a is None or b is None:
                continue
            gaps[f"{name}.{key}"] = abs(a - b)
        hash_equal = hash_equal and record["cells"][name]["logits_sha256"] == refit["cells"][name]["logits_sha256"]
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
    cfg = json.loads((root / "configs/exp0029_v1.json").read_text())
    validate_config(cfg)
    out = root / "artifacts/EXP-0029/v1"
    summary = json.loads((out / "analysis.json").read_text())["summary"]

    cand_rows = summary["arms"]["CAND"]["rows"]
    failures = [r for r in cand_rows if not r["passed"]]
    passes = [r for r in cand_rows if r["passed"]]
    targets = []
    if failures:
        targets.append((failures[0]["seed"], dict(kind="CAND", width=32, blocks=1)))
    if passes:
        targets.append((passes[-1]["seed"], dict(kind="CAND", width=32, blocks=1)))
    control = summary["arms"]["INV_COMP-w32-b1"]["rows"]
    targets.append((control[0]["seed"], dict(kind="INV_COMP", width=32, blocks=1)))
    hardest = min(cand_rows, key=lambda r: r["IID_U_flip"])
    targets.append((hardest["seed"], dict(kind="EXTRA_SUP", width=32, blocks=1)))

    results = []
    for seed, arm in targets:
        tag = model_tag(arm["kind"], arm["width"], arm["blocks"])
        record = json.loads((out / "runs" / f"{tag}_seed{seed}.json").read_text())
        refit_run = fit(arm["kind"], seed, dict(leak=0., sigma=0.), cfg, cfg["training"]["updates"],
                        width=arm["width"], blocks=arm["blocks"])
        refit = dict(record=refit_run["record"],
                     cells={name: metrics_for(refit_run["model"], stream_for(cfg, seed, cell))
                            for name, cell in cfg["cells"].items()})
        results.append(compare(f"{tag}:seed{seed}", record, refit))

    summary_out = dict(pass_=all(r["pass_"] for r in results), replays=results,
                       seconds=time.monotonic() - started,
                       scope="Fresh-process retrain replay; no retuning or reselection.")
    path = root / "research/evidence/EXP-0029/verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary_out, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary_out.items() if k != "replays"}, ensure_ascii=False))
    if not summary_out["pass_"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
