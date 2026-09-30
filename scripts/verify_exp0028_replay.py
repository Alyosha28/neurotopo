"""Independent replay verification for EXP-0028 v1 (fresh process, fresh source copy).

Rebuilds recorded runs from the frozen config and compares parameter/data/prediction hashes and metrics
against the archived records. It never retunes, reselects seeds or overwrites the original artifacts.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from neurotopo.arch0001_core import condition_streams, fit, metrics_for, validate_config


def compare(label: str, record: dict, refit: dict, metrics_keys: tuple[str, ...]) -> dict:
    new = refit["record"]
    details = dict(label=label, seed=record["seed"], final_hash_equal=record["final_hash"] == new["final_hash"],
                   data_hash_equal=record["training_data_sha256"] == new["training_data_sha256"])
    metric_gaps: dict[str, float] = {}
    hash_equal = True
    for cell in ("primary", "secondary"):
        if cell not in record:
            continue
        for key in metrics_keys:
            metric_gaps[f"{cell}.{key}"] = abs(record[cell][key] - refit[cell][key])
        hash_equal = hash_equal and record[cell]["logits_sha256"] == refit[cell]["logits_sha256"]
    details["logits_hash_equal"] = hash_equal
    details["metric_gaps"] = metric_gaps
    details["max_metric_gap"] = max(metric_gaps.values()) if metric_gaps else 0.
    details["pass_"] = (details["final_hash_equal"] and details["data_hash_equal"] and hash_equal
                        and details["max_metric_gap"] <= 1e-8)
    return details


def main() -> None:
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/exp0028_v1.json").read_text())
    validate_config(cfg)
    out = root / "artifacts/EXP-0028/v1"
    metrics_keys = ("flip_accuracy", "hold_accuracy", "accuracy_all", "bce", "response_error")
    results = []

    scan = json.loads((out / "scan" / "summary.json").read_text())
    by_id = {e["condition"]["id"]: e for e in scan["results"]}
    for condition_id, seed, width, blocks in (("N5_T64_leak0_s40", 28101, 32, 1),
                                              ("N8_T256_leak4_s40", 28102, 32, 1)):
        entry = by_id[condition_id]
        record = next(r for r in entry["seeds"] if r["seed"] == seed)
        condition = dict(entry["condition"])
        streams = condition_streams(cfg, condition)
        refit_run = fit("INV_COMP", seed, condition, cfg, cfg["training"]["updates"], width=width, blocks=blocks)
        refit = dict(record=refit_run["record"],
                     primary=metrics_for(refit_run["model"], streams["primary"]),
                     secondary=metrics_for(refit_run["model"], streams["secondary"]))
        results.append(compare(f"scan:{condition_id}:INV_COMP-w{width}-b{blocks}", record, refit, metrics_keys))

    record = json.loads((out / "pretest" / "CAND_seed28103.json").read_text())
    ns = cfg["streams"]["pretest_namespace"]
    from neurotopo.arch0001_core import generate
    cell_data = {k: generate(cfg["streams"]["eval_examples"], v[0], [ns, v[0], (1 if v[1] == "HOLD" else 2), 0],
                             v[1], v[2], v[3]) for k, v in cfg["pretest_cells"].items()}
    refit_run = fit("CAND", 28103, dict(leak=0., sigma=0.), cfg, cfg["training"]["updates"])
    refit = dict(record=refit_run["record"], cells={k: metrics_for(refit_run["model"], v) for k, v in cell_data.items()})
    gaps = {k: abs(record["cells"][k]["accuracy_all"] - refit["cells"][k]["accuracy_all"]) for k in record["cells"]}
    bce_gaps = {k: abs(record["cells"][k]["bce"] - refit["cells"][k]["bce"]) for k in record["cells"]}
    logits_equal = all(record["cells"][k]["logits_sha256"] == refit["cells"][k]["logits_sha256"]
                       for k in record["cells"])
    details = dict(label="pretest:CAND", seed=28103,
                   final_hash_equal=record["final_hash"] == refit["record"]["final_hash"],
                   data_hash_equal=record["training_data_sha256"] == refit["record"]["training_data_sha256"],
                   logits_hash_equal=bool(logits_equal),
                   accuracy_gaps=gaps, max_accuracy_gap=max(gaps.values()),
                   bce_gaps=bce_gaps, max_bce_gap=max(bce_gaps.values()))
    details["pass_"] = (details["final_hash_equal"] and details["data_hash_equal"] and details["logits_hash_equal"]
                        and details["max_accuracy_gap"] <= 1e-8 and details["max_bce_gap"] <= 1e-8)
    results.append(details)

    summary = dict(pass_=all(r["pass_"] for r in results), replays=results,
                   seconds=time.monotonic() - started, scope="Fresh-process retrain replay; no retuning or reselection.")
    path = root / "research/evidence/EXP-0028/verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "replays"}, ensure_ascii=False))
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    if not summary["pass_"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
