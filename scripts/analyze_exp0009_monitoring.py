"""Read-only, exploratory analysis of EXP-0009's three frozen d=1 monitors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SEEDS = tuple(range(9401, 9411))
STEPS = (300, 900, 1500)
MODELS = ("multi_gru", "random_gru")
BOOTSTRAP_SEED = 995001
BOOTSTRAP_SAMPLES = 20_000
ROOT = Path("artifacts/EXP-0009/v1")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_result(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), digest(raw)


def analyze(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    records: dict[tuple[int, str], dict] = {}
    sources = []
    errors = []
    for seed in SEEDS:
        for model in MODELS:
            path = ROOT / f"seed-{seed}" / model / "result.json"
            try:
                result, source_sha = load_result(path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{path}: {error}")
                continue
            records[seed, model] = result
            sources.append({"seed": seed, "model": model, "path": str(path),
                            "sha256": source_sha})
            if result.get("seed") != seed or result.get("model") != model:
                errors.append(f"{path}: seed/model identity mismatch")
            if set(result.get("monitoring", {})) != {str(step) for step in STEPS}:
                errors.append(f"{path}: monitor steps mismatch")
            if result.get("status") != "COMPLETED":
                errors.append(f"{path}: not completed")

    for seed in SEEDS:
        if not all((seed, model) in records for model in MODELS):
            continue
        m3, c0 = (records[seed, name] for name in MODELS)
        for field in ("training_dataset_sha256", "training_schedule_sha256",
                      "train_updates", "train_examples", "train_tokens",
                      "training_forward_linear_flops"):
            if m3.get(field) != c0.get(field):
                errors.append(f"seed={seed}: {field} differs between models")
        if m3.get("train_updates") != 1500:
            errors.append(f"seed={seed}: unexpected update budget")
        for step in STEPS:
            first = m3.get("monitoring", {}).get(str(step), {})
            second = c0.get("monitoring", {}).get(str(step), {})
            if not first or not second:
                continue
            if first.get("dataset_sha256") != second.get("dataset_sha256"):
                errors.append(f"seed={seed},step={step}: monitor dataset differs")
            first_step = m3.get("monitoring", {}).get(str(STEPS[0]), {})
            if first.get("dataset_sha256") != first_step.get("dataset_sha256"):
                errors.append(f"seed={seed},step={step}: monitor dataset changed across steps")
    monitor_hashes = {
        result["monitoring"][str(step)].get("dataset_sha256")
        for result in records.values() for step in STEPS
        if str(step) in result.get("monitoring", {})
    }
    if len(monitor_hashes) != 1 or None in monitor_hashes:
        errors.append("Monitor dataset differs across seeds or contains a missing hash")

    report = {
        "kind": "EXP0009_FROZEN_MONITOR_EXPLORATORY_NOT_GATE1_EVIDENCE",
        "design": "research/design/exp0009_monitor_reanalysis.md",
        "design_sha256": digest(Path("research/design/exp0009_monitor_reanalysis.md").read_bytes()),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "seeds": list(SEEDS), "steps": list(STEPS),
        "monitor_delay": 1, "monitor_examples": 1024,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "source_files": sources, "input_errors": errors,
        "bootstrap_scope": "Paired model/graph/training seeds; one fixed d=1 validation stream; no episode or checkpoint resampling",
        "compute_scope": "Stored analytical training forward-linear FLOP proxy and training wall time; not measured total training FLOPs",
    }
    if errors:
        report["strategy_label"] = "INVALID_INPUT"
    else:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        rows = []
        summaries = {}
        for step in STEPS:
            m3_vals = np.array([records[seed, "multi_gru"]["monitoring"][str(step)]["validation"]["final_joint_kl_to_oracle"] for seed in SEEDS], dtype=np.float64)
            c0_vals = np.array([records[seed, "random_gru"]["monitoring"][str(step)]["validation"]["final_joint_kl_to_oracle"] for seed in SEEDS], dtype=np.float64)
            delta = c0_vals - m3_vals
            bootstrap = delta[rng.integers(0, len(SEEDS), size=(BOOTSTRAP_SAMPLES, len(SEEDS)))].mean(axis=1)
            for index, seed in enumerate(SEEDS):
                rows.append({"seed": seed, "step": step,
                             "m3_kl_nat": float(m3_vals[index]),
                             "c0_kl_nat": float(c0_vals[index]),
                             "c0_minus_m3_kl_nat": float(delta[index]),
                             "m3_full_run_training_seconds": records[seed, "multi_gru"]["training_seconds"],
                             "c0_full_run_training_seconds": records[seed, "random_gru"]["training_seconds"],
                             "full_run_forward_linear_flops_per_arm": records[seed, "multi_gru"]["training_forward_linear_flops"]})
            lower, upper = np.percentile(bootstrap, [2.5, 97.5])
            summaries[str(step)] = {
                "m3_mean_kl_nat": float(m3_vals.mean()),
                "c0_mean_kl_nat": float(c0_vals.mean()),
                "mean_c0_minus_m3_kl_nat": float(delta.mean()),
                "sample_sd_c0_minus_m3_kl_nat": float(delta.std(ddof=1)),
                "positive_seed_count": int((delta > 0).sum()),
                "paired_bootstrap_95ci": [float(lower), float(upper)],
            }
        report["rows"] = rows
        report["summaries"] = summaries
        early = [summaries[str(step)] for step in STEPS[:2]]
        if any(summary["c0_mean_kl_nat"] < 0.01 for summary in early):
            label = "UNINFORMATIVE_CEILING"
        elif all(summary["mean_c0_minus_m3_kl_nat"] > 0 and
                 summary["paired_bootstrap_95ci"][0] > 0 and
                 summary["positive_seed_count"] >= 8 for summary in early):
            label = "EARLY_SIGNAL_PRESENT"
        else:
            label = "NO_CONSISTENT_EARLY_SIGNAL"
        report["strategy_label"] = label
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/analysis/exp0009_monitor_reanalysis_v1.json"))
    args = parser.parse_args()
    report = analyze(args.output)
    print(json.dumps({"output": str(args.output), "label": report["strategy_label"],
                      "input_errors": len(report["input_errors"])}), flush=True)


if __name__ == "__main__":
    main()
