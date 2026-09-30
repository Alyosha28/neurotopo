"""Paired-seed uncertainty and frozen decision rules for EXP-0001."""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .run import write_json


def paired_summary(differences: np.ndarray, samples: int, seed: int) -> dict[str, Any]:
    if len(differences) < 2:
        return {"mean": float(differences.mean()), "sd": None, "ci95": None, "n_seeds": len(differences)}
    rng = np.random.default_rng(seed)
    bootstrap = differences[rng.integers(len(differences), size=(samples, len(differences)))].mean(axis=1)
    return {"mean": float(differences.mean()), "sd": float(differences.std(ddof=1)),
            "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(), "n_seeds": len(differences),
            "differences": differences.tolist()}


def analyze(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    if manifest["status"] != "COMPLETED":
        raise ValueError("Refuse scientific analysis of incomplete run")
    seeds = manifest["seeds_run"]
    results = {(name, seed): json.loads((run / f"seed-{seed}" / name / "result.json").read_text())
               for name in config["models"] for seed in seeds}
    delays = [str(d) for d in config["eval_delays"]]
    comparisons, means = {}, {}
    matching = True
    equality_fields = ["allocated_parameters", "effective_parameters", "hidden_state_floats",
                       "dense_forward_linear_macs_per_example", "active_forward_linear_macs_per_example",
                       "mask_multiplications_per_forward_batch"]
    for seed in seeds:
        m, r = results["multi_gru", seed], results["random_gru", seed]
        matching &= all(m["resources"][d][key] == r["resources"][d][key] for d in delays for key in equality_fields)
        matching &= m["train_tokens"] == r["train_tokens"] and m["train_examples"] == r["train_examples"]
        matching &= m["dataset_sha256"] == r["dataset_sha256"]
        matching &= m["graph"]["diagonal_connections"] == r["graph"]["diagonal_connections"]
    for delay in delays:
        means[delay] = {}
        for name in config["models"]:
            rows = [results[name, seed]["metrics"][delay] for seed in seeds]
            means[delay][name] = {key: {"mean": float(np.mean([row[key] for row in rows])),
                                          "sd": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else None}
                                    for key in ["joint_pair_accuracy", "pre_accuracy", "confirm_accuracy", "revise_accuracy"]}
        comparisons[delay] = {}
        for baseline in ["random_gru", "gru_wide"]:
            differences = np.array([results["multi_gru", seed]["metrics"][delay]["joint_pair_accuracy"] -
                                    results[baseline, seed]["metrics"][delay]["joint_pair_accuracy"] for seed in seeds])
            comparisons[delay][baseline] = paired_summary(differences, config["bootstrap_samples"], config["bootstrap_seed"])
    validation_ok = manifest.get("validation", {}).get("pass", False) and manifest.get("validation", {}).get("source_sha256") == manifest["source_sha256"]
    valid = matching and validation_ok and manifest["run_kind"] == "REGISTERED" and seeds == config["seeds"]
    iid = delays[0]
    learned = all(means[iid][name]["pre_accuracy"]["mean"] >= config["minimum_pre_accuracy"] for name in ["multi_gru", "random_gru"])
    confirm_ok = all(means[d]["multi_gru"]["confirm_accuracy"]["mean"] - means[d]["random_gru"]["confirm_accuracy"]["mean"] >= -config["max_confirm_degradation"] for d in delays)
    def positive(baseline: str, delay: str) -> bool:
        summary = comparisons[delay][baseline]
        return summary["ci95"] is not None and summary["ci95"][0] > 0 and summary["mean"] >= config["minimum_effect"]
    structural = all(positive("random_gru", d) for d in delays) and confirm_ok
    wider = all(comparisons[d]["gru_wide"]["mean"] >= config["minimum_effect"] for d in delays)
    primary_ci = comparisons[iid]["random_gru"]["ci95"]
    if not valid or not learned:
        decision = "INCONCLUSIVE"
    elif structural and wider:
        decision = "SUPPORTED"
    elif structural:
        decision = "PARTIALLY_SUPPORTED"
    elif primary_ci and primary_ci[1] < config["minimum_effect"]:
        decision = "FALSIFIED"
    else:
        decision = "INCONCLUSIVE"
    report = {"decision": decision, "scope": "Fixed-budget block-vs-random recurrence on delayed-evidence-v1 only",
              "resource_matching_pass": bool(matching), "registered_complete": bool(valid),
              "pre_accuracy_requirement_pass": bool(learned), "confirm_guardrail_pass": bool(confirm_ok),
              "means": means, "paired_comparisons": comparisons,
              "gate1_candidate_pass": decision == "SUPPORTED",
              "uncertainty": "Paired training/graph-seed percentile bootstrap; fixed test set; secondary comparisons exploratory.",
              "wall_seconds": manifest["wall_seconds"]}
    write_json(run / "analysis.json", report)
    lines = [f"# EXP-0001 {config['version']} results", "", f"Decision: **{decision}**. Scope: {report['scope']}.", "",
             "| Delay | Model | Joint pair accuracy (mean ± seed SD) | Pre | Confirm | Revise |",
             "|---|---|---:|---:|---:|---:|"]
    for delay in delays:
        for name, row in means[delay].items():
            sd = row["joint_pair_accuracy"]["sd"]
            sd_text = f"{sd:.4f}" if sd is not None else "NA"
            lines.append(f"| {delay} | {name} | {row['joint_pair_accuracy']['mean']:.4f} ± {sd_text} | {row['pre_accuracy']['mean']:.4f} | {row['confirm_accuracy']['mean']:.4f} | {row['revise_accuracy']['mean']:.4f} |")
    lines += ["", "Paired effect M3 − control (joint success; proportion units):", ""]
    for delay in delays:
        for name, row in comparisons[delay].items():
            lines.append(f"- Delay {delay}, {name}: {row['mean']:.4f}, 95% paired-seed bootstrap CI {row['ci95']}.")
    lines += ["", f"Exact primary resource matching: {matching}. Adequate initial-answer accuracy: {learned}.",
              "No latent hypothesis specialization, SAE, biological effect or general architectural superiority is established.",
              "Full seed records are retained in seed-N/model/result.json; active and dense executed compute differ for dense controls."]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), indent=2))


if __name__ == "__main__":
    main()
