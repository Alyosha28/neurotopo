"""Registered EXP-0009 resource checks, paired inference and bounded decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze import paired_summary
from .run import digest, write_json


PRIMARY = ("multi_gru", "random_gru")
MATCHED_RESOURCE_FIELDS = (
    "allocated_parameters", "effective_parameters", "active_recurrent_parameters",
    "hidden_state_floats", "hidden_state_bytes_per_example", "gate_affine_workspace_floats_per_example",
    "parameter_bytes", "mask_buffer_bytes", "dense_forward_linear_macs_per_example",
    "active_forward_linear_macs_per_example", "dense_forward_linear_flops_per_batch",
    "internal_steps_per_input", "sequence_length",
)


def classify(primary: dict[str, Any], *, matching: bool, adequacy: bool,
             headroom: bool, iid_kl_cost: float, h_accuracy_cost: float,
             config: dict[str, Any]) -> str:
    """Apply the registered strict useful-effect and cost-guard rules."""
    if not matching or not adequacy or not headroom:
        return "INCONCLUSIVE"
    threshold = config["minimum_useful_kl_gain"]
    lower, upper = primary["ci95"]
    cost_ok = (iid_kl_cost <= config["max_iid_kl_cost"] and
               h_accuracy_cost >= -config["max_h_accuracy_cost"])
    if primary["mean"] >= threshold and lower > threshold and cost_ok:
        return "SUPPORTED"
    if upper < threshold:
        return "FALSIFIED"
    if lower > 0 and (lower <= threshold <= upper or
                      (lower > threshold and not cost_ok)):
        return "PARTIALLY_SUPPORTED"
    return "INCONCLUSIVE"


def analyze(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    if (manifest["status"] != "COMPLETED" or manifest["run_kind"] != "REGISTERED" or
            manifest["seeds_run"] != config["seeds"]):
        raise ValueError("A complete registered ten-seed run is required")
    if (config["experiment_id"], config["version"], config["analysis"]) != (
            "EXP-0009", "v1", "matched_nuisance_transfer"):
        raise ValueError("Wrong registered analysis")
    preflight = manifest["preflight"]
    validation_ok = (manifest["validation"]["pass"] and
                     manifest["validation"]["source_sha256"] == manifest["source_sha256"] and
                     preflight["pass"] and preflight["source_sha256"] == manifest["source_sha256"] and
                     preflight["config_sha256"] == manifest["config_sha256"] and
                     digest(Path("artifacts/validation/exp0009_preflight.json").read_bytes()) ==
                     manifest["preflight_sha256"])
    results = {(name, seed): json.loads((run / f"seed-{seed}" / name / "result.json").read_text())
               for seed in config["seeds"] for name in config["models"]}
    issues: list[str] = []
    if not validation_ok:
        issues.append("validation/preflight provenance mismatch")
    rows = []
    for seed in config["seeds"]:
        source = next(row for row in preflight["seeds"] if row["seed"] == seed)
        m, c = results["multi_gru", seed], results["random_gru", seed]
        if m["initial_parameters_sha256"] != c["initial_parameters_sha256"]:
            issues.append(f"seed{seed}: primary initial parameters differ")
        if m["cross_original_block_edges"] != 0 or c["cross_original_block_edges"] <= 0:
            issues.append(f"seed{seed}: block/cross-block graph descriptors invalid")
        for name in config["models"]:
            result = results[name, seed]
            if result["status"] != "COMPLETED" or result["seed"] != seed or result["model"] != name:
                issues.append(f"seed{seed}/{name}: incomplete or misidentified result")
            if (result["initial_parameters_sha256"] != source["models"][name]["initial_parameters_sha256"] or
                    result["mask_sha256"] != source["models"][name]["mask_sha256"]):
                issues.append(f"seed{seed}/{name}: preflight model identity mismatch")
            if result["training_dataset_sha256"] != source["training_dataset_sha256"]:
                issues.append(f"seed{seed}/{name}: training data mismatch")
            if result["training_schedule_sha256"] != preflight["schedule_sha256"]:
                issues.append(f"seed{seed}/{name}: schedule mismatch")
            if (result["train_updates"] != config["updates"] or
                    result["train_examples"] != config["updates"] * config["batch_size"] or
                    result["train_tokens"] != preflight["training_tokens_per_arm"] or
                    result["training_forward_linear_flops"] != preflight["training_forward_linear_flops_per_arm"]):
                issues.append(f"seed{seed}/{name}: training budget mismatch")
            if result["dataset_sha256"] != preflight["final_dataset_sha256"]:
                issues.append(f"seed{seed}/{name}: final test stream mismatch")
            checkpoint = run / result["checkpoint_file"]
            if not checkpoint.is_file() or digest(checkpoint.read_bytes()) != result["checkpoint_sha256"]:
                issues.append(f"seed{seed}/{name}: checkpoint digest mismatch")
            predictions = run / f"seed-{seed}" / name / "predictions.npz"
            if not predictions.is_file() or digest(predictions.read_bytes()) != result["predictions_sha256"]:
                issues.append(f"seed{seed}/{name}: predictions digest mismatch")
            graph = result["graph"]
            if name in PRIMARY:
                if (graph["diagonal_connections"] != 64 or
                        any(degree != 16 for degree in graph["in_degree"] + graph["out_degree"]) or
                        (name == "random_gru" and not graph["strongly_connected"])):
                    issues.append(f"seed{seed}/{name}: graph degree/self-loop/connectivity mismatch")
            for d in config["eval_distractors_per_evidence"]:
                key = f"d{d}"
                metrics = result["metrics"][key]
                rows.append({"seed": seed, "model": name, "condition": key, **metrics})
                if result["resources"][key] != source["models"][name]["resources"][key]:
                    issues.append(f"seed{seed}/{name}/{key}: preflight resource mismatch")
                if name in PRIMARY:
                    counterpart = c if name == "multi_gru" else m
                    for field in MATCHED_RESOURCE_FIELDS:
                        if result["resources"][key][field] != counterpart["resources"][key][field]:
                            issues.append(f"seed{seed}/{key}: primary {field} mismatch")
        names = [results[name, seed] for name in config["models"]]
        if len({row["training_dataset_sha256"] for row in names}) != 1 or len({row["train_tokens"] for row in names}) != 1:
            issues.append(f"seed{seed}: arms did not receive matched training")
        for d in config["eval_distractors_per_evidence"]:
            key = f"d{d}"
            if len({row["dataset_sha256"][key] for row in names}) != 1:
                issues.append(f"seed{seed}/{key}: arms did not receive matched final data")

    def series(name: str, d: int, metric: str) -> np.ndarray:
        return np.asarray([results[name, seed]["metrics"][f"d{d}"][metric]
                           for seed in config["seeds"]], dtype=np.float64)

    comparisons = {}
    for d in config["eval_distractors_per_evidence"]:
        comparisons[f"d{d}"] = {
            "C0_minus_M3_final_kl": paired_summary(
                series("random_gru", d, "final_joint_kl_to_oracle") -
                series("multi_gru", d, "final_joint_kl_to_oracle"),
                config["bootstrap_samples"], config["bootstrap_seed"]),
            "M3_minus_C0_h_accuracy": paired_summary(
                series("multi_gru", d, "final_h_true_accuracy") -
                series("random_gru", d, "final_h_true_accuracy"),
                config["bootstrap_samples"], config["bootstrap_seed"]),
        }
    primary = comparisons["d4"]["C0_minus_M3_final_kl"]
    iid_kl_cost = float((series("multi_gru", 1, "final_joint_kl_to_oracle") -
                         series("random_gru", 1, "final_joint_kl_to_oracle")).mean())
    h_accuracy_cost = float((series("multi_gru", 4, "final_h_true_accuracy") -
                             series("random_gru", 4, "final_h_true_accuracy")).mean())
    adequacy = all(series(name, 1, "final_joint_kl_to_oracle").mean() <= config["iid_mean_kl_max"] and
                   np.all(series(name, 1, "final_joint_kl_to_oracle") <= config["iid_each_seed_kl_max"])
                   for name in PRIMARY)
    control = series("random_gru", 4, "final_joint_kl_to_oracle")
    headroom = (config["control_primary_mean_kl_min"] <= control.mean() <=
                config["control_primary_mean_kl_max"] and
                int((control >= config["control_seed_kl_min"]).sum()) >=
                config["control_min_seeds_with_headroom"])
    cost_ok = (iid_kl_cost <= config["max_iid_kl_cost"] and
               h_accuracy_cost >= -config["max_h_accuracy_cost"])
    decision = classify(primary, matching=not issues, adequacy=bool(adequacy),
                        headroom=bool(headroom), iid_kl_cost=iid_kl_cost,
                        h_accuracy_cost=h_accuracy_cost, config=config)
    branch = ("REPAIR_INVALID_EVIDENCE" if issues else
              "INDEPENDENT_REPRODUCTION_AND_SCIENTIFIC_REVIEW" if decision == "SUPPORTED" else
              "STOP_MECHANISM_ESCALATION")
    means = {f"d{d}": {name: {metric: float(series(name, d, metric).mean())
                              for metric in ("pre_joint_kl_to_oracle", "final_joint_kl_to_oracle",
                                             "final_joint_oracle_map_agreement", "final_h_true_accuracy",
                                             "final_joint_true_accuracy")}
                        for name in config["models"]}
             for d in config["eval_distractors_per_evidence"]}
    report = {
        "decision": decision, "scope": "One trained block-vs-degree-matched-rewiring transfer task only",
        "registered_full_run": True, "matching_pass": not issues, "matching_issues": issues,
        "guards": {"iid_adequacy": bool(adequacy), "control_headroom": bool(headroom),
                   "iid_kl_cost": iid_kl_cost, "h_accuracy_cost": h_accuracy_cost,
                   "cost_guard_pass": bool(cost_ok)},
        "primary_effect": primary, "comparisons": comparisons, "means": means, "rows": rows,
        "test_stream_sha256": preflight["final_dataset_sha256"],
        "uncertainty": "Paired seed/bootstrap interval conditional on one frozen final test stream; no test-instance resampling.",
        "next_branch": branch, "gate1_automatically_advanced": False,
        "wall_seconds": manifest["wall_seconds"],
    }
    write_json(run / "analysis.json", report)
    lines = ["# EXP-0009 v1 results", "", f"Decision: **{decision}** for the registered >0.05-nat matched structural effect.",
             "", f"Primary C0−M3 d=4 final KL: {primary['mean']:.4f}; paired-seed 95% CI {primary['ci95']}.",
             f"IID adequacy: {adequacy}; C0 primary headroom: {headroom}; matching: {not issues}; cost guard: {cost_ok}.",
             "", "| d | Model | Pre joint KL | Final joint KL | H true accuracy |", "|---:|---|---:|---:|---:|"]
    for d in config["eval_distractors_per_evidence"]:
        for name in config["models"]:
            entry = means[f"d{d}"][name]
            lines.append(f"| {d} | {name} | {entry['pre_joint_kl_to_oracle']:.4f} | "
                         f"{entry['final_joint_kl_to_oracle']:.4f} | {entry['final_h_true_accuracy']:.4f} |")
    lines += ["", "The dense H64 model is an active-parameter-unmatched practical context, not the causal null.",
              "No Gate 1 passage, semantic hypothesis trajectory or unique time/interference mechanism follows from this result.",
              f"Next branch: {branch}. All seed records and predictions remain under this run directory."]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
