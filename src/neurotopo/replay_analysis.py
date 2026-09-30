"""Registered seed-level interaction analysis of frozen model responses."""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze import paired_summary
from .run import digest, source_fingerprints, write_json


def analyze(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    frozen_config_path = Path(manifest["config_path"])
    frozen_config_bytes = frozen_config_path.read_bytes()
    if (manifest["experiment_id"] != config["experiment_id"] or manifest["version"] != config["version"] or
        digest(frozen_config_bytes) != manifest["config_sha256"] or
        json.loads(frozen_config_bytes) != config or source_fingerprints() != manifest["source_sha256"]):
        raise ValueError("Frozen config or scientific source mismatch")
    if manifest["status"] != "COMPLETED" or manifest["run_kind"] != "REGISTERED" or manifest["seeds_run"] != config["seeds"]:
        raise ValueError("Complete registered EXP-0005 run required")
    if not manifest["validation"]["pass"] or manifest["validation"]["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Source validation mismatch")
    source_run_manifest = Path("artifacts") / config["source_experiment"] / "manifest.json"
    if (digest(source_run_manifest.read_bytes()) != manifest["source_run_manifest_sha256"] or
        not json.loads(source_run_manifest.read_text())["git_commit"].startswith(config["source_commit"])):
        raise ValueError("Source experiment manifest mismatch")
    records = {(name, seed): json.loads((run / f"seed-{seed}" / name / "result.json").read_text())
               for name in config["models"] for seed in config["seeds"]}
    checkpoint_manifest_path = Path(config["checkpoint_manifest"])
    checkpoint_inputs = json.loads(checkpoint_manifest_path.read_text())["copies"]
    primary_models = ("multi_gru", "random_gru")
    matched = manifest["checkpoint_manifest_sha256"] == digest(checkpoint_manifest_path.read_bytes())
    resource_fields = ["allocated_parameters", "effective_parameters", "hidden_state_floats",
        "dense_forward_linear_macs_per_example", "active_forward_linear_macs_per_example",
        "internal_steps_per_input", "sequence_length", "mask_buffer_bytes",
        "dense_forward_linear_flops_per_batch"]
    expected_keys = {f"{payload}_delay{delay}" for delay in config["delays"] for payload in config["payloads"]}
    for seed in config["seeds"]:
        a, b = (records[name, seed] for name in primary_models)
        matched &= a["initial_parameters_sha256"] == b["initial_parameters_sha256"]
        matched &= a["mask_sha256"] != b["mask_sha256"]
        for name in config["models"]:
            record = records[name, seed]
            matched &= record["status"] == "COMPLETED" and record["source_replay"] == "EXACT"
            matched &= record["checkpoint_sha256"] == checkpoint_inputs[str(seed)][name]["sha256"]
            matched &= set(record["metrics"]) == expected_keys
            for delay in config["delays"]:
                reference_target = a["target_sha256"][f"full_delay{delay}"]
                for payload in config["payloads"]:
                    key = f"{payload}_delay{delay}"
                    matched &= record["target_sha256"][key] == reference_target
                    matched &= record["input_sha256"][key] == a["input_sha256"][key]
                    matched &= record["metrics"][key]["pairs"] == config["test_pairs"]
                    matched &= sum(record["pair_overlap_with_full"][key].values()) == config["test_pairs"]
                    if name == "random_gru":
                        matched &= all(a["resources"][key][field] == b["resources"][key][field] for field in resource_fields)
        for delay in config["delays"]:
            labels = {a["target_sha256"][f"{payload}_delay{delay}"] for payload in config["payloads"]}
            matched &= len(labels) == 1
    if len(records) != len(config["models"]) * len(config["seeds"]):
        matched = False

    def joint(name: str, seed: int, payload: str, delay: int) -> float:
        return records[name, seed]["metrics"][f"{payload}_delay{delay}"]["joint_pair_accuracy"]

    primary_delay = config["primary_delay"]
    interactions: dict[str, dict[str, Any]] = {}
    for delay in config["delays"]:
        interactions[str(delay)] = {}
        for payload in config["payloads"]:
            gaps = np.array([joint("multi_gru", seed, payload, delay) - joint("random_gru", seed, payload, delay)
                             for seed in config["seeds"]])
            interactions[str(delay)][f"gap_{payload}"] = paired_summary(gaps, config["bootstrap_samples"], config["bootstrap_seed"])
            if payload != "full":
                full_gaps = np.array([joint("multi_gru", seed, "full", delay) - joint("random_gru", seed, "full", delay)
                                      for seed in config["seeds"]])
                delta = gaps - full_gaps
                interactions[str(delay)][f"interaction_{payload}"] = paired_summary(delta, config["bootstrap_samples"], config["bootstrap_seed"])
    primary = interactions[str(primary_delay)]["interaction_blank"]
    full_gap = interactions[str(primary_delay)]["gap_full"]["mean"]
    blank_gap = interactions[str(primary_delay)]["gap_blank"]["mean"]
    block_full = np.mean([joint("multi_gru", seed, "full", primary_delay) for seed in config["seeds"]])
    control_full = np.mean([joint("random_gru", seed, "full", primary_delay) for seed in config["seeds"]])
    block_blank = np.mean([joint("multi_gru", seed, "blank", primary_delay) for seed in config["seeds"]])
    control_blank = np.mean([joint("random_gru", seed, "blank", primary_delay) for seed in config["seeds"]])
    block_gain = float(block_blank - block_full)
    control_gain = float(control_blank - control_full)
    dynamic = (config["minimum_dynamic_joint"] < block_full < config["maximum_dynamic_joint"] and
        config["minimum_dynamic_joint"] < control_full < config["maximum_dynamic_joint"] and
        block_blank > config["minimum_dynamic_joint"] and control_blank > config["minimum_dynamic_joint"] and
        full_gap <= config["minimum_original_gap"] and
        control_gain >= -config["max_control_blank_degradation"])
    low, high = primary["ci95"]
    cutoff = config["meaningful_interaction"]
    if not matched or not dynamic:
        decision = "INCONCLUSIVE"
    elif primary["mean"] >= cutoff and low > 0 and block_gain >= config["minimum_block_gain"]:
        decision = "SUPPORTED"
    elif primary["mean"] > 0 and low > 0:
        decision = "PARTIALLY_SUPPORTED"
    elif high < cutoff and low <= 0:
        decision = "FALSIFIED"
    else:
        decision = "INCONCLUSIVE"
    means = {str(delay): {name: {payload: {
        metric: float(np.mean([records[name, seed]["metrics"][f"{payload}_delay{delay}"][metric]
            for seed in config["seeds"]])) for metric in ["joint_pair_accuracy", "pre_accuracy", "confirm_accuracy",
                                                 "revise_accuracy", "pre_nll", "confirm_nll", "revise_nll", "evidence_sensitivity"]}
        for payload in config["payloads"]} for name in config["models"]} for delay in config["delays"]}
    report = {"decision": decision, "scope": "Frozen-weight EXP-0005 payload sensitivity at delay32 only",
        "complete": True, "provenance_and_matching_pass": bool(matched), "dynamic_range_pass": bool(dynamic),
        "primary": {"interaction": primary, "full_gap_mean": full_gap, "blank_gap_mean": blank_gap,
            "block_full_mean": float(block_full), "control_full_mean": float(control_full),
            "block_blank_mean": float(block_blank), "control_blank_mean": float(control_blank),
            "block_absolute_gain": block_gain, "control_absolute_gain": control_gain,
            "minimum_useful_gain_excluded": high < cutoff},
        "interactions": interactions, "mean_metrics": means,
        "secondary_status": "descriptive only; no secondary contrast changes the primary decision",
        "gate1_advanced": False, "next_branch": "STOP_MECHANISM_ESCALATION",
        "uncertainty": "Paired model-seed bootstrap conditional on fixed new test instances; reused EXP-0004 training seeds."}
    write_json(run / "analysis.json", report)
    lines = ["# EXP-0005 v2 results", "", f"Decision: **{decision}**. Scope: {report['scope']}.", "",
        f"Primary Δ (blank gap minus full gap) = {primary['mean']:.4f}, 95% paired-seed CI {primary['ci95']}.",
        f"Full M3−C0 gap = {full_gap:.4f}; blank gap = {blank_gap:.4f}; M3 absolute blank gain = {block_gain:.4f}.",
        f"Provenance/resource matching: {matched}; dynamic range: {dynamic}. Gate1 remains closed.", "",
        "| Delay | Payload | M3 joint | C0 joint | Wide GRU joint |", "|---|---|---:|---:|---:|"]
    for delay in config["delays"]:
        for payload in config["payloads"]:
            lines.append(f"| {delay} | {payload} | {means[str(delay)]['multi_gru'][payload]['joint_pair_accuracy']:.4f} | {means[str(delay)]['random_gru'][payload]['joint_pair_accuracy']:.4f} | {means[str(delay)]['gru_wide'][payload]['joint_pair_accuracy']:.4f} |")
    lines += ["", "Absolute pre/confirm/revise/NLL, every seed and exploratory contrasts are in analysis.json and per-model result.json.",
        "The interventions were absent from training. A positive interaction is payload sensitivity, not identification of pure decay or semantic hypotheses."]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), indent=2))


if __name__ == "__main__":
    main()
