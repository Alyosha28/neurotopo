"""Registered seed-level analysis of uncued linear information access."""
import argparse
import json
from pathlib import Path

import numpy as np

from .analyze import paired_summary
from .run import digest, source_fingerprints, write_json


def analyze(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    c = manifest["config"]
    if manifest["status"] != "COMPLETED" or manifest["run_kind"] != "REGISTERED" or manifest["seeds_run"] != c["seeds"]:
        raise ValueError("Complete registered EXP-0006 run required")
    path = Path(manifest["config_path"])
    if (manifest["experiment_id"] != c["experiment_id"] or manifest["version"] != c["version"] or
        digest(path.read_bytes()) != manifest["config_sha256"] or json.loads(path.read_text()) != c or
        manifest["source_sha256"] != source_fingerprints() or
        not manifest["validation"]["pass"] or
        manifest["validation"]["source_sha256"] != manifest["source_sha256"]):
        raise ValueError("Frozen config/source/validation mismatch")
    checkpoint_manifest_path = Path(c["checkpoint_manifest"])
    inputs = json.loads(checkpoint_manifest_path.read_text())["copies"]
    source_path = Path("artifacts") / c["source_experiment"] / "manifest.json"
    if (manifest["checkpoint_manifest_sha256"] != digest(checkpoint_manifest_path.read_bytes()) or
        manifest["source_run_manifest_sha256"] != digest(source_path.read_bytes()) or
        not json.loads(source_path.read_text())["git_commit"].startswith(c["source_commit"])):
        raise ValueError("Frozen checkpoint/source experiment mismatch")
    records = {(name, seed): json.loads((run / f"seed-{seed}" / name / "result.json").read_text())
               for name in c["models"] for seed in c["seeds"]}
    matching = len(records) == len(c["models"]) * len(c["seeds"])
    data_reference = records[c["models"][0], c["seeds"][0]]["delays"]
    resources = ["allocated_parameters", "effective_parameters", "hidden_state_floats",
                 "dense_forward_linear_macs_per_example", "active_forward_linear_macs_per_example",
                 "internal_steps_per_input", "mask_buffer_bytes", "sequence_length"]
    for seed in c["seeds"]:
        block, control = records["multi_gru", seed], records["random_gru", seed]
        matching &= block["initial_parameters_sha256"] == control["initial_parameters_sha256"]
        matching &= block["mask_sha256"] != control["mask_sha256"]
        for name in c["models"]:
            record = records[name, seed]
            matching &= record["status"] == "COMPLETED" and record["source_replay"] == "EXACT"
            matching &= record["checkpoint_sha256"] == inputs[str(seed)][name]["sha256"]
            matching &= set(record["delays"]) == {str(delay) for delay in c["delays"]}
            for delay in c["delays"]:
                d = record["delays"][str(delay)]
                matching &= d["probe_parameters"] == 1040 and d["train_only_scaling"] is True
                matching &= d["split_unique_prefix_counts"] == {split: c[f"{split}_pairs"]
                    for split in ("train", "validation", "test")}
                matching &= d["ridge_lambda"] == c["ridge_lambda"] and d["std_floor"] == c["std_floor"]
                matching &= d["datasets"] == data_reference[str(delay)]["datasets"]
                matching &= all(d["scores"][split]["uncued_accuracy"] >= 0 and
                                d["scores"][split]["uncued_accuracy"] <= 1 for split in ("train","validation","test"))
                matching &= len(d["test_pair_bits"]["early"]) == len(d["test_pair_bits"]["late"]) == c["test_pairs"]
                early = np.asarray(d["test_pair_bits"]["early"], dtype=np.uint8)
                late = np.asarray(d["test_pair_bits"]["late"], dtype=np.uint8)
                matching &= digest(early.tobytes()) == d["test_pair_early_sha256"]
                matching &= digest(late.tobytes()) == d["test_pair_late_sha256"]
                matching &= np.isclose(np.mean(early), d["behavior"]["pre_accuracy"], atol=1e-8)
                matching &= np.isclose(np.mean(early & late), d["behavior"]["joint_pair_accuracy"], atol=1e-8)
                probe_path = run / f"seed-{seed}" / name / f"delay-{delay}-probe.npz"
                matching &= digest(probe_path.read_bytes()) == d["probe_file_sha256"]
                with np.load(probe_path, allow_pickle=False) as artifact:
                    matching &= artifact["coefficients"].shape == (65,16)
                    matching &= digest(artifact["coefficients"].tobytes()) == d["probe_coefficients_sha256"]
                if name == "random_gru":
                    matching &= all(block["delays"][str(delay)]["resources"][f] ==
                                    control["delays"][str(delay)]["resources"][f] for f in resources)
                    matching &= (block["delays"][str(delay)]["state_extraction_forward_linear_flops_proxy"] ==
                                 control["delays"][str(delay)]["state_extraction_forward_linear_flops_proxy"])

    def score(model: str, seed: int, delay: int, split: str, metric: str) -> float:
        return records[model, seed]["delays"][str(delay)]["scores"][split][metric]

    primary_delay = c["primary_delay"]
    primary_diff = np.array([score("multi_gru", seed, primary_delay, "test", "uncued_accuracy") -
                             score("random_gru", seed, primary_delay, "test", "uncued_accuracy") for seed in c["seeds"]])
    primary = paired_summary(primary_diff, c["bootstrap_samples"], c["bootstrap_seed"])
    adequacy = {}
    for model in ("random_gru", "gru_wide"):
        scores = [score(model, seed, primary_delay, "validation", "uncued_accuracy") for seed in c["seeds"]]
        adequacy[model] = {"mean": float(np.mean(scores)), "seeds_at_or_above_threshold":
                           sum(value >= c["adequate_seed_validation_uncued"] for value in scores)}
    adequate = all(row["mean"] >= c["adequate_mean_validation_uncued"] and
                   row["seeds_at_or_above_threshold"] >= c["adequate_seed_count"] for row in adequacy.values())
    mean_behavior = {name: float(np.mean([records[name, seed]["delays"][str(primary_delay)]["behavior"]["joint_pair_accuracy"]
                    for seed in c["seeds"]])) for name in c["models"]}
    behavior_gap = mean_behavior["multi_gru"] - mean_behavior["random_gru"]
    behavior_ok = (behavior_gap <= c["minimum_behavior_gap"] and
                   all(mean_behavior[name] > c["minimum_behavior_joint"] for name in ("multi_gru", "random_gru")))
    low, high = primary["ci95"]
    cutoff = -c["minimum_uncued_deficit"]
    if not matching or not adequate or not behavior_ok:
        decision = "INCONCLUSIVE"
    elif high < cutoff:
        decision = "SUPPORTED"
    elif high < 0:
        decision = "PARTIALLY_SUPPORTED"
    elif low > cutoff:
        decision = "FALSIFIED"
    else:
        decision = "INCONCLUSIVE"
    common = {}
    for delay in c["delays"]:
        rows = []
        for seed in c["seeds"]:
            block = records["multi_gru", seed]["delays"][str(delay)]["test_pair_bits"]
            control = records["random_gru", seed]["delays"][str(delay)]["test_pair_bits"]
            overlap = np.asarray(block["early"], dtype=bool) & np.asarray(control["early"], dtype=bool)
            n = int(overlap.sum())
            row = {"seed": seed, "common_early_correct_pairs": n,
                "block_final_pair_accuracy": float(np.mean(np.asarray(block["late"], dtype=bool)[overlap])) if n >= c["minimum_common_early_pairs"] else None,
                "control_final_pair_accuracy": float(np.mean(np.asarray(control["late"], dtype=bool)[overlap])) if n >= c["minimum_common_early_pairs"] else None}
            rows.append(row)
        common[str(delay)] = rows
    means = {str(delay): {name: {split: {metric: float(np.mean([score(name, seed, delay, split, metric)
             for seed in c["seeds"]])) for metric in ("uncued_accuracy", "prior_accuracy",
             "all_candidate_accuracy", "all_four_correct", "mean_squared_error")}
             for split in ("train", "validation", "test")} for name in c["models"]} for delay in c["delays"]}
    secondary_delay = [delay for delay in c["delays"] if delay != primary_delay]
    secondary_diffs = {str(delay): paired_summary(np.array([score("multi_gru", seed, delay, "test", "uncued_accuracy") -
                       score("random_gru", seed, delay, "test", "uncued_accuracy") for seed in c["seeds"]]),
                       c["bootstrap_samples"], c["bootstrap_seed"]) for delay in secondary_delay}
    report = {"decision": decision, "scope": "Frozen-state linear accessibility of uncued candidate values, delay32",
        "complete": True, "provenance_and_matching_pass": bool(matching), "probe_adequacy_pass": bool(adequate),
        "probe_adequacy": adequacy, "behavioral_guard_pass": bool(behavior_ok),
        "behavioral_joint_means": mean_behavior, "behavioral_gap": behavior_gap,
        "primary_difference": primary, "secondary_differences": secondary_diffs,
        "minimum_useful_deficit_excluded": low > cutoff,
        "mean_scores": means, "common_early_correct_pair_analysis": common,
        "conditional_baseline_for_uncued_with_prior_label": 1/3,
        "gate1_advanced": False, "next_branch": "STOP_MECHANISM_ESCALATION",
        "uncertainty": "Paired trained-seed bootstrap conditional on fixed new test set and fixed linear decoder family; observational probe only."}
    write_json(run / "analysis.json", report)
    lines = ["# EXP-0006 v2 results", "", f"Decision: **{decision}**. Scope: {report['scope']}.", "",
        f"Primary M3−C0 uncued linear-decoding effect: {primary['mean']:.4f}, paired-seed 95% CI {primary['ci95']}.",
        f"Probe adequacy: {adequate}; original-task gap guard: {behavior_ok}; provenance/matching: {matching}.",
        "", "| Delay | Model | Uncued test accuracy | Prior test accuracy | All-four exact |",
        "|---|---|---:|---:|---:|"]
    for delay in c["delays"]:
        for name in c["models"]:
            row = means[str(delay)][name]["test"]
            lines.append(f"| {delay} | {name} | {row['uncued_accuracy']:.4f} | {row['prior_accuracy']:.4f} | {row['all_four_correct']:.4f} |")
    lines += ["", "All seeds, train/validation scores, behavior and common early-correct subsets are retained in analysis.json and per-model records.",
        "This is linear accessibility, not total stored information, semantic specialization or a causal mediation finding. Gate1 stays closed."]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    print(json.dumps(analyze(parser.parse_args().run), indent=2))


if __name__ == "__main__":
    main()
