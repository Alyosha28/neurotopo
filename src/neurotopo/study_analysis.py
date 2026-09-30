"""Explicit registered decisions for staged studies; never score-maximize."""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze import paired_summary
from .run import write_json


def analyze(run: Path) -> dict[str, Any]:
    m = json.loads((run / "manifest.json").read_text())
    c = m["config"]
    if m["status"] != "COMPLETED" or m["run_kind"] != "REGISTERED" or m["seeds_run"] != c["seeds"]:
        raise ValueError("Full registered study required")
    if not m["validation"]["pass"] or m["validation"]["source_sha256"] != m["source_sha256"]:
        raise ValueError("Validation mismatch")
    results = {(a["name"], s): json.loads((run / f"seed-{s}" / a["name"] / "result.json").read_text()) for a in c["arms"] for s in c["seeds"]}
    rows = []
    for (name, seed), r in results.items():
        if r["status"] != "COMPLETED":
            raise ValueError("Incomplete arm")
        for dataset, metrics in r["metrics"].items():
            rows.append({"arm": name, "seed": seed, "dataset": dataset, **metrics})
    def feasible(name: str, dataset: str) -> bool:
        return all(results[name, s]["metrics"][dataset]["pre_accuracy"] >= c["minimum_pre_accuracy"] and
                   results[name, s]["metrics"][dataset]["joint_pair_accuracy"] >= c["minimum_joint_accuracy"] for s in c["seeds"])
    comparisons = {}
    if c["analysis"] == "learnability_factorial":
        feasibility = {a["name"]: feasible(a["name"], f"{a['payload']}_delay8") for a in c["arms"]}
        gains = [results["noise_curriculum", s]["metrics"]["noise_delay8"]["joint_pair_accuracy"] -
                 results["noise_direct", s]["metrics"]["noise_delay8"]["joint_pair_accuracy"] for s in c["seeds"]]
        decision = "SUPPORTED" if feasibility["noise_curriculum"] and not feasibility["noise_direct"] and min(gains) >= c["minimum_recovery_gain"] else ("PARTIALLY_SUPPORTED" if any(feasibility.values()) else "INCONCLUSIVE")
        comparisons["noise_curriculum_minus_direct_per_seed"] = gains
        branch = "MATCHED_ARCHITECTURE_CONFIRMATION" if feasibility["noise_curriculum"] else "FURTHER_LEARNABILITY_DIAGNOSIS"
        matching = True
        for s in c["seeds"]:
            for a in c["arms"]:
                r = results[a["name"], s]
                matching &= r["train_updates"] == 6000 and r["train_examples"] == 6000 * m["base_config"]["batch_size"]
                matching &= r["dataset_sha256"] == results["noise_direct", s]["dataset_sha256"]
            for schedule in ["direct", "curriculum"]:
                a, b = results[f"noise_{schedule}", s], results[f"blank_{schedule}", s]
                matching &= all(a[k] == b[k] for k in ["initial_parameters_sha256", "train_tokens", "training_forward_linear_flops", "resources"])
        contrasts = [("blank_direct", "noise_direct"), ("blank_curriculum", "noise_curriculum"),
                     ("noise_curriculum", "noise_direct"), ("blank_curriculum", "blank_direct")]
        comparisons["common_test_per_seed_differences"] = {
            dataset: {metric: {f"{a}_minus_{b}": [results[a, s]["metrics"][dataset][metric] - results[b, s]["metrics"][dataset][metric] for s in c["seeds"]] for a,b in contrasts}
                      for metric in ["pre_accuracy", "joint_pair_accuracy", "revise_accuracy", "revise_nll"]}
            for dataset in results["noise_direct", c["seeds"][0]]["metrics"]}
    elif c["analysis"] == "matched_architecture":
        fields = ["effective_parameters", "allocated_parameters", "hidden_state_floats", "dense_forward_linear_macs_per_example", "active_forward_linear_macs_per_example"]
        matching = True
        for s in c["seeds"]:
            a, b = results["multi_gru", s], results["random_gru", s]
            matching &= a["dataset_sha256"] == b["dataset_sha256"]
            matching &= a["initial_parameters_sha256"] == b["initial_parameters_sha256"]
            matching &= all(results[arm["name"], s]["dataset_sha256"] == a["dataset_sha256"] for arm in c["arms"])
            for arm in c["arms"]:
                expected_updates = sum(count for _, count in arm["schedule"])
                matching &= results[arm["name"], s]["train_updates"] == expected_updates
                matching &= results[arm["name"], s]["train_examples"] == expected_updates * m["base_config"]["batch_size"]
            matching &= all(a[k] == b[k] for k in ["train_examples", "train_tokens", "train_updates", "training_forward_linear_flops"])
            matching &= a["graph"]["diagonal_connections"] == b["graph"]["diagonal_connections"]
            matching &= all(a["resources"][d][f] == b["resources"][d][f] for d in a["resources"] for f in fields)
        feasibility = {a["name"]: feasible(a["name"], "noise_delay8") for a in c["arms"]}
        for delay in c["eval_delays"]:
            key = f"noise_delay{delay}"
            comparisons[key] = {}
            for baseline in ["random_gru", "gru_wide"]:
                delta = np.array([results["multi_gru", s]["metrics"][key]["joint_pair_accuracy"] - results[baseline, s]["metrics"][key]["joint_pair_accuracy"] for s in c["seeds"]])
                comparisons[key][baseline] = paired_summary(delta, c["bootstrap_samples"], c["bootstrap_seed"])
        targets = [c["primary_delay"], c["robustness_delay"]]
        primary = comparisons[f"noise_delay{c['primary_delay']}"]["random_gru"]
        learned = feasibility["multi_gru"] and feasibility["random_gru"]
        floor = all(np.mean([results[n,s]["metrics"][f"noise_delay{c['primary_delay']}"]["joint_pair_accuracy"] for s in c["seeds"]]) < c["primary_floor"] for n in ["multi_gru","random_gru"])
        positive = all(comparisons[f"noise_delay{d}"]["random_gru"]["mean"] >= c["minimum_effect"] and comparisons[f"noise_delay{d}"]["random_gru"]["ci95"][0] > 0 for d in targets)
        confirm_ok = all(np.mean([results["multi_gru", s]["metrics"][f"noise_delay{d}"]["confirm_accuracy"] - results["random_gru", s]["metrics"][f"noise_delay{d}"]["confirm_accuracy"] for s in c["seeds"]]) >= -c["max_confirm_degradation"] for d in targets)
        wide = all(comparisons[f"noise_delay{d}"]["gru_wide"]["mean"] >= c["minimum_effect"] for d in targets)
        comparisons["guards"] = {"iid_learned": learned, "primary_floor": bool(floor), "confirm_guardrail": bool(confirm_ok)}
        decision = "INCONCLUSIVE" if not learned or not matching or floor else ("SUPPORTED" if positive and confirm_ok and wide else "PARTIALLY_SUPPORTED" if positive and confirm_ok else "FALSIFIED" if primary["ci95"][1] < c["minimum_effect"] else "INCONCLUSIVE")
        branch = "STAGE_GATE_REVIEW" if decision == "SUPPORTED" else "STOP_MECHANISM_ESCALATION"
    else:
        raise ValueError(c["analysis"])
    if not matching:
        decision = "INCONCLUSIVE"
        branch = "REPAIR_INVALID_EVIDENCE"
    report = {"decision": decision, "analysis": c["analysis"], "feasibility": feasibility,
              "resource_matching_pass": bool(matching), "comparisons": comparisons,
              "next_branch": branch, "gate1_automatically_advanced": False,
              "rows": rows, "wall_seconds": m["wall_seconds"]}
    write_json(run / "analysis.json", report)
    lines = [f"# {c['experiment_id']} {c['version']} results", "", f"Decision: **{decision}**; scope: {c['analysis']}.", "", "| Arm | Dataset | Pre accuracy mean | Joint accuracy mean |", "|---|---|---:|---:|"]
    for arm in c["arms"]:
        for dataset in results[arm["name"], c["seeds"][0]]["metrics"]:
            values = [r for r in rows if r["arm"] == arm["name"] and r["dataset"] == dataset]
            lines.append(f"| {arm['name']} | {dataset} | {np.mean([r['pre_accuracy'] for r in values]):.4f} | {np.mean([r['joint_pair_accuracy'] for r in values]):.4f} |")
    lines += ["", f"Registered next branch: {branch}. No automatic gate advancement.", "", "See analysis.json for every seed and comparisons; resource/timing records and monitoring curves remain in per-arm result.json."]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    print(json.dumps(analyze(parser.parse_args().run), indent=2))
