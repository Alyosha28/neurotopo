"""Seed-level registered decision for frozen final-state revise-answer access."""
import argparse
import json
from pathlib import Path

import numpy as np

from .analyze import paired_summary
from .run import digest, source_fingerprints, write_json


def analyze(run: Path) -> dict:
    manifest=json.loads((run/"manifest.json").read_text())
    c=manifest["config"]
    if manifest["status"]!="COMPLETED" or manifest["run_kind"]!="REGISTERED" or manifest["seeds_run"]!=c["seeds"]:
        raise ValueError("Full registered EXP-0008 run required")
    frozen=Path(manifest["config_path"])
    if (manifest["experiment_id"]!=c["experiment_id"] or manifest["version"]!=c["version"] or
        digest(frozen.read_bytes())!=manifest["config_sha256"] or json.loads(frozen.read_text())!=c or
        source_fingerprints()!=manifest["source_sha256"] or
        not manifest["validation"]["pass"] or manifest["validation"]["source_sha256"]!=manifest["source_sha256"]):
        raise ValueError("Frozen config/source/validation mismatch")
    checkpoint_path=Path(c["checkpoint_manifest"])
    copies=json.loads(checkpoint_path.read_text())["copies"]
    source_path=Path("artifacts")/c["source_experiment"]/"manifest.json"
    if (digest(checkpoint_path.read_bytes())!=manifest["checkpoint_manifest_sha256"] or
        digest(source_path.read_bytes())!=manifest["source_run_manifest_sha256"] or
        not json.loads(source_path.read_text())["git_commit"].startswith(c["source_commit"])):
        raise ValueError("Frozen model-source input mismatch")
    rows={(name,seed):json.loads((run/f"seed-{seed}"/name/"result.json").read_text())
          for name in c["models"] for seed in c["seeds"]}
    matched=len(rows)==len(c["models"])*len(c["seeds"])
    ref=rows[c["models"][0],c["seeds"][0]]["delays"]
    resource_fields=["allocated_parameters","effective_parameters","hidden_state_floats",
        "dense_forward_linear_macs_per_example","active_forward_linear_macs_per_example",
        "internal_steps_per_input","mask_buffer_bytes","sequence_length"]
    for seed in c["seeds"]:
        block,control=rows["multi_gru",seed],rows["random_gru",seed]
        matched &= block["initial_parameters_sha256"]==control["initial_parameters_sha256"]
        matched &= block["mask_sha256"]!=control["mask_sha256"]
        for name in c["models"]:
            record=rows[name,seed]
            matched &= record["status"]=="COMPLETED" and record["source_replay"]=="EXACT"
            matched &= record["checkpoint_sha256"]==copies[str(seed)][name]["sha256"]
            matched &= set(record["delays"])=={str(delay) for delay in c["delays"]}
            for delay in c["delays"]:
                item=record["delays"][str(delay)]
                original=ref[str(delay)]
                matched &= item["datasets"]==original["datasets"]
                matched &= item["pair_split_verified"] is True and item["full_final_readout_parity_verified"] is True
                matched &= item["probe_parameters"]==260 and item["train_only_scaling"] is True
                matched &= item["ridge_lambda"]==c["ridge_lambda"]
                matched &= item["ridge_fit_macs_proxy"]==4*c["train_pairs"]*(65*65+65*4)
                matched &= item["original_behavior"]["pairs"]==c["test_pairs"]
                matched &= set(item["heads"])=={f"H{delay}A",f"H{delay}B"}
                for kind in ("train_a","train_b","validation","test"):
                    expected=c["train_pairs"] if kind.startswith("train_") else c[f"{kind}_pairs"]
                    matched &= item["datasets"][kind]["pairs"]==expected
                    matched &= item["datasets"][kind]["examples"]==2*expected
                for suffix in ("A","B"):
                    key=f"H{delay}{suffix}"
                    head=item["heads"][key]
                    matched &= head["parameters"]==260 and head["fit_pairs"]==c["train_pairs"]
                    matched &= head["fit_examples"]==2*c["train_pairs"] and head["ridge_lambda"]==c["ridge_lambda"]
                    matched &= head["normalizer_sha256"]==item["normalizer_sha256"]
                    file=run/f"seed-{seed}"/name/f"H{delay}{suffix}-final.npz"
                    matched &= digest(file.read_bytes())==head["file_sha256"]
                    with np.load(file,allow_pickle=False) as artifact:
                        matched &= artifact["coefficients"].shape==(65,4)
                        matched &= digest(artifact["coefficients"].tobytes())==head["coefficient_sha256"]
                        matched &= digest(artifact["mean"].tobytes()+artifact["scale"].tobytes())==item["normalizer_sha256"]
                if name=="random_gru":
                    matched &= all(item["resources"][f]==block["delays"][str(delay)]["resources"][f]
                                   for f in resource_fields)
                    matched &= item["state_extraction_forward_linear_flops_proxy"]==block["delays"][str(delay)]["state_extraction_forward_linear_flops_proxy"]
                    matched &= item["ridge_fit_macs_proxy"]==block["delays"][str(delay)]["ridge_fit_macs_proxy"]

    def revised(model: str,seed: int,delay: int,head: str,split: str) -> float:
        return rows[model,seed]["delays"][str(delay)]["heads"][f"H{delay}{head}"][split]["revise_accuracy"]

    delay=c["primary_delay"]
    delta=np.array([revised("multi_gru",seed,delay,"A","test")-revised("random_gru",seed,delay,"A","test")
                    for seed in c["seeds"]])
    primary=paired_summary(delta,c["bootstrap_samples"],c["bootstrap_seed"])
    adequate={}
    stability={}
    for name in c["models"]:
        scores={suffix:[revised(name,seed,delay,suffix,"validation") for seed in c["seeds"]]
                for suffix in ("A","B")}
        adequate[name]={suffix:{"mean":float(np.mean(values)),
            "seeds_at_or_above_threshold":sum(value>=c["adequate_seed_validation_revise"] for value in values)}
            for suffix,values in scores.items()}
        stability[name]={"per_seed_abs": [abs(revised(name,seed,delay,"A","test")-
                                         revised(name,seed,delay,"B","test")) for seed in c["seeds"]]}
        stability[name]["mean_abs"]=float(np.mean(stability[name]["per_seed_abs"]))
    adequate_ok=all(adequate[name][suffix]["mean"]>=c["adequate_mean_validation_revise"] and
        adequate[name][suffix]["seeds_at_or_above_threshold"]>=c["adequate_seed_count"]
        for name in ("random_gru","gru_wide") for suffix in ("A","B"))
    stable=all(stability[name]["mean_abs"]<=c["maximum_refit_variation"] for name in ("multi_gru","random_gru"))
    original={name:float(np.mean([rows[name,seed]["delays"][str(delay)]["original_behavior"]["revise_accuracy"]
        for seed in c["seeds"]])) for name in c["models"]}
    original_gap=original["multi_gru"]-original["random_gru"]
    behavior_ok=(original_gap<=c["minimum_behavior_gap"] and
                 all(original[name]>c["minimum_behavior_revise"] for name in ("multi_gru","random_gru")))
    low,high=primary["ci95"]
    cutoff=-c["minimum_linear_deficit"]
    if not (matched and adequate_ok and stable and behavior_ok):
        decision="INCONCLUSIVE"
    elif high<cutoff:
        decision="SUPPORTED"
    elif high<0:
        decision="PARTIALLY_SUPPORTED"
    elif low>cutoff:
        decision="FALSIFIED"
    else:
        decision="INCONCLUSIVE"
    means={str(d):{name:{suffix:{split:{metric:float(np.mean([rows[name,seed]["delays"][str(d)]["heads"][f"H{d}{suffix}"][split][metric]
               for seed in c["seeds"]])) for metric in ("revise_accuracy","confirm_accuracy","all_accuracy","both_pair_accuracy","mean_squared_error")}
               for split in ("train","validation","test")}
               for suffix in ("A","B")} for name in c["models"]} for d in c["delays"]}
    original_by_delay={str(d):{name:{metric:float(np.mean([rows[name,seed]["delays"][str(d)]["original_behavior"][metric]
       for seed in c["seeds"]])) for metric in ("pre_accuracy","confirm_accuracy","revise_accuracy","joint_pair_accuracy")}
       for name in c["models"]} for d in c["delays"]}
    secondary={str(d):paired_summary(np.array([revised("multi_gru",seed,d,"A","test")-
        revised("random_gru",seed,d,"A","test") for seed in c["seeds"]]),c["bootstrap_samples"],c["bootstrap_seed"])
        for d in c["delays"] if d!=delay}
    report={"decision":decision,"scope":"Frozen final-state revise-answer linear access, delay32",
        "complete":True,"provenance_and_matching_pass":bool(matched),
        "positive_control_pass":bool(adequate_ok),"refit_stability_pass":bool(stable),
        "original_revise_behavior_guard_pass":bool(behavior_ok),
        "primary_difference":primary,"secondary_differences":secondary,
        "positive_control_validation":adequate,"refit_stability":stability,
        "original_revise_means":original,"original_revise_gap":original_gap,
        "mean_probe_scores":means,"original_behavior_by_delay":original_by_delay,
        "minimum_useful_deficit_excluded":low>cutoff,
        "gate1_advanced":False,"next_branch":"STOP_MECHANISM_ESCALATION",
        "uncertainty":"Paired trained-model seed bootstrap conditional on fixed full-noise test pairs; newly supervised linear head, no model-weight update."}
    write_json(run/"analysis.json",report)
    lines=["# EXP-0008 v1 results","",f"Decision: **{decision}**. Scope: {report['scope']}.","",
           f"Primary M3−C0 revise linear-decoding difference: {primary['mean']:.4f}; paired-seed95%CI {primary['ci95']}.",
           f"Provenance/matching: {matched}; positive controls: {adequate_ok}; refit: {stable}; behavior guard: {behavior_ok}.",
           "","| Delay | Model | Probe revise test | Probe confirm test | Original revise |", "|---|---|---:|---:|---:|"]
    for d in c["delays"]:
        for name in c["models"]:
            row=means[str(d)][name]["A"]["test"]
            lines.append(f"| {d} | {name} | {row['revise_accuracy']:.4f} | {row['confirm_accuracy']:.4f} | {original_by_delay[str(d)][name]['revise_accuracy']:.4f} |")
    lines += ["", "Full per-seed/probe scores, validation adequacy, A/B refit and resource records are in analysis.json and result.json.",
              "A fitted head's linear accuracy is not the original task head's computation or proof of semantic hypotheses. Gate1 remains closed."]
    (run/"report.md").write_text("\n".join(lines)+"\n")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    print(json.dumps(analyze(parser.parse_args().run),indent=2))
