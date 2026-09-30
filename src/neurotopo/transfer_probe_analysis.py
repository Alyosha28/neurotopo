"""Paired-seed cross-delay transfer-loss decision for EXP-0007."""
import argparse
import json
from pathlib import Path

import numpy as np

from .analyze import paired_summary
from .run import digest, source_fingerprints, write_json


def analyze(run: Path) -> dict:
    manifest=json.loads((run/"manifest.json").read_text())
    config=manifest["config"]
    if manifest["status"]!="COMPLETED" or manifest["run_kind"]!="REGISTERED" or manifest["seeds_run"]!=config["seeds"]:
        raise ValueError("Full registered EXP-0007 run required")
    frozen_config=Path(manifest["config_path"])
    if (manifest["experiment_id"]!=config["experiment_id"] or manifest["version"]!=config["version"] or
        digest(frozen_config.read_bytes())!=manifest["config_sha256"] or
        json.loads(frozen_config.read_text())!=config or
        source_fingerprints()!=manifest["source_sha256"] or
        not manifest["validation"]["pass"] or
        manifest["validation"]["source_sha256"]!=manifest["source_sha256"]):
        raise ValueError("Frozen config/source/validation mismatch")
    copies_path=Path(config["checkpoint_manifest"])
    copies=json.loads(copies_path.read_text())["copies"]
    source_path=Path("artifacts")/config["source_experiment"]/"manifest.json"
    if (digest(copies_path.read_bytes())!=manifest["checkpoint_manifest_sha256"] or
        digest(source_path.read_bytes())!=manifest["source_run_manifest_sha256"] or
        not json.loads(source_path.read_text())["git_commit"].startswith(config["source_commit"])):
        raise ValueError("Source checkpoint/experiment mismatch")
    records={(name,seed):json.loads((run/f"seed-{seed}"/name/"result.json").read_text())
             for name in config["models"] for seed in config["seeds"]}
    matching=len(records)==len(config["models"])*len(config["seeds"])
    reference=records[config["models"][0],config["seeds"][0]]["datasets"]
    expected_data={f"{kind}{delay}" for delay in config["fit_delays"] for kind in ("A","B","V")}
    expected_data.update(f"T{delay}" for delay in config["test_delays"])
    expected_heads={f"H{delay}{kind}" for delay in config["fit_delays"] for kind in ("A","B")}
    resource_fields=["allocated_parameters","effective_parameters","hidden_state_floats",
        "dense_forward_linear_macs_per_example","active_forward_linear_macs_per_example",
        "internal_steps_per_input","mask_buffer_bytes","sequence_length"]
    for seed in config["seeds"]:
        block,control=records["multi_gru",seed],records["random_gru",seed]
        matching &= block["initial_parameters_sha256"]==control["initial_parameters_sha256"]
        matching &= block["mask_sha256"]!=control["mask_sha256"]
        for name in config["models"]:
            row=records[name,seed]
            matching &= row["status"]=="COMPLETED" and row["source_replay"]=="EXACT"
            matching &= row["checkpoint_sha256"]==copies[str(seed)][name]["sha256"]
            matching &= set(row["datasets"])==expected_data and row["datasets"]==reference
            matching &= set(row["heads"])==expected_heads and row["head_capacity_parameters"]==260
            matching &= row["train_only_delay8_scaling"] is True
            for data_name in expected_data:
                expected=config["test_pairs"] if data_name.startswith("T") else (config["validation_pairs"] if data_name.startswith("V") else config["train_pairs"])
                matching &= row["datasets"][data_name]["unique_prefixes"]==expected
            for head_name,head in row["heads"].items():
                matching &= head["parameters"]==260 and head["fit_unique_prefixes"]==config["train_pairs"]
                matching &= head["ridge_lambda"]==config["ridge_lambda"]
                matching &= head["short_train_normalizer_sha256"]==row["normalizer_sha256"]
                file=run/f"seed-{seed}"/name/f"{head_name}.npz"
                matching &= digest(file.read_bytes())==head["file_sha256"]
                with np.load(file,allow_pickle=False) as artifact:
                    matching &= artifact["coefficients"].shape==(65,4)
                    matching &= digest(artifact["coefficients"].tobytes())==head["coefficient_sha256"]
                    matching &= digest(artifact["mean"].tobytes()+artifact["scale"].tobytes())==row["normalizer_sha256"]
            for delay in config["test_delays"]:
                test=row["test"][str(delay)]
                matching &= test["dataset_sha256"]==reference[f"T{delay}"]["task_sha256"]
                matching &= test["behavior"]["pairs"]==config["test_pairs"]
                if name=="random_gru":
                    matching &= all(test["resources"][f]==block["test"][str(delay)]["resources"][f] for f in resource_fields)
        matching &= block["state_extraction_forward_linear_flops_proxy"]==control["state_extraction_forward_linear_flops_proxy"]

    def accuracy(model: str,seed: int,head: str,delay: int) -> float:
        return records[model,seed]["heads"][head][f"test_delay{delay}"]["accuracy"]

    primary_delay=config["primary_delay"]
    losses={name: [accuracy(name,seed,f"H{primary_delay}A",primary_delay)-accuracy(name,seed,"H8A",primary_delay)
                   for seed in config["seeds"]] for name in config["models"]}
    delta=np.array(losses["multi_gru"])-np.array(losses["random_gru"])
    primary=paired_summary(delta,config["bootstrap_samples"],config["bootstrap_seed"])
    source_quality={}
    target_quality={}
    refit={}
    for name in config["models"]:
        source_quality[name]={suffix:[records[name,seed]["heads"][f"H8{suffix}"]["validation_accuracy"]
            for seed in config["seeds"]] for suffix in ("A","B")}
        target_quality[name]={suffix:[records[name,seed]["heads"][f"H{primary_delay}{suffix}"]["validation_accuracy"]
            for seed in config["seeds"]] for suffix in ("A","B")}
        refit[name]={"source_test_mean_abs":float(np.mean([abs(accuracy(name,seed,"H8A",primary_delay)-accuracy(name,seed,"H8B",primary_delay)) for seed in config["seeds"]])),
            "target_test_mean_abs":float(np.mean([abs(accuracy(name,seed,f"H{primary_delay}A",primary_delay)-accuracy(name,seed,f"H{primary_delay}B",primary_delay)) for seed in config["seeds"]]))}
    def enough(name: str) -> bool:
        return all(value>=config["minimum_validation_accuracy"] for suffix in ("A","B")
            for value in source_quality[name][suffix]+target_quality[name][suffix])
    adequate=all(enough(name) for name in ("multi_gru","random_gru")) and all(np.mean(source_quality["gru_wide"][suffix])>=config["minimum_validation_accuracy"] and np.mean(target_quality["gru_wide"][suffix])>=config["minimum_validation_accuracy"] for suffix in ("A","B"))
    stable=all(refit[name]["source_test_mean_abs"]<=config["maximum_source_refit_variation"] and
               refit[name]["target_test_mean_abs"]<=config["maximum_refit_variation"]
               for name in ("multi_gru","random_gru"))
    behavior={name:float(np.mean([records[name,seed]["test"][str(primary_delay)]["behavior"]["pre_accuracy"]
        for seed in config["seeds"]])) for name in config["models"]}
    behavioral_gap=behavior["multi_gru"]-behavior["random_gru"]
    behavior_ok=behavioral_gap<=config["minimum_behavior_gap"] and all(behavior[name]>config["minimum_behavior_pre"] for name in ("multi_gru","random_gru"))
    low,high=primary["ci95"]
    cutoff=config["minimum_excess_transfer_loss"]
    if not (matching and adequate and stable and behavior_ok):
        decision="INCONCLUSIVE"
    elif low>cutoff:
        decision="SUPPORTED"
    elif low>0:
        decision="PARTIALLY_SUPPORTED"
    elif high<cutoff:
        decision="FALSIFIED"
    else:
        decision="INCONCLUSIVE"
    secondary={}
    for delay in config["test_delays"]:
        secondary[str(delay)]={}
        for name in config["models"]:
            secondary[str(delay)][name]={"short_A_mean":float(np.mean([accuracy(name,seed,"H8A",delay) for seed in config["seeds"]])),
                "short_B_mean":float(np.mean([accuracy(name,seed,"H8B",delay) for seed in config["seeds"]])),
                "matched_A_mean":float(np.mean([accuracy(name,seed,f"H{delay}A",delay) for seed in config["seeds"]])),
                "matched_B_mean":float(np.mean([accuracy(name,seed,f"H{delay}B",delay) for seed in config["seeds"]]))}
    report={"decision":decision,"scope":"Fixed-ridge cross-delay prior-label readout transfer at delay32",
        "complete":True,"provenance_and_matching_pass":bool(matching),"probe_adequacy_pass":bool(adequate),
        "head_refit_stability_pass":bool(stable),"behavioral_guard_pass":bool(behavior_ok),
        "primary_difference":primary,"transfer_losses_per_seed":losses,
        "refit_variation":refit,"source_validation_scores":source_quality,
        "target_validation_scores":target_quality,"behavioral_pre_means":behavior,
        "behavioral_pre_gap":behavioral_gap,"secondary_absolute_scores":secondary,
        "minimum_excess_transfer_loss_excluded":high<cutoff,
        "gate1_advanced":False,"next_branch":"STOP_MECHANISM_ESCALATION",
        "uncertainty":"Paired trained-seed percentile bootstrap on one fixed new test set; shared delay8-A scaler, frozen models, linear probe family only."}
    write_json(run/"analysis.json",report)
    lines=["# EXP-0007 v2 results","",f"Decision: **{decision}**. Scope: {report['scope']}.","",
        f"Primary excess transfer loss M3−C0: {primary['mean']:.4f}, paired-seed95%CI {primary['ci95']}.",
        f"Provenance: {matching}; validation adequacy: {adequate}; source/target head stability: {stable}; original pre-gap guard: {behavior_ok}.",
        "","| Delay | Model | H8A→test | H8B→test | matched A→test | original pre |",
        "|---|---|---:|---:|---:|---:|"]
    for delay in config["test_delays"]:
        for name in config["models"]:
            row=secondary[str(delay)][name]
            original=float(np.mean([records[name,seed]["test"][str(delay)]["behavior"]["pre_accuracy"] for seed in config["seeds"]]))
            lines.append(f"| {delay} | {name} | {row['short_A_mean']:.4f} | {row['short_B_mean']:.4f} | {row['matched_A_mean']:.4f} | {original:.4f} |")
    lines += ["","All seed/head scores, source and target validation, refit controls and resource records are in analysis.json and result.json.",
              "This is fixed-probe transfer, not a causal explanation of the model's pretrained readout or pure temporal decay. Gate1 stays closed."]
    (run/"report.md").write_text("\n".join(lines)+"\n")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    print(json.dumps(analyze(parser.parse_args().run),indent=2))
