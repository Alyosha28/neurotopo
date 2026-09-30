"""Frozen transfer-decision gates on synthetic records only."""
import json
from pathlib import Path

import numpy as np
import pytest

from neurotopo.run import digest, source_fingerprints
from neurotopo.transfer_probe_analysis import analyze


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("case,expected", [
    ("excess", "SUPPORTED"), ("small_positive", "PARTIALLY_SUPPORTED"),
    ("no_excess", "FALSIFIED"), ("source_refit_unstable", "INCONCLUSIVE"),
    ("target_refit_unstable", "INCONCLUSIVE"), ("target_specific_cutoff", "INCONCLUSIVE"),
    ("source_specific_cutoff", "INCONCLUSIVE"), ("weak_validation", "INCONCLUSIVE"),
    ("behavior_gap_missing", "INCONCLUSIVE"), ("data_mismatch", "INCONCLUSIVE"),
    ("tampered_config", "REJECT")])
def test_transfer_decision_requires_stable_shared_evidence(tmp_path: Path, case: str, expected: str):
    names=["multi_gru","random_gru","gru_wide"]
    seeds=[1,2]
    cfg={"experiment_id":"SYNTHETIC","version":"v2","models":names,"seeds":seeds,
        "source_experiment":"EXP-0004/v1","source_commit":"eec9ca4",
        "checkpoint_manifest":str(tmp_path/"checkpoints.json"),
        "fit_delays":[8,32,64],"test_delays":[32,64],"primary_delay":32,
        "train_pairs":20,"validation_pairs":10,"test_pairs":10,
        "ridge_lambda":1.0,"minimum_validation_accuracy":.90,
        "maximum_source_refit_variation":.02,"maximum_refit_variation":.02,
        "minimum_behavior_pre":.10,
        "minimum_behavior_gap":-.01,"minimum_excess_transfer_loss":.05,
        "bootstrap_samples":100,"bootstrap_seed":1}
    if case=="target_specific_cutoff":
        cfg["maximum_source_refit_variation"]=.03
        cfg["maximum_refit_variation"]=.005
    if case=="source_specific_cutoff":
        cfg["maximum_source_refit_variation"]=.005
        cfg["maximum_refit_variation"]=.03
    checkpoints={str(s):{name:{"sha256":f"cp-{s}-{name}"} for name in names} for s in seeds}
    save(Path(cfg["checkpoint_manifest"]),{"copies":checkpoints})
    datasets={f"{kind}{delay}":{"prefix_sha256":f"{kind}{delay}",
        "task_sha256":f"task-{kind}{delay}","target_sha256":f"target-{kind}{delay}",
        "unique_prefixes":10 if kind in ("V","T") else 20}
        for delay in cfg["fit_delays"] for kind in ("A","B","V","T") if kind!="T" or delay in cfg["test_delays"]}
    resource_fields=["allocated_parameters","effective_parameters","hidden_state_floats",
        "dense_forward_linear_macs_per_example","active_forward_linear_macs_per_example",
        "internal_steps_per_input","mask_buffer_bytes","sequence_length"]
    normalizer=np.zeros(64,dtype=np.float64)
    scale=np.ones(64,dtype=np.float64)
    norm_hash=digest(normalizer.tobytes()+scale.tobytes())
    for seed in seeds:
        for name in names:
            short=.70 if name=="multi_gru" else .80
            if case=="small_positive" and name=="multi_gru": short=.77
            if case=="no_excess" and name=="multi_gru":short=.80
            directory=tmp_path/f"seed-{seed}"/name
            directory.mkdir(parents=True)
            heads={}
            for delay in cfg["fit_delays"]:
                for suffix in ("A","B"):
                    head=f"H{delay}{suffix}"
                    file=directory/f"{head}.npz"
                    coefficients=np.zeros((65,4),dtype=np.float64)
                    np.savez_compressed(file,mean=normalizer,scale=scale,coefficients=coefficients)
                    val=.80 if case=="weak_validation" and name=="multi_gru" and head=="H8A" else .99
                    head_record={"fit_delay":delay,"fit_unique_prefixes":20,"parameters":260,
                        "ridge_lambda":1.0,"short_train_normalizer_sha256":norm_hash,
                        "coefficient_sha256":digest(coefficients.tobytes()),
                        "file_sha256":digest(file.read_bytes()),"validation_accuracy":val}
                    for test_delay in cfg["test_delays"]:
                        if delay==8 or delay==test_delay:
                            accuracy=short if delay==8 else .95
                            if (case=="source_refit_unstable" and name=="multi_gru" and head=="H8B" and test_delay==32):accuracy=.59
                            if (case=="target_refit_unstable" and name=="multi_gru" and head=="H32B" and test_delay==32):accuracy=.85
                            if (case=="source_specific_cutoff" and name=="multi_gru" and head=="H8B" and test_delay==32):accuracy=short-.01
                            if (case=="target_specific_cutoff" and name=="multi_gru" and head=="H32B" and test_delay==32):accuracy=.94
                            head_record[f"test_delay{test_delay}"]={"accuracy":accuracy}
                    heads[head]=head_record
            pre=.70 if name=="multi_gru" else .80
            if case=="behavior_gap_missing" and name=="multi_gru":pre=.80
            observed={"status":"COMPLETED","source_replay":"EXACT",
                "checkpoint_sha256":checkpoints[str(seed)][name]["sha256"],
                "initial_parameters_sha256":"same" if name!="gru_wide" else "wide",
                "mask_sha256":name,"datasets":datasets if case!="data_mismatch" or name!="gru_wide" else {**datasets,"T32":{"prefix_sha256":"bad","task_sha256":"task-T32","target_sha256":"target-T32","unique_prefixes":10}},
                "heads":heads,"head_capacity_parameters":260,
                "normalizer_sha256":norm_hash,"train_only_delay8_scaling":True,
                "state_extraction_forward_linear_flops_proxy":1000,
                "test":{str(delay):{"dataset_sha256":f"task-T{delay}",
                    "behavior":{"pairs":10,"pre_accuracy":pre},
                    "resources":{field:64 for field in resource_fields}}
                    for delay in cfg["test_delays"]}}
            save(directory/"result.json",observed)
    config_path=tmp_path/"config.json"
    save(config_path,cfg)
    source=Path("artifacts/EXP-0004/v1/manifest.json")
    fingerprints=source_fingerprints()
    save(tmp_path/"manifest.json",{"status":"COMPLETED","run_kind":"REGISTERED",
        "seeds_run":seeds,"config":cfg,"config_path":str(config_path),
        "experiment_id":cfg["experiment_id"],"version":cfg["version"],
        "config_sha256":digest(config_path.read_bytes()),
        "checkpoint_manifest_sha256":digest(Path(cfg["checkpoint_manifest"]).read_bytes()),
        "source_run_manifest_sha256":digest(source.read_bytes()),
        "source_sha256":fingerprints,"validation":{"pass":True,"source_sha256":fingerprints}})
    if case=="tampered_config":
        config_path.write_text(config_path.read_text()+"\n")
        with pytest.raises(ValueError,match="Frozen config"):
            analyze(tmp_path)
        return
    report=analyze(tmp_path)
    assert report["decision"]==expected
    assert report["gate1_advanced"] is False
    if case=="data_mismatch":assert report["provenance_and_matching_pass"] is False
