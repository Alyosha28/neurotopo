"""Registered revise-only decision guards on synthetic records."""
import json
from pathlib import Path

import numpy as np
import pytest

from neurotopo.final_probe_analysis import analyze
from neurotopo.run import digest, source_fingerprints


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("case,expected", [
    ("large_deficit","SUPPORTED"),("small_deficit","PARTIALLY_SUPPORTED"),
    ("no_deficit","FALSIFIED"),("weak_positive_control","INCONCLUSIVE"),
    ("unstable_refit","INCONCLUSIVE"),("behavior_gap_missing","INCONCLUSIVE"),
    ("data_mismatch","INCONCLUSIVE"),("fit_proxy_mismatch","INCONCLUSIVE"),
    ("config_tamper","REJECT")])
def test_revise_decision_requires_pair_and_source_controls(tmp_path: Path,case: str,expected: str):
    models=["multi_gru","random_gru","gru_wide"]
    seeds=[1,2]
    cfg={"experiment_id":"SYNTHETIC","version":"v1","source_experiment":"EXP-0004/v1",
        "source_commit":"eec9ca4","checkpoint_manifest":str(tmp_path/"checkpoints.json"),
        "models":models,"seeds":seeds,"delays":[32],"primary_delay":32,
        "train_pairs":10,"validation_pairs":10,"test_pairs":10,
        "ridge_lambda":1.0,"adequate_mean_validation_revise":.90,
        "adequate_seed_validation_revise":.85,"adequate_seed_count":2,
        "maximum_refit_variation":.02,"minimum_behavior_revise":.50,
        "minimum_behavior_gap":-.01,"minimum_linear_deficit":.03,
        "bootstrap_samples":100,"bootstrap_seed":3}
    copies={str(seed):{name:{"sha256":f"cp-{seed}-{name}"} for name in models} for seed in seeds}
    write(Path(cfg["checkpoint_manifest"]),{"copies":copies})
    sample={kind:{"task_sha256":kind,"base_prefix_sha256":kind,
        "final_target_sha256":kind,"pairs":10,"examples":20}
        for kind in ("train_a","train_b","validation","test")}
    resource_fields=["allocated_parameters","effective_parameters","hidden_state_floats",
        "dense_forward_linear_macs_per_example","active_forward_linear_macs_per_example",
        "internal_steps_per_input","mask_buffer_bytes","sequence_length"]
    mean=np.zeros(64,dtype=np.float64)
    scale=np.ones(64,dtype=np.float64)
    coefficient=np.zeros((65,4),dtype=np.float64)
    normhash=digest(mean.tobytes()+scale.tobytes())
    for seed in seeds:
        for name in models:
            revise=.80 if name=="multi_gru" else .90
            if case=="small_deficit" and name=="multi_gru":revise=.88
            if case=="no_deficit" and name=="multi_gru":revise=.90
            validation=.80 if case=="weak_positive_control" and name=="random_gru" else .95
            original=.80 if name=="multi_gru" else .85
            if case=="behavior_gap_missing" and name=="multi_gru":original=.85
            directory=tmp_path/f"seed-{seed}"/name
            directory.mkdir(parents=True)
            heads={}
            for suffix in ("A","B"):
                head=f"H32{suffix}"
                file=directory/f"{head}-final.npz"
                np.savez_compressed(file,mean=mean,scale=scale,coefficients=coefficient)
                testval=.70 if case=="unstable_refit" and name=="multi_gru" and suffix=="B" else revise
                metrics=lambda x:{"revise_accuracy":x,"confirm_accuracy":.95,
                    "all_accuracy":(x+.95)/2,"both_pair_accuracy":x*.95,"mean_squared_error":.1}
                heads[head]={"fit_pairs":10,"fit_examples":20,"parameters":260,"ridge_lambda":1.0,
                    "normalizer_sha256":normhash,"coefficient_sha256":digest(coefficient.tobytes()),
                    "file_sha256":digest(file.read_bytes()),"train":metrics(1.0),
                    "validation":metrics(validation),"test":metrics(testval)}
            datasets=sample if case!="data_mismatch" or name!="gru_wide" else {**sample,"test":{**sample["test"],"task_sha256":"wrong"}}
            item={"status":"COMPLETED","source_replay":"EXACT",
                "checkpoint_sha256":copies[str(seed)][name]["sha256"],
                "initial_parameters_sha256":"same" if name!="gru_wide" else "wide",
                "mask_sha256":name,"delays":{"32":{"heads":heads,"normalizer_sha256":normhash,
                    "datasets":datasets,"pair_split_verified":True,"full_final_readout_parity_verified":True,
                    "probe_parameters":260,"train_only_scaling":True,"ridge_lambda":1.0,
                    "ridge_fit_macs_proxy":4*10*(65*65+65*4)+(1 if case=="fit_proxy_mismatch" and name=="random_gru" else 0),
                    "original_behavior":{"pairs":10,"pre_accuracy":.9,"confirm_accuracy":.95,
                        "revise_accuracy":original,"joint_pair_accuracy":.8},
                    "resources":{field:64 for field in resource_fields},
                    "state_extraction_forward_linear_flops_proxy":1000}}}
            write(directory/"result.json",item)
    config_path=tmp_path/"config.json"
    write(config_path,cfg)
    source=Path("artifacts/EXP-0004/v1/manifest.json")
    fingerprints=source_fingerprints()
    write(tmp_path/"manifest.json",{"status":"COMPLETED","run_kind":"REGISTERED",
        "seeds_run":seeds,"config":cfg,"config_path":str(config_path),
        "experiment_id":cfg["experiment_id"],"version":cfg["version"],
        "config_sha256":digest(config_path.read_bytes()),
        "checkpoint_manifest_sha256":digest(Path(cfg["checkpoint_manifest"]).read_bytes()),
        "source_run_manifest_sha256":digest(source.read_bytes()),
        "source_sha256":fingerprints,"validation":{"pass":True,"source_sha256":fingerprints}})
    if case=="config_tamper":
        config_path.write_text(config_path.read_text()+"\n")
        with pytest.raises(ValueError,match="Frozen config"):
            analyze(tmp_path)
        return
    report=analyze(tmp_path)
    assert report["decision"]==expected
    assert report["gate1_advanced"] is False
    if case in ("data_mismatch","fit_proxy_mismatch"):
        assert report["provenance_and_matching_pass"] is False
