"""Decision gates on synthetic probe records, never experimental findings."""
import json
from pathlib import Path

import numpy as np
import pytest

from neurotopo.probe_analysis import analyze
from neurotopo.run import digest, source_fingerprints


def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


@pytest.mark.parametrize("case,expected", [
    ("large_deficit", "SUPPORTED"), ("small_deficit", "PARTIALLY_SUPPORTED"),
    ("no_deficit", "FALSIFIED"), ("weak_decoder", "INCONCLUSIVE"),
    ("behavior_gap_missing", "INCONCLUSIVE"), ("data_mismatch", "INCONCLUSIVE"),
    ("config_tamper", "REJECT")])
def test_linear_access_decision_requires_frozen_evidence(tmp_path: Path, case: str, expected: str):
    names = ["multi_gru", "random_gru", "gru_wide"]
    seeds = [1, 2]
    config = {"experiment_id": "SYNTHETIC", "version": "v2", "source_experiment": "EXP-0004/v1",
        "source_commit": "eec9ca4", "checkpoint_manifest": str(tmp_path / "checkpoints.json"),
        "models": names, "seeds": seeds, "delays": [32], "primary_delay": 32,
        "train_pairs": 12, "validation_pairs": 10, "test_pairs": 10,
        "ridge_lambda": 1.0, "std_floor": .001, "bootstrap_samples": 100,
        "bootstrap_seed": 1, "adequate_mean_validation_uncued": .45,
        "adequate_seed_validation_uncued": .40, "adequate_seed_count": 2,
        "minimum_behavior_gap": -.01, "minimum_behavior_joint": .10,
        "minimum_uncued_deficit": .05, "minimum_common_early_pairs": 5}
    checkpoints = {str(seed): {name: {"sha256": f"cp-{seed}-{name}"} for name in names} for seed in seeds}
    write(Path(config["checkpoint_manifest"]), {"copies": checkpoints})
    dataset = {split: {"prefix_sha256": split, "full_task_sha256": split} for split in ("train","validation","test")}
    fields = ["allocated_parameters", "effective_parameters", "hidden_state_floats",
              "dense_forward_linear_macs_per_example", "active_forward_linear_macs_per_example",
              "internal_steps_per_input", "mask_buffer_bytes", "sequence_length"]
    for seed in seeds:
        for name in names:
            uncued = .50 if name == "multi_gru" else .60
            if case == "small_deficit" and name == "multi_gru": uncued = .58
            if case == "no_deficit" and name == "multi_gru": uncued = .60
            validation = .35 if case == "weak_decoder" and name == "random_gru" else .60
            metric = lambda value: {"uncued_accuracy": value, "prior_accuracy": .80,
                "all_candidate_accuracy": .65, "all_four_correct": .30, "mean_squared_error": .10}
            coefficient = np.zeros((65, 16), dtype=np.float64)
            directory = tmp_path / f"seed-{seed}" / name
            directory.mkdir(parents=True)
            probe_path = directory / "delay-32-probe.npz"
            np.savez_compressed(probe_path, mean=np.zeros(64), scale=np.ones(64), coefficients=coefficient)
            late = [1] * (7 if name == "random_gru" or case == "behavior_gap_missing" else 6) + [0] * (3 if name == "random_gru" or case == "behavior_gap_missing" else 4)
            early = [1] * 10
            scores = {"train": metric(.8), "validation": metric(validation), "test": metric(uncued)}
            record = {"status": "COMPLETED", "source_replay": "EXACT",
                "checkpoint_sha256": checkpoints[str(seed)][name]["sha256"],
                "initial_parameters_sha256": "same" if name != "gru_wide" else "wide",
                "mask_sha256": name, "delays": {"32": {"probe_parameters": 1040,
                    "split_unique_prefix_counts": {"train": 12, "validation": 10, "test": 10},
                    "state_extraction_forward_linear_flops_proxy": 1000,
                    "train_only_scaling": True, "ridge_lambda": 1.0, "std_floor": .001,
                    "scores": scores, "behavior": {"pre_accuracy": 1.0,
                        "joint_pair_accuracy": sum(late) / 10},
                    "test_pair_bits": {"early": early, "late": late},
                    "test_pair_early_sha256": digest(np.asarray(early, dtype=np.uint8).tobytes()),
                    "test_pair_late_sha256": digest(np.asarray(late, dtype=np.uint8).tobytes()),
                    "datasets": {**dataset} if case != "data_mismatch" or name != "gru_wide" else
                                {**dataset, "test": {"prefix_sha256": "different", "full_task_sha256": "different"}},
                    "probe_file_sha256": digest(probe_path.read_bytes()),
                    "probe_coefficients_sha256": digest(coefficient.tobytes()),
                    "resources": {field: 64 for field in fields}}}}
            write(directory / "result.json", record)
    config_path = tmp_path / "config.json"
    write(config_path, config)
    source = Path("artifacts/EXP-0004/v1/manifest.json")
    fingerprints = source_fingerprints()
    write(tmp_path / "manifest.json", {"status": "COMPLETED", "run_kind": "REGISTERED",
        "seeds_run": seeds, "config": config, "config_path": str(config_path),
        "experiment_id": config["experiment_id"], "version": config["version"],
        "config_sha256": digest(config_path.read_bytes()),
        "checkpoint_manifest_sha256": digest(Path(config["checkpoint_manifest"]).read_bytes()),
        "source_run_manifest_sha256": digest(source.read_bytes()),
        "source_sha256": fingerprints, "validation": {"pass": True, "source_sha256": fingerprints}})
    if case == "config_tamper":
        config_path.write_text(config_path.read_text() + "\n")
        with pytest.raises(ValueError, match="Frozen config"):
            analyze(tmp_path)
        return
    report = analyze(tmp_path)
    assert report["decision"] == expected
    assert report["gate1_advanced"] is False
    if case == "data_mismatch":
        assert report["provenance_and_matching_pass"] is False
