"""Causal pairing and decision-boundary checks on synthetic records only."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neurotopo.replay import target_digest, verify_variant
from neurotopo.replay_analysis import analyze
from neurotopo.run import digest, source_fingerprints
from neurotopo.study import variant_batch


def test_frozen_intervention_instances_share_all_non_distractor_evidence():
    inputs = {}
    for payload in ["full", "blank", "repeat", "id_only", "value_only"]:
        rng = np.random.default_rng(742032)
        inputs[payload] = variant_batch(8, 32, rng, payload)
        verify_variant(inputs["full"], inputs[payload])
        assert target_digest(inputs[payload]) == target_digest(inputs["full"])
        assert torch.equal(inputs[payload].x[::2, :inputs[payload].pre_index + 1],
                           inputs[payload].x[1::2, :inputs[payload].pre_index + 1])


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


@pytest.mark.parametrize("case,decision", [
    ("supported", "SUPPORTED"), ("small_positive", "PARTIALLY_SUPPORTED"),
    ("excluded_gain", "FALSIFIED"), ("control_collapse", "INCONCLUSIVE"),
    ("no_original_deficit", "INCONCLUSIVE"), ("input_mismatch", "INCONCLUSIVE"),
    ("config_tamper", "REJECT")])
def test_interaction_decision_respects_guards(tmp_path: Path, case: str, decision: str):
    names = ["multi_gru", "random_gru", "gru_wide"]
    seeds = [1, 2]
    payloads = ["full", "blank", "repeat", "id_only", "value_only"]
    config = {"experiment_id": "SYNTHETIC_TEST", "version": "v2", "source_experiment": "EXP-0004/v1",
        "source_commit": "eec9ca4", "checkpoint_manifest": str(tmp_path / "checkpoints.json"), "models": names,
        "seeds": seeds, "delays": [32], "primary_delay": 32, "payloads": payloads,
        "test_pairs": 10, "bootstrap_samples": 100, "bootstrap_seed": 1,
        "meaningful_interaction": .03, "minimum_block_gain": .03,
        "minimum_dynamic_joint": .10, "maximum_dynamic_joint": .97,
        "minimum_original_gap": -.01, "max_control_blank_degradation": .03}
    checkpoints = {str(seed): {name: {"sha256": f"weight-{name}-{seed}"} for name in names} for seed in seeds}
    _write(Path(config["checkpoint_manifest"]), {"copies": checkpoints})
    resources = {key: 64 for key in ["allocated_parameters", "effective_parameters", "hidden_state_floats",
        "dense_forward_linear_macs_per_example", "active_forward_linear_macs_per_example",
        "internal_steps_per_input", "sequence_length", "mask_buffer_bytes",
        "dense_forward_linear_flops_per_batch"]}
    for seed in seeds:
        for name in names:
            full = .55 if name == "multi_gru" else .65
            blank = .75 if name == "multi_gru" else .75
            if case == "small_positive":
                blank = .62 if name == "multi_gru" else .70
            if case == "excluded_gain":
                blank = .56 if name == "multi_gru" else .66
            if case == "control_collapse" and name == "random_gru":
                blank = .20
            if case == "no_original_deficit" and name == "multi_gru":
                full = .66
            metric = lambda x: {"pairs": 10, "joint_pair_accuracy": x, "pre_accuracy": .95,
                "confirm_accuracy": x, "revise_accuracy": x, "pre_nll": 0.1,
                "confirm_nll": 0.1, "revise_nll": 0.1, "evidence_sensitivity": .5}
            record = {"status": "COMPLETED", "source_replay": "EXACT",
                "initial_parameters_sha256": "same" if name != "gru_wide" else "different",
                "mask_sha256": name, "checkpoint_sha256": checkpoints[str(seed)][name]["sha256"],
                "metrics": {f"{p}_delay32": metric(full if p == "full" else blank) for p in payloads},
                "input_sha256": {f"{p}_delay32": "DIFFERENT" if case == "input_mismatch" and name == "gru_wide" and p == "blank" else p for p in payloads},
                "target_sha256": {f"{p}_delay32": "same-target" for p in payloads},
                "pair_overlap_with_full": {f"{p}_delay32": {"both_correct": 10, "rescued": 0, "harmed": 0, "both_wrong": 0} for p in payloads},
                "resources": {f"{p}_delay32": resources for p in payloads}}
            _write(tmp_path / f"seed-{seed}" / name / "result.json", record)
    config_path = tmp_path / "config.json"
    _write(config_path, config)
    source_manifest_path = Path("artifacts/EXP-0004/v1/manifest.json")
    _write(tmp_path / "manifest.json", {"status": "COMPLETED", "run_kind": "REGISTERED",
        "seeds_run": seeds, "config": config, "validation": {"pass": True, "source_sha256": {}},
        "experiment_id": config["experiment_id"], "version": config["version"],
        "config_path": str(config_path), "config_sha256": digest(config_path.read_bytes()),
        "source_sha256": source_fingerprints(),
        "source_run_manifest_sha256": digest(source_manifest_path.read_bytes()),
        "checkpoint_manifest_sha256": digest(Path(config["checkpoint_manifest"]).read_bytes())})
    manifest_path = tmp_path / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["validation"]["source_sha256"] = m["source_sha256"]
    _write(manifest_path, m)
    if case == "config_tamper":
        config_path.write_text(config_path.read_text() + "\n")
        with pytest.raises(ValueError, match="Frozen config"):
            analyze(tmp_path)
        return
    result = analyze(tmp_path)
    assert result["decision"] == decision
    assert result["gate1_advanced"] is False
    if case == "input_mismatch":
        assert result["provenance_and_matching_pass"] is False
