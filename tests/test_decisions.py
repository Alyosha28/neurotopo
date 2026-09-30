"""Synthetic decision checks, never scientific evidence."""
import json

import pytest

from neurotopo.study_analysis import analyze


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("case,expected", [("equal_learned", "FALSIFIED"), ("floor", "INCONCLUSIVE"),
    ("failed_learning", "INCONCLUSIVE"), ("mismatch", "INCONCLUSIVE"), ("initialization_mismatch", "INCONCLUSIVE"),
    ("contextual_data_mismatch", "INCONCLUSIVE"), ("budget_mismatch", "INCONCLUSIVE"), ("positive", "SUPPORTED")])
def test_meaningful_gain_requires_valid_learned_nonfloor_comparison(tmp_path, case, expected):
    arms = ["multi_gru", "random_gru", "gru_wide"]
    c = {"experiment_id": "SYNTHETIC_TEST", "version": "v1", "analysis": "matched_architecture",
         "arms": [{"name": n, "schedule": [[8,6000]]} for n in arms], "seeds": [1,2], "eval_delays": [8,32,64],
         "primary_delay": 32, "robustness_delay": 64, "minimum_pre_accuracy": .9,
         "minimum_joint_accuracy": .5, "minimum_effect": .03, "primary_floor": .1,
         "max_confirm_degradation": .01, "bootstrap_samples": 100, "bootstrap_seed": 11}
    write(tmp_path / "manifest.json", {"config": c, "status": "COMPLETED", "run_kind": "REGISTERED",
         "seeds_run": [1,2], "validation": {"pass": True, "source_sha256": {}}, "source_sha256": {}, "wall_seconds": 0, "base_config": {"batch_size": 64}})
    fields = ["effective_parameters", "allocated_parameters", "hidden_state_floats", "dense_forward_linear_macs_per_example", "active_forward_linear_macs_per_example"]
    for name in arms:
        for seed in c["seeds"]:
            metrics = {}
            for delay in c["eval_delays"]:
                value = 1 if delay == 8 else (.01 if case == "floor" else .8)
                if case == "positive" and delay != 8:
                    value = .95 if name == "multi_gru" else .75
                pre = .25 if case == "failed_learning" and name == "multi_gru" else 1
                metrics[f"noise_delay{delay}"] = {"pre_accuracy": pre, "joint_pair_accuracy": value, "confirm_accuracy": value}
            row = {"status": "COMPLETED", "metrics": metrics,
                   "initial_parameters_sha256": "different" if case == "initialization_mismatch" and name == "random_gru" else "same",
                   "dataset_sha256": {"same": "different" if case == "contextual_data_mismatch" and name == "gru_wide" else "hash"},
                   "resources": {d: {f: 100 for f in fields} for d in metrics},
                   "train_examples": 384000, "train_tokens": 41 if case == "mismatch" and name == "random_gru" else 40,
                   "train_updates": 5999 if case == "budget_mismatch" else 6000, "training_forward_linear_flops": 100, "graph": {"diagonal_connections": 4}}
            write(tmp_path / f"seed-{seed}" / name / "result.json", row)
    report = analyze(tmp_path)
    assert report["decision"] == expected
    if case in {"mismatch", "initialization_mismatch", "contextual_data_mismatch", "budget_mismatch"}:
        assert report["next_branch"] == "REPAIR_INVALID_EVIDENCE"
    assert report["gate1_automatically_advanced"] is False
