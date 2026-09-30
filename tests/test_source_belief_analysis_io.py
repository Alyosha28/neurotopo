"""Complete synthetic run records exercise EXP-0009 provenance and decision ingestion."""

import json

from neurotopo.run import digest, write_json
from neurotopo.source_belief_analysis import analyze, MATCHED_RESOURCE_FIELDS


def test_complete_matched_run_analyzes_and_initialization_mismatch_blocks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "run"
    seeds = [9401, 9402]
    models = ["multi_gru", "random_gru", "gru_wide"]
    c = {"experiment_id": "EXP-0009", "version": "v1", "analysis": "matched_nuisance_transfer",
         "seeds": seeds, "models": models, "updates": 1500, "batch_size": 128,
         "eval_distractors_per_evidence": [1, 4, 8], "bootstrap_samples": 1000,
         "bootstrap_seed": 10, "iid_mean_kl_max": .05, "iid_each_seed_kl_max": .10,
         "control_primary_mean_kl_min": .07, "control_primary_mean_kl_max": 1.,
         "control_seed_kl_min": .05, "control_min_seeds_with_headroom": 2,
         "max_iid_kl_cost": .01, "max_h_accuracy_cost": .01,
         "minimum_useful_kl_gain": .05}
    resource = {field: 100 for field in MATCHED_RESOURCE_FIELDS}
    resources = {f"d{d}": dict(resource) for d in (1, 4, 8)}
    preflight = {"pass": True, "source_sha256": {}, "config_sha256": "cfg",
                 "schedule_sha256": "schedule", "final_dataset_sha256": {f"d{d}": "same" for d in (1, 4, 8)},
                 "training_tokens_per_arm": 3456000, "training_forward_linear_flops_per_arm": 99,
                 "seeds": []}
    for seed in seeds:
        source = {"seed": seed, "training_dataset_sha256": f"train-{seed}",
                  "models": {name: {"initial_parameters_sha256": "initial",
                                    "mask_sha256": f"mask-{name}", "resources": resources}
                             for name in models}}
        preflight["seeds"].append(source)
        for name in models:
            path = run / f"seed-{seed}" / name
            path.mkdir(parents=True)
            (path / "checkpoints").mkdir()
            checkpoint = path / "checkpoints/final.pt"
            predictions = path / "predictions.npz"
            checkpoint.write_bytes(b"checkpoint")
            predictions.write_bytes(b"predictions")
            metrics = {}
            for d in (1, 4, 8):
                kl = (.01 if d == 1 else .10 if name == "multi_gru" else .30)
                metrics[f"d{d}"] = {"pre_joint_kl_to_oracle": kl,
                                    "final_joint_kl_to_oracle": kl,
                                    "final_joint_oracle_map_agreement": .8,
                                    "final_h_true_accuracy": .8,
                                    "final_joint_true_accuracy": .7}
            row = {"status": "COMPLETED", "seed": seed, "model": name,
                   "initial_parameters_sha256": "initial", "mask_sha256": f"mask-{name}",
                   "cross_original_block_edges": 0 if name == "multi_gru" else 10,
                   "graph": {"diagonal_connections": 64, "in_degree": [16] * 64,
                             "out_degree": [16] * 64, "strongly_connected": name != "multi_gru"},
                   "training_dataset_sha256": f"train-{seed}",
                   "training_schedule_sha256": "schedule", "train_updates": 1500,
                   "train_examples": 192000, "train_tokens": 3456000,
                   "training_forward_linear_flops": 99, "metrics": metrics,
                   "dataset_sha256": preflight["final_dataset_sha256"], "resources": resources,
                   "checkpoint_file": str(checkpoint.relative_to(run)),
                   "checkpoint_sha256": digest(checkpoint.read_bytes()),
                   "predictions_sha256": digest(predictions.read_bytes())}
            write_json(path / "result.json", row)
    preflight_path = tmp_path / "artifacts/validation/exp0009_preflight.json"
    write_json(preflight_path, preflight)
    manifest = {"status": "COMPLETED", "run_kind": "REGISTERED", "seeds_run": seeds,
                "config": c, "source_sha256": {}, "config_sha256": "cfg",
                "validation": {"pass": True, "source_sha256": {}},
                "preflight": preflight, "preflight_sha256": digest(preflight_path.read_bytes()),
                "wall_seconds": 1.0}
    write_json(run / "manifest.json", manifest)
    report = analyze(run)
    assert report["matching_pass"]
    assert report["decision"] == "SUPPORTED"
    assert report["gate1_automatically_advanced"] is False
    assert (run / "report.md").is_file()
    bad = run / "seed-9401/random_gru/result.json"
    altered = json.loads(bad.read_text())
    altered["initial_parameters_sha256"] = "different"
    write_json(bad, altered)
    report = analyze(run)
    assert report["decision"] == "INCONCLUSIVE"
    assert report["next_branch"] == "REPAIR_INVALID_EVIDENCE"
