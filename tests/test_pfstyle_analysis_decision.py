import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from neurotopo import pfstyle_analysis
from neurotopo.pfstyle_analysis import analyze
from neurotopo.pfstyle_task import make_batch


ROOT = Path(__file__).resolve().parents[1]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _completed_fixture(tmp_path: Path, monkeypatch, *, k4_ood: float,
                       sparse_ood: float, dense_ood: float) -> Path:
    monkeypatch.chdir(tmp_path)
    # This fixture verifies the decision and saved-logit audit. Registered
    # stream identity is tested separately with the real task generator.
    monkeypatch.setattr(pfstyle_analysis, "_registered_test_identity", lambda *args: None)
    config = json.loads((ROOT / "configs/exp0011_v6.json").read_text())
    config["matrix_file"] = str(ROOT / "configs/exp0011_matrices.json")
    for phase in ("qualification", "formal", "replication"):
        config[phase]["test_examples_per_length"] = 8
    config_path = Path("configs/exp0011_v6.json")
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(config))
    source = {"src/fake.py": "isolated-analysis-source"}
    monkeypatch.setattr(pfstyle_analysis, "source_fingerprints", lambda: source)
    preflight_path = Path("artifacts/validation/exp0011_preflight.json")
    preflight_path.parent.mkdir(parents=True)
    preflight_path.write_text(json.dumps({"resource_status": "PASS"}))
    receipt_path = preflight_path.with_name("exp0011_preflight_receipt.json")
    receipt = {"preflight_report_sha256": _sha(preflight_path.read_bytes()),
               "budget_status": "PASS"}
    receipt_path.write_text(json.dumps(receipt))
    run = Path("run")
    run.mkdir()
    records = []
    for phase, seeds, arms in (
        ("qualification", config["qualification"]["seeds"], config["qualification"]["controls"]),
        ("formal", config["formal"]["seeds"], config["models"]),
        ("replication", config["replication"]["seeds"], config["models"]),
    ):
        for seed in seeds:
            seed_directory = run / phase / f"seed-{seed}"
            seed_directory.mkdir(parents=True)
            (seed_directory / "test_stream_manifest.json").write_text(json.dumps({
                "phase": phase, "seed": seed,
                "root": config[phase]["test_root_base"] + seed,
                "sealed_before_training": True,
                "streams": {condition: {"examples": 8, "length": length,
                                        "dataset_sha256": f"{condition}:{phase}:{seed}"}
                            for condition, length in (("iid16", 16), ("ood32", 32))},
            }))
            for arm in arms:
                directory = run / phase / f"seed-{seed}" / arm
                directory.mkdir()
                checkpoint = directory / "final.pt"
                predictions = directory / "predictions.npz"
                checkpoint.write_bytes(f"checkpoint:{phase}:{seed}:{arm}".encode())
                ood = (k4_ood if arm == "K4_score" else sparse_ood if arm == "S68_sparse"
                       else dense_ood if arm == "S68_dense" else 0.13)
                predictions_by_condition = {}
                metrics = {}
                for condition, kl in (("iid16", 0.05), ("ood32", ood)):
                    target = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (8, 1))
                    ptrue = np.exp(-kl)
                    logits = np.tile(np.log([ptrue, *(3 * [(1 - ptrue) / 3])]), (8, 1))
                    truth = np.zeros(8, dtype=np.int64)
                    arrays = {"pre_logits": logits.copy(), "final_logits": logits.copy(),
                              "pre_target": target.copy(), "final_target": target.copy(),
                              "truth": truth}
                    predictions_by_condition.update({f"{condition}_{key}": value
                                                     for key, value in arrays.items()})
                    metrics[condition] = {
                        "examples": 8,
                        "pre_joint_kl_to_oracle": kl,
                        "final_joint_kl_to_oracle": kl,
                        "predicted_mode_map_fraction": [1.0, 0.0, 0.0, 0.0],
                        "predicted_entropy_final": float(-ptrue * np.log(ptrue)
                                                         - (1 - ptrue) * np.log((1 - ptrue) / 3)),
                        "prediction_sha256": _sha(b"".join(arrays[key].tobytes()
                                                       for key in sorted(arrays))),
                        "dataset_sha256": f"{condition}:{phase}:{seed}",
                    }
                np.savez_compressed(predictions, **predictions_by_condition)
                result = {
                    "phase": phase, "seed": seed, "arm": arm, "status": "COMPLETED",
                    "train_updates": 1200, "train_examples": 1200 * 128,
                    "train_tokens": 1200 * 128 * 16,
                    "training_forward_linear_flops_proxy": 1000 * 2 * 128 * 1200,
                    "resources": {"iid16": {"dense_forward_linear_macs_per_example": 1000}},
                    "training_data_sha256": f"train:{phase}:{seed}",
                    "checkpoint_file": str(checkpoint.relative_to(run)),
                    "checkpoint_sha256": _sha(checkpoint.read_bytes()),
                    "predictions_file": str(predictions.relative_to(run)),
                    "predictions_sha256": _sha(predictions.read_bytes()),
                    "training_seconds_including_monitor": 1.0,
                    "metrics": metrics,
                }
                (directory / "result.json").write_text(json.dumps(result))
                records.append({"phase": phase, "seed": seed, "arm": arm})
    manifest = {
        "config": config,
        "config_sha256": _sha(config_path.read_bytes()),
        "matrix_json_sha256": config["matrix_json_sha256"],
        "status": "COMPLETED", "research_updates": 97200,
        "preflight": {"resource_status": "PASS"},
        "preflight_sha256": _sha(preflight_path.read_bytes()),
        "preflight_receipt": receipt,
        "preflight_receipt_sha256": _sha(receipt_path.read_bytes()),
        "source_sha256": source,
        "task_information": {"status": "PASS"},
        "qualification": {"status": "PASS"},
        "records": records,
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def test_complete_registered_paired_result_is_field_supported(tmp_path, monkeypatch):
    run = _completed_fixture(tmp_path, monkeypatch, k4_ood=0.10,
                             sparse_ood=0.20, dense_ood=0.18)
    result = analyze(run)
    assert result["decision"] == "SUPPORTED"
    assert result["replication_guard"] is True
    assert result["paired_effect"]["sparse_minus_k4"]["mean"] > 0.05
    assert result["scope"].endswith("no project Gate 1 credit")


def test_complete_qualified_result_can_falsify_useful_gain(tmp_path, monkeypatch):
    run = _completed_fixture(tmp_path, monkeypatch, k4_ood=0.19,
                             sparse_ood=0.20, dense_ood=0.18)
    result = analyze(run)
    assert result["decision"] == "FALSIFIED"
    assert result["reason"] == "registered_0.05nat_primary_useful_gain_excluded"


def test_summary_kl_cannot_disagree_with_saved_logits(tmp_path, monkeypatch):
    run = _completed_fixture(tmp_path, monkeypatch, k4_ood=0.10,
                             sparse_ood=0.20, dense_ood=0.18)
    result_path = run / "formal/seed-12201/K4_score/result.json"
    result = json.loads(result_path.read_text())
    result["metrics"]["ood32"]["final_joint_kl_to_oracle"] = 0.001
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="summary KL does not match saved"):
        analyze(run)


def test_registered_stream_identity_checks_targets_and_input_digest():
    seed = 9_900_201
    root_base = 9_900_000
    config = {"matrix_file": str(ROOT / "configs/exp0011_matrices.json"),
              "qualification": {"test_examples_per_length": 8, "test_root_base": root_base}}
    arrays = {}
    result = {"metrics": {}}
    for condition, length in (("iid16", 16), ("ood32", 32)):
        batch = make_batch(8, length, root_base + seed, Path(config["matrix_file"]))
        arrays.update({f"{condition}_pre_target": batch.pre_target.numpy(),
                       f"{condition}_final_target": batch.final_target.numpy(),
                       f"{condition}_truth": batch.truth.numpy()})
        digest = _sha(b"".join((batch.x.numpy().tobytes(),
                                 batch.pre_target.numpy().tobytes(),
                                 batch.final_target.numpy().tobytes(),
                                 batch.truth.numpy().tobytes(),
                                 batch.mode_means.numpy().tobytes())))
        result["metrics"][condition] = {"dataset_sha256": digest}
    pfstyle_analysis._registered_test_identity(config, "qualification", seed, result, arrays)
    corrupted = dict(arrays)
    corrupted["ood32_truth"] = arrays["ood32_truth"].copy()
    corrupted["ood32_truth"][0] = (corrupted["ood32_truth"][0] + 1) % 4
    with pytest.raises(ValueError, match="targets differ"):
        pfstyle_analysis._registered_test_identity(config, "qualification", seed, result, corrupted)
