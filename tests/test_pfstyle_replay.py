"""EXP-0011 replay checks use fake runs and isolated engineering seeds only."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neurotopo import pfstyle_replay
from neurotopo.pfstyle_models import build_model
from neurotopo.pfstyle_study import _parameter_sha, evaluate
from neurotopo.pfstyle_task import make_batch


ROOT = Path(__file__).resolve().parents[1]
ENGINEERING_SEED = 9_900_001


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch, status: str) -> tuple[Path, dict]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pfstyle_replay, "git_output",
                        lambda *args: "" if args[0] == "status" else "fake-clean-commit")
    source = {"src/engineering_fixture.py": "sha"}
    monkeypatch.setattr(pfstyle_replay, "source_fingerprints", lambda: source)
    config = json.loads((ROOT / "configs/exp0011_v6.json").read_text())
    config["matrix_file"] = str(ROOT / "configs/exp0011_matrices.json")
    config["task"]["task_information_root"] = ENGINEERING_SEED + 1
    for phase in ("qualification", "formal", "replication"):
        config[phase]["seeds"] = [ENGINEERING_SEED]
        config[phase]["test_examples_per_length"] = 2
        config[phase]["test_root_base"] = ENGINEERING_SEED + 100
    config["training"]["updates_per_arm_seed"] = 0
    monkeypatch.setattr(pfstyle_replay, "REGISTERED_ROOTS", {
        "task_information": config["task"]["task_information_root"],
        "qualification_seed": ENGINEERING_SEED,
        "qualification_test": config["qualification"]["test_root_base"],
        "formal_seed": ENGINEERING_SEED,
        "formal_test": config["formal"]["test_root_base"],
    })
    config_path = Path("configs/exp0011_v6.json")
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(config))
    verify_path = Path("artifacts/validation/exp0011_matrix_verify.json")
    verify_path.parent.mkdir(parents=True)
    verify_path.write_bytes((ROOT / "artifacts/validation/exp0011_matrix_verify.json").read_bytes())
    preflight_path = Path("artifacts/validation/exp0011_preflight.json")
    preflight = {"resource_status": "RESOURCE_MISMATCH" if status == "RESOURCE_STOP" else "PASS",
                 "config_sha256": _sha(config_path),
                 "matrix_json_sha256": _sha(Path(config["matrix_file"])),
                 "source_sha256": source}
    preflight_path.write_text(json.dumps(preflight))
    receipt_path = preflight_path.with_name("exp0011_preflight_receipt.json")
    receipt = {"preflight_report_sha256": _sha(preflight_path), "budget_status": "PASS",
               "seconds_after_report_fsync": 1.0}
    receipt_path.write_text(json.dumps(receipt))
    Path("uv.lock").write_text("isolated fake lock")
    run = tmp_path / "fake-run"
    run.mkdir()
    manifest = {"status": status, "config": config, "config_sha256": _sha(config_path),
                "matrix_json_sha256": _sha(Path(config["matrix_file"])),
                "preflight_sha256": _sha(preflight_path), "preflight": preflight,
                "preflight_receipt_sha256": _sha(receipt_path), "preflight_receipt": receipt,
                "source_sha256": source, "lock_sha256": _sha(Path("uv.lock")),
                "git_commit": "fake-clean-commit",
                "research_updates": 97_200 if status == "COMPLETED" else
                                    7_200 if status == "QUALIFICATION_STOP" else 0,
                "records": []}
    (run / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(pfstyle_replay, "CONFIG_PATH", config_path)
    monkeypatch.setattr(pfstyle_replay, "MATRIX_VERIFY_PATH", verify_path)
    monkeypatch.setattr(pfstyle_replay, "PREFLIGHT_PATH", preflight_path)
    monkeypatch.setattr(pfstyle_replay, "PREFLIGHT_RECEIPT_PATH", receipt_path)
    return run, config


def _arm(run: Path, config: dict, phase: str, arm: str) -> None:
    seed = ENGINEERING_SEED
    directory = run / phase / f"seed-{seed}" / arm
    directory.mkdir(parents=True)
    model = build_model(arm, seed, config["training"]["graph_seed_base"] + seed)
    checkpoint = directory / "final.pt"
    torch.save(model.state_dict(), checkpoint)
    metrics = {}
    predictions = {}
    for condition, length in (("iid16", 16), ("ood32", 32)):
        block = config[phase]
        batch = make_batch(2, length, block["test_root_base"] + seed, Path(config["matrix_file"]))
        metric, arrays = evaluate(model, batch)
        metrics[condition] = metric
        predictions.update({f"{condition}_{key}": value for key, value in arrays.items()})
    prediction_path = directory / "predictions.npz"
    np.savez_compressed(prediction_path, **predictions)
    result = {"phase": phase, "seed": seed, "arm": arm, "status": "COMPLETED",
              "train_updates": 0, "metrics": metrics,
              "training_data_sha256": hashlib.sha256(b"").hexdigest(),
              "initial_parameters_sha256": _parameter_sha(model),
              "final_parameters_sha256": _parameter_sha(model),
              "graph_mask_sha256": hashlib.sha256(model.mask.numpy().tobytes()).hexdigest(),
              "checkpoint_file": str(checkpoint.relative_to(run)),
              "checkpoint_sha256": _sha(checkpoint),
              "predictions_file": str(prediction_path.relative_to(run)),
              "predictions_sha256": _sha(prediction_path)}
    (directory / "result.json").write_text(json.dumps(result))


def test_resource_stop_only_checks_preflight_and_matrix(tmp_path, monkeypatch):
    run, _ = _fixture(tmp_path, monkeypatch, "RESOURCE_STOP")
    monkeypatch.setattr(pfstyle_replay, "make_batch", lambda *args, **kwargs: 1 / 0)
    monkeypatch.setattr(pfstyle_replay, "task_information", lambda *args: 1 / 0)
    report = pfstyle_replay.replay(run, tmp_path / "report.json")
    assert report["status"] == "PASS"
    assert report["selected_phase"] == "resource"
    assert report["optimizer_updates"] == 0
    assert not (run / "task_information.json").exists()
    receipt = json.loads((tmp_path / "report_receipt.json").read_text())
    assert receipt["replay_report_sha256"] == _sha(tmp_path / "report.json")
    assert receipt["budget_status"] == "PASS"
    with pytest.raises(FileExistsError):
        pfstyle_replay.replay(run, tmp_path / "report.json")


def test_resource_numeric_pass_but_receipt_timeout_is_valid_stop(tmp_path, monkeypatch):
    run, _ = _fixture(tmp_path, monkeypatch, "RESOURCE_STOP")
    preflight_path = pfstyle_replay.PREFLIGHT_PATH
    preflight = json.loads(preflight_path.read_text())
    preflight["resource_status"] = "PASS"
    preflight_path.write_text(json.dumps(preflight))
    receipt_path = pfstyle_replay.PREFLIGHT_RECEIPT_PATH
    receipt = json.loads(receipt_path.read_text())
    receipt["preflight_report_sha256"] = _sha(preflight_path)
    receipt["budget_status"] = "INCOMPLETE_BUDGET"
    receipt_path.write_text(json.dumps(receipt))
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["preflight"] = preflight
    manifest["preflight_sha256"] = _sha(preflight_path)
    manifest["preflight_receipt"] = receipt
    manifest["preflight_receipt_sha256"] = _sha(receipt_path)
    manifest["stop_reason"] = "INCOMPLETE_BUDGET"
    manifest_path.write_text(json.dumps(manifest))
    report = pfstyle_replay.replay(run, tmp_path / "budget-report.json")
    assert report["status"] == "PASS"
    assert report["selected_phase"] == "resource"
    assert report["optimizer_updates"] == 0


def test_task_stop_only_checks_task_report(tmp_path, monkeypatch):
    run, _ = _fixture(tmp_path, monkeypatch, "TASK_STOP")
    info = {"status": "TASK_UNINFORMATIVE", "ood32_dataset_sha256": "fixture"}
    (run / "task_information.json").write_text(json.dumps(info))
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["task_information"] = info
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(pfstyle_replay, "task_information", lambda config: info)
    monkeypatch.setattr(pfstyle_replay, "make_batch", lambda *args, **kwargs: 1 / 0)
    report = pfstyle_replay.replay(run, tmp_path / "report.json")
    assert report["status"] == "PASS"
    assert report["selected_phase"] == "task_information"


def test_qualification_pair_byte_replay_detects_changed_archive(tmp_path, monkeypatch):
    run, config = _fixture(tmp_path, monkeypatch, "QUALIFICATION_STOP")
    for arm in config["qualification"]["controls"]:
        _arm(run, config, "qualification", arm)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["records"] = [{"phase": "qualification", "seed": ENGINEERING_SEED, "arm": arm}
                           for arm in config["qualification"]["controls"]]
    manifest_path.write_text(json.dumps(manifest))
    passed = pfstyle_replay.replay(run, tmp_path / "passed.json")
    assert passed["status"] == "PASS"
    assert len(passed["replayed_arms"]) == 2
    archive = run / "qualification" / f"seed-{ENGINEERING_SEED}" / "S68_dense" / "predictions.npz"
    with np.load(archive) as original:
        changed = {key: original[key].copy() for key in original.files}
    changed["ood32_final_logits"][0, 0] += 1
    np.savez_compressed(archive, **changed)
    failed = pfstyle_replay.replay(run, tmp_path / "failed.json")
    assert failed["status"] == "FAIL"
    assert any("predictions_sha256" in row["check"] for row in failed["failures"])
    assert any("ood32_final_logits" in row["check"] for row in failed["failures"])
    assert failed["optimizer_updates"] == 0


def test_complete_formal_takes_priority_over_qualification(tmp_path, monkeypatch):
    run, config = _fixture(tmp_path, monkeypatch, "COMPLETED")
    for arm in ("K4_score", "S68_sparse"):
        _arm(run, config, "formal", arm)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["records"] = [{"phase": "formal", "seed": ENGINEERING_SEED, "arm": arm}
                           for arm in config["models"]]
    manifest_path.write_text(json.dumps(manifest))
    report = pfstyle_replay.replay(run, tmp_path / "report.json")
    assert report["status"] == "PASS"
    assert report["selected_phase"] == "formal"
    assert {item.rsplit("/", 1)[-1] for item in report["replayed_arms"]} == set(pfstyle_replay.PAIR)
