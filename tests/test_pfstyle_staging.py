import hashlib
import json
from pathlib import Path

import pytest

from neurotopo import pfstyle_analysis, pfstyle_study
from neurotopo.pfstyle_models import build_model
from neurotopo.pfstyle_analysis import analyze, paired_summary
from neurotopo.pfstyle_preflight import classify_resource
from neurotopo.pfstyle_task import make_batch


ROOT = Path(__file__).resolve().parents[1]


def _fake_registered_context(tmp_path: Path, monkeypatch, resource_status: str) -> Path:
    monkeypatch.chdir(tmp_path)
    config = json.loads((ROOT / "configs/exp0011_v6.json").read_text())
    config["matrix_file"] = str(ROOT / "configs/exp0011_matrices.json")
    config_path = Path("configs/exp0011_v6.json")
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(config))
    validation = Path("artifacts/validation")
    validation.mkdir(parents=True)
    (validation / "exp0011_matrix_verify.json").write_bytes(
        (ROOT / "artifacts/validation/exp0011_matrix_verify.json").read_bytes())
    source = {"src/fake.py": "engineering-only-test"}
    (validation / "suite.json").write_text(json.dumps({"pass": True, "source_sha256": source}))
    (validation / "exp0011_preflight.json").write_text(json.dumps({
        "engineering_valid": True,
        "resource_status": resource_status,
        "source_sha256": source,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "matrix_json_sha256": config["matrix_json_sha256"],
    }))
    receipt = {"budget_status": "PASS",
               "preflight_report_sha256": hashlib.sha256(
                   (validation / "exp0011_preflight.json").read_bytes()).hexdigest()}
    (validation / "exp0011_preflight_receipt.json").write_text(json.dumps(receipt))
    Path("uv.lock").write_text("isolated engineering test lock")
    monkeypatch.setattr(pfstyle_study, "source_fingerprints", lambda: source)
    monkeypatch.setattr(pfstyle_analysis, "source_fingerprints", lambda: source)
    monkeypatch.setattr(pfstyle_study, "git_output", lambda *args: "" if args[0] == "status" else "a" * 40)
    monkeypatch.setattr(pfstyle_study, "_verify_preflight", lambda *args: receipt)
    return config_path


def test_resource_failure_closes_analysis_without_task_or_formal(tmp_path, monkeypatch):
    config_path = _fake_registered_context(tmp_path, monkeypatch, "RESOURCE_MISMATCH")
    monkeypatch.setattr(pfstyle_study, "task_information", lambda _: pytest.fail("task root was accessed"))
    pfstyle_study.run(config_path, tmp_path / "run")
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["status"] == "RESOURCE_STOP"
    assert manifest["research_updates"] == 0
    assert not (tmp_path / "run/task_information.json").exists()
    assert not (tmp_path / "run/formal").exists()
    report = analyze(tmp_path / "run")
    assert report["decision"] == "INCONCLUSIVE"
    assert report["substatus"] == "RESOURCE_MISMATCH"


def test_task_information_failure_uses_zero_research_updates(tmp_path, monkeypatch):
    config_path = _fake_registered_context(tmp_path, monkeypatch, "PASS")
    monkeypatch.setattr(pfstyle_study, "task_information",
                        lambda _: {"status": "TASK_UNINFORMATIVE", "ood32_oracle_revision_rate_t8_to_t32": 0.0})
    pfstyle_study.run(config_path, tmp_path / "run")
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["status"] == "TASK_STOP"
    assert manifest["research_updates"] == 0
    assert not (tmp_path / "run/qualification").exists()
    assert analyze(tmp_path / "run")["substatus"] == "TASK_UNINFORMATIVE"


def test_preflight_receipt_budget_failure_closes_without_task_data(tmp_path, monkeypatch):
    config_path = _fake_registered_context(tmp_path, monkeypatch, "PASS")
    receipt = {"budget_status": "INCOMPLETE_BUDGET",
               "preflight_report_sha256": hashlib.sha256(
                   Path("artifacts/validation/exp0011_preflight.json").read_bytes()).hexdigest()}
    Path("artifacts/validation/exp0011_preflight_receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(pfstyle_study, "_verify_preflight", lambda *args: receipt)
    monkeypatch.setattr(pfstyle_study, "task_information", lambda _: pytest.fail("task root was accessed"))
    pfstyle_study.run(config_path, tmp_path / "run")
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["research_updates"] == 0
    assert manifest["stop_reason"] == "INCOMPLETE_BUDGET"
    assert analyze(tmp_path / "run")["substatus"] == "INCOMPLETE_BUDGET"


def test_control_qualification_failure_never_builds_formal_candidate(tmp_path, monkeypatch):
    config_path = _fake_registered_context(tmp_path, monkeypatch, "PASS")
    monkeypatch.setattr(pfstyle_study, "task_information", lambda _: {"status": "PASS"})
    monkeypatch.setattr(pfstyle_study, "make_batch", lambda *args, **kwargs: object())
    monkeypatch.setattr(pfstyle_study, "_training_batches", lambda *args: ([], "same-seed-data"))
    monkeypatch.setattr(pfstyle_study, "_seal_test_streams", lambda *args: {"sealed_before_training": True})
    trained = []

    def fake_save_arm(output, *, config, seed, arm, batches, training_data_sha, tests,
                      phase, started, on_update):
        assert phase == "qualification"
        assert arm in ("S68_sparse", "S68_dense")
        trained.append((seed, arm))
        for _ in range(config["training"]["updates_per_arm_seed"]):
            on_update()
        return {"arm": arm, "seed": seed, "metrics": {
            "iid16": {"final_joint_kl_to_oracle": 0.05},
            "ood32": {"final_joint_kl_to_oracle": 0.01},
        }}

    monkeypatch.setattr(pfstyle_study, "_save_arm", fake_save_arm)
    pfstyle_study.run(config_path, tmp_path / "run")
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert len(trained) == 6
    assert manifest["research_updates"] == 7200
    assert manifest["status"] == "QUALIFICATION_STOP"
    assert not (tmp_path / "run/formal").exists()
    assert not (tmp_path / "run/replication").exists()
    assert analyze(tmp_path / "run")["substatus"] == "CONTROL_FLOOR_OR_CEILING"


def test_registered_resource_window_and_shared_bootstrap_indices():
    c = json.loads((ROOT / "configs/exp0011_v6.json").read_text())
    base = {"total_dynamic_state_floats": 68, "allocated_parameters": 13200,
            "effective_parameters": 3984, "dense_forward_linear_macs_per_example": 205000}
    control = {"total_dynamic_state_floats": 68, "allocated_parameters": 14964,
               "effective_parameters": 4152, "dense_forward_linear_macs_per_example": 230000}
    assert classify_resource(base, control, [1.01] * 9, c) == "PASS"
    assert classify_resource(base, control, [1.09] + [1.01] * 8, c) == "RESOURCE_UNRESOLVED"
    assert classify_resource(base, control, [1.13] + [1.01] * 8, c) == "RESOURCE_MISMATCH"
    import numpy as np
    one = np.array([0.01, 0.02, 0.03])
    two = 2 * one
    report = paired_summary(one, two, 8600001, 200)
    assert report["dense_minus_k4"]["mean"] == 2 * report["sparse_minus_k4"]["mean"]
    assert report["dense_minus_k4"]["ci95"] == [2 * bound for bound in report["sparse_minus_k4"]["ci95"]]


def test_fixed_seed_scored_evaluation_reports_registered_interpretation_diagnostics():
    model = build_model("K4_score", 9_900_301, 9_900_302)
    batch = make_batch(8, 16, 9_900_303, ROOT / "configs/exp0011_matrices.json")
    metrics, arrays = pfstyle_study.evaluate(model, batch)
    diagnostics = metrics["score_diagnostics"]
    assert diagnostics["gaussian_head_mu_mse_vs_known_mode"] >= 0
    assert len(diagnostics["gaussian_head_mu_mse_by_mode"]) == 4
    assert all(value >= 0 for value in diagnostics["gaussian_head_mu_mse_by_mode"])
    assert 0.1 <= diagnostics["gaussian_head_sigma_mean"] <= 3.0
    assert len(diagnostics["gaussian_head_sigma_mean_by_mode"]) == 4
    assert diagnostics["state_pair_abs_cosine_final"] is not None
    assert abs(sum(metrics["predicted_mode_map_fraction"]) - 1) < 1e-12
    assert arrays["final_logits"].shape == (8, 4)
