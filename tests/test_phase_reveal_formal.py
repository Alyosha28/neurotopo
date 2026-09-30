"""Engineering-only tests for the EXP-0014 stop and analysis boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neurotopo.phase_reveal_formal import (QualificationGateError, engineering_preflight,
                                           qualification_gate, resource_verdict, run,
                                           transplant_diagnostic, _write_json_fsync_new,
                                           _verified_preflight_commit,
                                           _verify_terminal_seal)
from neurotopo.phase_reveal_formal_analysis import (analyze, classify_complete,
                                                    kl_rows, linear_cka_blocks,
                                                    paired_bootstrap, _operation_seed,
                                                    _operation_verdict,
                                                    _replay_task_information,
                                                    _replay_operation_from_checkpoint,
                                                    _replay_control_prediction)
from neurotopo.phase_reveal_models import build_model
from neurotopo.phase_reveal_resources import resource_account
from neurotopo.phase_reveal_task import (make_batch, cue_only_posterior,
                                         last_frame_posterior,
                                         truncated_prehistory_posterior,
                                         validate_oracles)
from neurotopo.phase_reveal_qualification import evaluate
from neurotopo.phase_reveal_qualification import _state_hash


ROOT = Path(__file__).resolve().parents[1]
LOCKED = (
    "configs/exp0014_v3.json", "configs/exp0013_v3.json",
    "configs/exp0013_matrices.json", "configs/exp0013_matrices_receipt.json",
    "experiments/EXP-0013/v3.md", "experiments/EXP-0014/v3.md",
    "research/registrations/G1-RS-0001_protocol_pair_v2.json",
    "research/registrations/G1-RS-0001_config_lock.json",
)


def _virtual_locked_root(tmp_path: Path) -> Path:
    root = tmp_path / "virtual_project"
    for relative in LOCKED:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    return root


def test_qualification_gate_stops_before_any_formal_root(tmp_path: Path) -> None:
    root = _virtual_locked_root(tmp_path)
    with pytest.raises(QualificationGateError, match="ResearchLoop ledger"):
        qualification_gate(root, root / "configs/exp0014_v3.json", root / "empty_loop")
    assert not (root / "artifacts/EXP-0014").exists()


def test_normal_source_stop_writes_analyzable_zero_update_run(tmp_path: Path) -> None:
    root = _virtual_locked_root(tmp_path)
    output = tmp_path / "stopped"
    record = run(root / "configs/exp0014_v3.json", repo=root, output=output,
                 loop_root=root / "empty_loop")
    assert record["status"] == "QUALIFICATION_SOURCE_NOT_READY"
    assert record["research_updates"] == record["formal_task_roots_read"] == 0
    analysis = analyze(output)
    assert analysis["decision"] == "INCONCLUSIVE"
    assert not (output / "formal").exists()


def test_preflight_evidence_file_is_immutable(tmp_path: Path) -> None:
    import time

    path = tmp_path / "validation" / "report.json"
    _write_json_fsync_new(path, {"status": "PASS"})
    assert json.loads(path.read_text()) == {"status": "PASS"}
    with pytest.raises(FileExistsError):
        _write_json_fsync_new(path, {"status": "FAIL"})
    receipt = tmp_path / "validation" / "late_receipt.json"
    status = {"status": "PASS"}
    assert not _write_json_fsync_new(receipt, status, deadline=time.monotonic() - 1)
    assert status["status"] == json.loads(receipt.read_text())["status"] == "INCOMPLETE_BUDGET"


def test_preflight_git_gate_rejects_ignored_and_changed_receipts(tmp_path: Path) -> None:
    import subprocess

    repo = tmp_path / "git_checkout"
    repo.mkdir()
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=repo, check=True,
                                capture_output=True, text=True)
        return result.stdout.strip()
    git("init", "-q")
    git("config", "user.name", "Engineering Test")
    git("config", "user.email", "engineering@example.invalid")
    (repo / ".gitignore").write_text("artifacts/validation/\n")
    (repo / "source.txt").write_text("source X\n")
    git("add", ".gitignore", "source.txt")
    git("commit", "-qm", "source X")
    source_commit = git("rev-parse", "HEAD")
    report = repo / "artifacts/validation/report.json"
    receipt = repo / "artifacts/validation/receipt.json"
    terminal = repo / "artifacts/validation/terminal.json"
    _write_json_fsync_new(report, {"status": "PASS"})
    _write_json_fsync_new(receipt, {"report": "report.json"})
    (repo / "later.txt").write_text("Y\n")
    git("add", "later.txt")
    git("commit", "-qm", "later Y without evidence")
    with pytest.raises(QualificationGateError, match="missing|absent from HEAD"):
        _verified_preflight_commit(repo, source_commit, report, receipt, terminal)
    git("add", "-f", "artifacts/validation/report.json", "artifacts/validation/receipt.json")
    git("commit", "-qm", "commit incomplete evidence")
    with pytest.raises(QualificationGateError, match="missing|absent from HEAD"):
        _verified_preflight_commit(repo, source_commit, report, receipt, terminal)
    _write_json_fsync_new(terminal, {"status": "PASS"})
    git("add", "-f", "artifacts/validation/terminal.json")
    git("commit", "-qm", "commit terminal seal")
    _verified_preflight_commit(repo, source_commit, report, receipt, terminal)
    with pytest.raises(QualificationGateError, match="later committed"):
        _verified_preflight_commit(repo, git("rev-parse", "HEAD"), report, receipt, terminal)
    receipt.write_text('{"report":"changed"}\n')
    with pytest.raises(QualificationGateError, match="HEAD/worktree bytes differ"):
        _verified_preflight_commit(repo, source_commit, report, receipt, terminal)


def test_terminal_seal_rejects_post_receipt_overrun_even_without_marker(tmp_path: Path) -> None:
    import hashlib

    report_path = tmp_path / "report.json"
    receipt_path = tmp_path / "receipt.json"
    report = {"commit": "a" * 40, "status": "PASS"}
    receipt = {"registered_cap_seconds": 300, "status": "PASS"}
    _write_json_fsync_new(report_path, report)
    _write_json_fsync_new(receipt_path, receipt)
    terminal = {"experiment_id": "EXP-0014", "version": "v3", "status": "PASS",
                "source_commit": report["commit"],
                "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                "registered_cap_seconds": 300,
                "seconds_after_receipt_fsync": 299.0,
                "publication": {"budget_class": "publication_metadata_outside_engineering_cap"}}
    _verify_terminal_seal(report_path, receipt_path, terminal, report, receipt)
    terminal["seconds_after_receipt_fsync"] = 300.01
    with pytest.raises(QualificationGateError, match="over budget"):
        _verify_terminal_seal(report_path, receipt_path, terminal, report, receipt)


def test_qualification_gate_detects_locked_byte_change(tmp_path: Path) -> None:
    root = _virtual_locked_root(tmp_path)
    protocol = root / "experiments/EXP-0013/v3.md"
    protocol.write_bytes(protocol.read_bytes() + b"\n")
    with pytest.raises(QualificationGateError, match="frozen byte mismatch"):
        qualification_gate(root, root / "configs/exp0014_v3.json", root / "empty_loop")


def test_qualification_gate_accepts_main_ledger_path_in_isolated_checkout(tmp_path: Path) -> None:
    """Path rebasing reaches the raw gate, while fabricated evidence never admits it."""
    import hashlib

    main_root = _virtual_locked_root(tmp_path / "main")
    clean_root = _virtual_locked_root(tmp_path / "clean")
    loop_root = main_root / "artifacts/auto_research/live"
    loop_root.mkdir(parents=True)
    relative_manifest = Path("artifacts/EXP-0013/v3/manifest.json")
    relative_analysis = Path("artifacts/EXP-0013/v3/analysis.json")
    raw_manifest = {"experiment_id": "EXP-0013", "version": "v3",
                    "source_hashes": {"configs/exp0013_v3.json": hashlib.sha256(
                        (main_root / "configs/exp0013_v3.json").read_bytes()).hexdigest()},
                    "lock_sha256": hashlib.sha256((main_root /
                        "research/registrations/G1-RS-0001_config_lock.json").read_bytes()).hexdigest()}
    payload = json.dumps(raw_manifest).encode()
    for root in (main_root, clean_root):
        target = root / relative_manifest
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        analysis = root / relative_analysis
        analysis.write_text(json.dumps({"source_manifest_sha256": hashlib.sha256(payload).hexdigest()}))

    def evidence(path: Path) -> dict:
        return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size}

    phase_paths = {}
    for phase in ("PRIOR_ART", "CRITIQUE", "REGISTERED", "VALIDATED", "RUNNING"):
        path = main_root / f"{phase}.json"
        path.write_text("{}")
        phase_paths[phase] = path
    phase_paths["ANALYZED"] = main_root / relative_analysis
    review = main_root / "review.json"
    review.write_text(json.dumps({"verdict": "CONTROL_STRONG_ENOUGH"}))
    phase_paths["REVIEWED"] = review
    decision = main_root / "decision.json"
    decision.write_text(json.dumps({"decision": "SUPPORTED", "substatus": "CONTROL_STRONG_ENOUGH"}))
    phase_paths["DECIDED"] = decision
    entries = {phase: evidence(path) for phase, path in phase_paths.items()}
    packet = main_root / "packet.json"
    packet.write_text(json.dumps({"design": {"sha256": entries["REGISTERED"]["sha256"]},
                                  "action": {"sha256": entries["ANALYZED"]["sha256"]}}))
    reflection_review = main_root / "reflection_review.json"
    reflection_review.write_text("{}")
    reaction = main_root / "reaction.json"
    reaction.write_text("{}")
    records = {key: evidence(path) for key, path in
               (("packet", packet), ("review", reflection_review), ("reaction", reaction))}
    events = []
    for phase in ("QUESTION", "PRIOR_ART", "CRITIQUE", "REGISTERED", "VALIDATED",
                  "RUNNING", "ANALYZED", "REVIEWED", "DECIDED"):
        current = {"id": "ITER-0014", "experiment": "EXP-0013", "evidence": entries,
                   "decision": "SUPPORTED"}
        events.append({"kind": "begin" if phase == "QUESTION" else "advance",
                       "experiment": "EXP-0013" if phase == "QUESTION" else None,
                       "progress": {"phase": phase, "current": current}})
    events.append({"kind": "reflection", "progress": {"phase": "DECIDED",
                   "current": {"id": "ITER-0014", "reflection": {"records": records}}}})
    (loop_root / "iteration_log.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")
    with pytest.raises(QualificationGateError, match="raw-array qualification audit failed"):
        qualification_gate(clean_root, clean_root / "configs/exp0014_v3.json", loop_root)


def test_kl_bootstrap_and_frozen_priority() -> None:
    p = np.array([[0.75, 0.1, 0.1, 0.05], [0.05, 0.1, 0.1, 0.75]])
    logits = np.log(p)
    assert np.max(np.abs(kl_rows(p, logits))) < 1e-12
    interval = paired_bootstrap(np.arange(10, dtype=float), 9_920_014, 100)
    assert interval["mean"] == 4.5
    assert interval["ci95"][0] < interval["ci95"][1]
    outcome = classify_complete((0.01, 0.04), (0.1, 0.2), (0.05, 0.1),
                                operation="FAIL", retained_positive=True,
                                replication_positive=True, learned_and_headroom=True,
                                raw_complete=True)
    assert outcome["decision"] == "FALSIFIED"
    assert classify_complete((0.06, 0.08), (0.1, 0.2), (0.05, 0.1),
                             operation="FAIL", retained_positive=True,
                             replication_positive=True, learned_and_headroom=True,
                             raw_complete=True)["substatus"] == "OPERATION_FAILED_OR_UNRESOLVED"


def test_virtual_resource_and_transplant_are_task_free() -> None:
    models = {arm: build_model(arm, 71, 14_000_071)
              for arm in ("RS4", "SC64", "D64_full")}
    accounts = {arm: resource_account(model, 32, 128)
                for arm, model in models.items()}
    config = json.loads((ROOT / "configs/exp0014_v3.json").read_text())
    ratios = {path: [1.01] * 3 for path in ("train32", "infer32", "infer48")}
    assert resource_verdict(config, accounts, ratios, 0.001, 0.001)["status"] == "PASS"
    ratios["train32"][1] = 1.13
    assert resource_verdict(config, accounts, ratios, 0.001, 0.001)["status"] == "RESOURCE_MISMATCH"
    x = torch.randn(8, 48, 16)
    oracle = np.full((8, 4), 0.25)
    result = transplant_diagnostic(models["RS4"], x, oracle, block=1,
                                   permutation_seed=9_930_015)
    assert result["non_target_t16_max"] == 0
    assert result["non_target_t48_max"] < 1e-6
    assert result["h16_original"].shape == (8, 64)
    assert result["transplant_logits"].shape == (8, 4)


def test_random_only_preflight_smoke() -> None:
    import time

    config = json.loads((ROOT / "configs/exp0014_v3.json").read_text())
    config["resource"].update(timing_blocks=1, timing_repetitions=1,
                              warmup_repetitions=1, tiny_overfit_updates_per_arm=2,
                              tiny_overfit_final_initial_ratio_max=100,
                              preflight_wall_seconds_cap=60)
    report = engineering_preflight(config, 0.001, time.monotonic() + 60)
    assert report["engineering_updates"] == 15
    assert set(report["operator_profile"]["RS4"]) == {"train32", "infer32", "infer48"}
    assert report["accounts"]["RS4"]["effective_parameters"] == 4484


def test_operation_archive_recomputes_random_tensor_diagnostics(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs/exp0014_v3.json").read_text())
    model = build_model("RS4", 71, 14_000_071)
    x = torch.randn(8, 48, 16)
    oracle = np.full((8, 4), 0.25)
    output = tmp_path / "operation"
    output.mkdir()
    records = {}
    original = None
    import hashlib

    for block in range(4):
        result = transplant_diagnostic(model, x, oracle, block,
                                       config["operation"]["prefix_permutation_seed_base"] + block)
        path = output / f"block-{block}.npz"
        np.savez_compressed(path, **{name: value for name, value in result.items()
                                    if isinstance(value, np.ndarray)})
        records[str(block)] = {**{name: value for name, value in result.items()
                                  if not isinstance(value, np.ndarray)},
                               "arrays_file": path.name,
                               "arrays_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if original is None:
            original = result["original_logits"]
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "arm": "RS4", "seed": 71,
                "step": config["training"]["updates_per_arm_seed"]}, checkpoint)
    row = {"operation_dir": "operation", "operation": records,
           "checkpoints": [{"checkpoint_file": checkpoint.name,
                            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}],
           "final_state_sha256": _state_hash(model)}
    checked = _operation_seed(tmp_path, row,
                              oracle, original, config)
    assert set(checked["blocks"]) == {"0", "1", "2", "3"}
    assert checked["zero_leakage"]
    _replay_operation_from_checkpoint(tmp_path, row, 71, x.numpy(), oracle, config)
    with np.load(output / "block-2.npz", allow_pickle=False) as archived:
        altered = {name: np.array(archived[name]) for name in archived.files}
    altered["transplant_logits"][0, 0] += 0.1
    np.savez_compressed(output / "block-2.npz", **altered)
    records["2"]["arrays_sha256"] = hashlib.sha256((output / "block-2.npz").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="checkpoint replay"):
        _replay_operation_from_checkpoint(tmp_path, row, 71, x.numpy(), oracle, config)


def test_operation_cka_and_worker_failure_fallback(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    state = rng.normal(size=(128, 4, 16))
    audit = linear_cka_blocks(state)
    assert not audit["collapsed"]
    assert audit["max_cka"] < 0.98
    stopped = tmp_path / "stopped"
    stopped.mkdir()
    (stopped / "worker_failure.json").write_text(json.dumps({"substatus": "INCOMPLETE_BUDGET"}))
    result = analyze(stopped)
    assert result["decision"] == "INCONCLUSIVE"
    assert result["substatus"] == "INCOMPLETE_BUDGET"
    assert not result["effect_computed"]


def test_operation_seed_counts_and_common_bootstrap_guard() -> None:
    config = json.loads((ROOT / "configs/exp0014_v3.json").read_text())
    def row(loss: float) -> dict:
        return {"operation": {"blocks": {str(k): {
            "state_change": 0.1, "centered_logit_change": 0.1,
            "transplant_kl_increase": loss, "lesion_kl_increase": loss,
        } for k in range(4)}, "redundancy": {"variance_by_block": [0.1] * 4,
                                             "max_cka": 0.1, "pass": True},
                              "zero_leakage": True}}
    values = {"formal": {str(k): row(0.02) for k in range(10)},
              "replication": {str(k): row(0.02) for k in range(5)}}
    assert _operation_verdict(values, config)["status"] == "PASS"
    values["formal"]["0"] = row(-0.2)
    assert _operation_verdict(values, config)["status"] == "FAIL"


def test_qualification_task_information_root_replay_catches_tamper() -> None:
    config = json.loads((ROOT / "configs/exp0013_v3.json").read_text())
    config["task_information"].update(root=42, retained_root=43, examples=2,
                                       retained_examples=2)
    matrix = ROOT / config["matrix_file"]
    task = config["task"]
    main = make_batch(2, 48, 42, matrix)
    retained = make_batch(2, 48, 43, matrix,
                          cue_reliability=task["retained_cue_reliability"],
                          switch_step=task["retained_switch_step"])
    raw = {"H": main.H, "cue": main.cue, "p_pre": main.oracle_pre,
           "p_full": main.oracle_final,
           "p_cue": cue_only_posterior(main.cue, task["cue_reliability"]),
           "p_last": last_frame_posterior(main.u, main.y, main.cue, matrix),
           "p_truncated": truncated_prehistory_posterior(main.u, main.y, main.cue, matrix),
           "retained_H": retained.H, "retained_cue": retained.cue,
           "retained_pre": retained.oracle_pre,
           "retained_full": retained.oracle_final,
           "retained_cue_posterior": cue_only_posterior(
               retained.cue, task["retained_cue_reliability"])}
    record = {"root": 42, "retained_root": 43,
              "oracle_check": validate_oracles(main, matrix)}
    _replay_task_information(raw, config, matrix, record)
    tampered = {**raw, "p_full": raw["p_full"].copy()}
    tampered["p_full"][0, 0] += 1e-5
    with pytest.raises(ValueError, match="p_full differs"):
        _replay_task_information(tampered, config, matrix, record)


def test_qualification_control_logits_bind_to_regenerated_engineering_input() -> None:
    model = build_model("SC64", 71, 14_000_071)
    batch = make_batch(8, 32, 42, ROOT / "configs/exp0013_matrices.json")
    result = evaluate(model, batch)
    arrays = {name: value for name, value in result.items() if isinstance(value, np.ndarray)}
    _replay_control_prediction(model, batch.x, arrays)
    altered_x = batch.x.copy()
    altered_x[0, 0, 0] += 0.5
    with pytest.raises(ValueError, match="frozen-root input replay"):
        _replay_control_prediction(model, altered_x, arrays)
    altered_arrays = {**arrays, "logits_final": arrays["logits_final"].copy()}
    altered_arrays["logits_final"][0, 0] += 0.1
    with pytest.raises(ValueError, match="frozen-root input replay"):
        _replay_control_prediction(model, batch.x, altered_arrays)
