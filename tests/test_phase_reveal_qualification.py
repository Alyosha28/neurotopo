"""Synthetic engineering tests only: no registered roots, seeds, or research updates."""

from pathlib import Path
from hashlib import sha256
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from neurotopo import phase_reveal_qualification as qualification
from neurotopo import phase_reveal_qualification_analysis as analysis
from neurotopo.phase_reveal_task import PhaseRevealBatch


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 4)
        self.graph_info = {"engineering": True}

    def forward(self, x: torch.Tensor, pre_index: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        return self.linear(x[:, pre_index - 1]), self.linear(x[:, -1])


def fake_batch(n: int, length: int, root: int, *, switch: int = 16,
               reliability: float = 0.75) -> PhaseRevealBatch:
    """Construct sealed arrays from an engineering root without task generation."""
    rng = np.random.default_rng(root)
    H = np.arange(n, dtype=np.int64) % 4
    cue = H.copy()
    cue[::4] = (cue[::4] + 1) % 4
    u = rng.standard_normal((n, length, 4))
    eps = rng.standard_normal((n, length, 4))
    y = 0.8 * eps
    prior = np.full((n, 4), (1-reliability)/3)
    prior[np.arange(n), cue] = reliability
    full = np.full((n, 4), 0.01)
    full[np.arange(n), H] = 0.97
    x = np.empty((n, length, 16), dtype=np.float32)
    x[:, :, 0::4] = u.astype(np.float32)
    x[:, :, 1::4] = y.astype(np.float32)
    x[:, :, 2::4] = (cue[:, None, None] == np.arange(4)).astype(np.float32)
    x[:, :, 3::4] = (np.arange(length) >= switch)[None, :, None]
    return PhaseRevealBatch(x, prior, full, H, cue, u, eps, y, switch, reliability)


def engineering_config() -> dict:
    return {
        "model": {"graph_seed_base": 100},
        "task": {"train_length": 24, "ood_length": 28, "cue_reliability": 0.75,
                 "switch_step": 16, "retained_cue_reliability": 0.6,
                 "retained_switch_step": 20},
        "training": {"betas": [0.9, 0.999], "adam_eps": 1e-8, "weight_decay": 0.0,
                     "grad_clip": 1.0, "batch_size": 4, "updates_per_arm_seed": 2,
                     "train_root_base": 1000, "monitor_root_base": 2000,
                     "monitor_steps": [1, 2]},
        "qualification": {"lr_menu": [0.001], "model_seeds": [7], "eval_examples": 8,
                          "tune_iid_root_base": 3000, "tune_ood_root_base": 4000,
                          "qual_iid_root_base": 5000, "qual_ood_root_base": 6000,
                          "iid_kl_per_seed_max_exclusive": 10.0,
                          "iid_kl_mean_max_exclusive": 10.0,
                          "ood_mean_windows": {"SC64": [0, 10], "D64_full": [0, 10]},
                          "ood_per_seed_windows": {"SC64": [0, 10], "D64_full": [0, 10]}},
        "resource": {"engineering_seed": 3, "engineering_update_cap": 4,
                     "tiny_overfit_updates_per_arm": 2,
                     "tiny_overfit_final_initial_ratio_max": 2.0,
                     "research_update_cap": 4},
        "task_information": {"examples": 8, "retained_examples": 8,
                             "root": 11, "retained_root": 12,
                             "bootstrap_seed": 13, "bootstrap_resamples": 8,
                             "h_fraction_window": [0.2, 0.3],
                             "cue_hit_window": [0.7, 0.8],
                             "oracle_uniform_kl_min": 0.0,
                             "wrong_cue_correction_count_min": 0,
                             "wrong_cue_correction_wilson_lower_min": 0.0,
                             "shortcut_kl_ci_lower_min": {"cue": 0.0, "last": 0.0,
                                                          "truncated_prehistory": 0.0},
                             "retained_correction_count_min": 0,
                             "retained_cue_kl_min": 0.0},
    }


def test_engineering_preflight_never_calls_task_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = engineering_config()
    monkeypatch.setattr(qualification, "build_model", lambda *_: TinyModel())
    monkeypatch.setattr(qualification, "make_batch", lambda *_args, **_kwargs: pytest.fail("task generator called"))
    monkeypatch.setattr(qualification, "_update", lambda *_args: 1.0)
    result = qualification.engineering_preflight(cfg, float("inf"))
    assert result["pass"]
    assert result["updates"] == 4


def test_separate_preflight_receipt_is_immutable_and_source_locked(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = engineering_config()
    cfg["resource"]["preflight_wall_seconds_cap"] = 300
    cfg["matrix_file"] = "unused-engineering-matrix.json"
    config = tmp_path / "dummy-engineering-config.json"
    config.write_text("{}")
    provenance = {"source_commit": "dummy-engineering-commit",
                  "frozen_hashes": {"dummy": "abc"},
                  "source_hashes": {"dummy.py": "def"},
                  "dependency_hashes": {}, "lock_sha256": "ghi"}
    monkeypatch.setattr(qualification, "locked_config", lambda *_: (cfg, {"dummy": "abc"}))
    monkeypatch.setattr(qualification, "_provenance", lambda *_: provenance)
    monkeypatch.setattr(qualification, "_commit_ancestor",
                        lambda _repo, earlier, later: earlier == "dummy-engineering-commit"
                        and later in ("dummy-engineering-commit", "archived-report-commit"))
    monkeypatch.setattr(qualification, "_committed_identical", lambda *_: True)
    monkeypatch.setattr(qualification, "engineering_preflight", lambda *_: {
        "updates": 4, "seconds": 0.01, "pass": True,
        "engineering_seed": 3, "learning_rate": 0.001,
        "arms": {arm: {"updates": 2, "ratio": 1.0} for arm in qualification.ARMS}})
    monkeypatch.setattr(qualification, "_engineering_oracle_check", lambda *_: [
        {"engineering_root": 42, "examples": 2, "length": 28,
         "switch_step": switch, "cue_reliability": reliability,
         "validation": {
            "source_sha256": "a" * 64, "full": 0.0,
            "last_frame": 0.0, "truncated": 0.0,
            "full_sha256": "b" * 64, "last_frame_sha256": "c" * 64,
            "truncated_sha256": "d" * 64}}
        for switch, reliability in ((16, 0.75), (20, 0.6))])
    monkeypatch.setattr(qualification, "make_batch", lambda *_args, **_kwargs: pytest.fail("task generator called"))
    result = qualification.run_engineering_preflight(config, repo=tmp_path)
    assert result["status"] == "PASS"
    with pytest.raises(ValueError, match="provenance"):
        qualification._verified_preflight(tmp_path, cfg, provenance, config)
    archived = qualification._verified_preflight(tmp_path, cfg, {**provenance,
                                           "source_commit": "archived-report-commit"}, config)
    assert archived["engineering"]["updates"] == 4
    assert len(archived["seal_sha256"]) == 64
    with pytest.raises(FileExistsError):
        qualification.run_engineering_preflight(config, repo=tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        qualification._verified_preflight(tmp_path, cfg, {**provenance,
                                              "source_commit": "unrelated-commit"}, config)
    monkeypatch.setattr(qualification, "_committed_identical", lambda *_: False)
    with pytest.raises(ValueError, match="provenance"):
        qualification._verified_preflight(tmp_path, cfg, {**provenance,
                                              "source_commit": "archived-report-commit"}, config)
    monkeypatch.setattr(qualification, "_committed_identical", lambda *_: True)
    late_repo = tmp_path / "late-engineering"
    late_repo.mkdir()
    late_config = late_repo / "dummy-engineering-config.json"
    late_config.write_text("{}")
    ticks = iter((0.0, 299.9, 299.9, 299.95, 300.2))
    monkeypatch.setattr(qualification, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    late_result = qualification.run_engineering_preflight(late_config, repo=late_repo)
    assert late_result["status"] == "ENGINEERING_ABORT"
    late_report_path, late_receipt_path = qualification._preflight_paths(late_repo, cfg)
    assert json.loads(late_report_path.read_text())["status"] == "PASS"
    late_receipt = json.loads(late_receipt_path.read_text())
    assert late_receipt["status"] == "PASS"
    assert late_receipt["seconds_after_report_fsync"] == pytest.approx(299.95)
    assert late_receipt["write_procedure"]["receipt"] == "exclusive_write_flush_fsync"
    late_seal = json.loads(qualification._seal_path(late_receipt_path).read_text())
    assert late_seal["status"] == "ENGINEERING_ABORT"
    assert late_seal["receipt_fsync_seconds"] == pytest.approx(300.2)
    with pytest.raises(ValueError, match="provenance"):
        qualification._verified_preflight(late_repo, cfg, provenance, late_config)


def test_task_information_raw_arrays_and_reanalysis(monkeypatch: pytest.MonkeyPatch,
                                                     tmp_path: Path) -> None:
    cfg = engineering_config()
    cfg["matrix_file"] = "unused-engineering"
    roots: list[int] = []
    def make(n: int, length: int, root: int, _matrix: Path, **kwargs: object) -> PhaseRevealBatch:
        roots.append(root)
        assert root in (11, 12)
        return fake_batch(n, length, root, switch=int(kwargs["switch_step"]),
                          reliability=float(kwargs["cue_reliability"]))
    monkeypatch.setattr(qualification, "make_batch", make)
    monkeypatch.setattr(qualification, "validate_oracles", lambda *_args: {"full": 0.0})
    monkeypatch.setattr(qualification, "last_frame_posterior",
                        lambda _u, _y, cue, _m, **kwargs: qualification.cue_only_posterior(
                            cue, kwargs.get("cue_reliability", 0.75)))
    monkeypatch.setattr(qualification, "truncated_prehistory_posterior",
                        lambda _u, _y, cue, _m, **kwargs: qualification.cue_only_posterior(
                            cue, kwargs.get("cue_reliability", 0.75)))
    measured = qualification.task_information(cfg, Path("unused-engineering"), tmp_path, float("inf"))
    assert measured["pass"] and roots == [11, 12]
    monkeypatch.setattr(analysis, "make_batch", make)
    monkeypatch.setattr(analysis, "last_frame_posterior", qualification.last_frame_posterior)
    monkeypatch.setattr(analysis, "truncated_prehistory_posterior", qualification.truncated_prehistory_posterior)
    manifest = {"config": cfg, "source_repo": str(tmp_path), "task_information": measured}
    independently = analysis.recompute_task_info(tmp_path, manifest)
    assert independently["pass"]
    assert independently["correction_count"] == measured["correction_count"]
    assert measured["source_arrays"]["main"]["x"]["shape"] == [8, 28, 16]
    assert measured["source_arrays"]["retained"]["eps"]["shape"] == [8, 28, 4]
    original_source_hash = measured["source_arrays"]["retained"]["u"]["sha256"]
    measured["source_arrays"]["retained"]["u"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source bytes"):
        analysis.recompute_task_info(tmp_path, manifest)
    measured["source_arrays"]["retained"]["u"]["sha256"] = original_source_hash
    with np.load(tmp_path / "task_information.npz", allow_pickle=False) as raw:
        assert raw["p_full"].shape == (8, 4)
        assert raw["retained_full"].shape == (8, 4)
        tampered = {key: raw[key] for key in raw.files}
    tampered["p_full"] = tampered["p_full"].copy()
    tampered["p_full"][0] = np.roll(tampered["p_full"][0], 1)
    np.savez_compressed(tmp_path / "task_information.npz", **tampered)
    measured["arrays_sha256"] = qualification.digest(tmp_path / "task_information.npz")
    with pytest.raises(ValueError, match="frozen-root"):
        analysis.recompute_task_info(tmp_path, manifest)


def test_fake_train_arm_uses_engineering_roots_and_no_optimizer_step(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = engineering_config()
    roots = []
    monkeypatch.setattr(qualification, "build_model", lambda *_: TinyModel())
    monkeypatch.setattr(qualification, "resource_account", lambda *_args: {
        "engineering": True, "dense_forward_linear_flops_per_batch": 1})
    monkeypatch.setattr(qualification, "_update", lambda *_args: 1.0)
    def make(n: int, length: int, root: int, _matrix: Path, **_kwargs: object) -> PhaseRevealBatch:
        roots.append(root)
        assert root in (71001, 71002, 2070, 2071)
        return fake_batch(n, length, root)
    monkeypatch.setattr(qualification, "make_batch", make)
    result = qualification.train_arm(cfg, "SC64", 7, 0.001, Path("unused-engineering"),
                                     tmp_path / "arm", float("inf"))
    assert result["status"] == "COMPLETED"
    assert result["updates"] == 2
    assert roots == [71001, 2070, 71002, 2071]
    assert len(result["checkpoints"]) == 2


def test_failed_optimizer_step_is_charged_without_real_update(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = engineering_config()
    monkeypatch.setattr(qualification, "build_model", lambda *_: TinyModel())
    monkeypatch.setattr(qualification, "resource_account", lambda *_: {
        "dense_forward_linear_flops_per_batch": 1})
    monkeypatch.setattr(qualification, "make_batch", lambda n, length, root, _matrix, **_kw:
                        fake_batch(n, length, root))
    monkeypatch.setattr(qualification, "_update", lambda *_args:
                        (_ for _ in ()).throw(qualification.CompletedOptimizerStepError("mock post-step failure")))
    row = qualification.train_arm(cfg, "SC64", 7, 0.001,
                                  Path("unused-engineering"), tmp_path / "failed", float("inf"))
    assert row["status"] == "CONTROL_UNRESOLVED"
    assert row["updates"] == 1
    assert row["train_examples"] == cfg["training"]["batch_size"]


def test_analysis_recomputes_logits_and_rejects_summary_tampering(tmp_path: Path) -> None:
    run = tmp_path
    arm_dir = run / "SC64" / "0.001" / "7"
    arm_dir.mkdir(parents=True)
    posterior = np.tile(np.array([[0.7, 0.1, 0.1, 0.1]]), (8, 1))
    logits = np.log(posterior)
    path = arm_dir / "tune_iid.npz"
    np.savez_compressed(path, oracle_pre=posterior, oracle_final=posterior,
                        logits_pre=logits, logits_final=logits)
    row = {"evaluations": {"tune_iid": {"root": 3007, "examples": 8,
            "pre_kl": 0.0, "final_kl": 0.0,
            "predictions_file": "tune_iid.npz", "predictions_sha256": qualification.digest(path)}}}
    result = analysis._read_evaluation(run, "SC64", 0.001, 7, "tune_iid", row,
                                       engineering_config())
    assert result["final_kl"] < 1e-12
    row["evaluations"]["tune_iid"]["final_kl"] = 0.2
    with pytest.raises(ValueError, match="differs"):
        analysis._read_evaluation(run, "SC64", 0.001, 7, "tune_iid", row,
                                  engineering_config())


def test_complete_control_numeric_audit_stays_pending_review(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = engineering_config()
    cfg["matrix_file"] = "unused-engineering"
    # Floating-point KL of identical arrays can be a tiny negative rounding error.
    for windows in ("ood_mean_windows", "ood_per_seed_windows"):
        for arm in qualification.ARMS:
            cfg["qualification"][windows][arm][0] = -1e-10
    posterior = np.tile(np.array([[0.7, 0.1, 0.1, 0.1]]), (8, 1))
    logits = np.log(posterior)
    rows = {}
    replay_arms = {}
    def make_eval(n: int, length: int, root: int, _matrix: Path, **_kwargs: object) -> PhaseRevealBatch:
        original = fake_batch(n, length, root)
        return PhaseRevealBatch(original.x, posterior, posterior, original.H, original.cue,
                                original.u, original.eps, original.y, original.switch_step,
                                original.cue_reliability)
    monkeypatch.setattr(analysis, "make_batch", make_eval)
    monkeypatch.setattr(analysis, "_registered_sc_mask", lambda _seed: np.stack([
        np.isin(np.arange(64), (node + np.arange(16)) % 64).astype(np.float32)
        for node in range(64)]))
    for arm in qualification.ARMS:
        arm_dir = tmp_path / arm / "0.001" / "7"
        arm_dir.mkdir(parents=True)
        degree = 16 if arm == "SC64" else 64
        base = np.ones((64, 64), dtype=np.float32) if arm == "D64_full" else np.zeros((64, 64), dtype=np.float32)
        if arm == "SC64":
            for node in range(64):
                base[node, (node + np.arange(16)) % 64] = 1
        input_base = np.ones((64, 16), dtype=np.float32) if arm == "D64_full" else np.zeros((64, 16), dtype=np.float32)
        if arm == "SC64":
            for group in range(4):
                input_base[group*16:(group+1)*16, group*4:(group+1)*4] = 1
        state = {"mask": torch.from_numpy(np.tile(base, (3, 1))),
                 "input_mask": torch.from_numpy(np.tile(input_base, (3, 1)))}
        for name, shape in (("weight_ih", (192, 16)), ("weight_hh", (192, 64)),
                            ("bias_ih", (192,)), ("bias_hh", (192,)),
                            ("readout_weight", (4, 64)), ("readout_bias", (4,))):
            state[name] = torch.zeros(shape)
        parameter_hasher = sha256()
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh",
                     "readout_weight", "readout_bias"):
            parameter_hasher.update(name.encode())
            parameter_hasher.update(state[name].numpy().tobytes())
        checkpoints = []
        for step in (1, 2):
            cp = arm_dir / f"checkpoint-{step}.pt"
            torch.save({"state_dict": state, "arm": arm, "seed": 7,
                        "lr": 0.001, "step": step}, cp)
            monitor = arm_dir / f"monitor-{step}.npz"
            np.savez_compressed(monitor, oracle_final=posterior, logits_final=logits)
            checkpoints.append({"step": step, "checkpoint_file": cp.name,
                                "checkpoint_sha256": qualification.digest(cp),
                                "prediction_file": monitor.name,
                                "prediction_sha256": qualification.digest(monitor),
                                "iid_final_kl": 0.0, "train_loss": 1.0,
                                "root": 2070 + step - 1})
        evaluations = {}
        for split, eval_base in (("tune_iid", 3000), ("tune_ood", 4000),
                            ("qual_iid", 5000), ("qual_ood", 6000)):
            path = arm_dir / f"{split}.npz"
            np.savez_compressed(path, oracle_pre=posterior, oracle_final=posterior,
                                logits_pre=logits, logits_final=logits)
            evaluations[split] = {"root": eval_base + 7, "examples": 8,
                                  "pre_kl": 0.0, "final_kl": 0.0,
                                  "input_sha256": qualification._array_hash(
                                      fake_batch(8, 24 if split.endswith("iid") else 28,
                                                 eval_base + 7).x),
                                  "predictions_file": path.name,
                                  "predictions_sha256": qualification.digest(path)}
        resources = {"allocated_parameters": 16004,
                     "effective_parameters": 4484 if arm == "SC64" else 16004,
                     "active_input_parameters": 768 if arm == "SC64" else 3072,
                     "active_recurrent_parameters": 3072 if arm == "SC64" else 12288,
                     "hidden_state_floats": 64, "pre_readout_cache_floats_per_example": 4,
                     "internal_steps_per_observation": 1,
                     "dense_forward_linear_flops_per_batch": 1,
                     "operator_shapes": {
                         "input_linear_per_token": {"weight": [192, 16]},
                         "recurrent_linear_per_token": {"weight": [192, 64]},
                         "query_readout": {"weight": [4, 64]}}}
        graph = {"diagonal_connections": 64, "strongly_connected": True,
                 "mask_sha256": sha256(base.astype(np.uint8).tobytes()).hexdigest(),
                 "seed": 107, "topology": "random" if arm == "SC64" else "dense",
                 "in_degree": [degree] * 64,
                 "out_degree": [degree] * 64,
                 "successful_swaps": 48000 if arm == "SC64" else 0}
        boundary = {}
        for step in (1, 2):
            root = 1000 + 7 * 10_000 + step
            batch = fake_batch(4, 24, root)
            boundary[str(step)] = {"root": root,
                "x_sha256": qualification._array_hash(batch.x),
                "oracle_pre_sha256": qualification._array_hash(batch.oracle_pre),
                "oracle_final_sha256": qualification._array_hash(batch.oracle_final)}
        rows[arm] = {"0.001": {"7": {"arm": arm, "seed": 7, "lr": 0.001,
                                        "status": "COMPLETED", "updates": 2,
                                        "resources": resources, "graph": graph,
                                        "train_examples": 8, "train_observation_steps": 8 * 24,
                                        "dense_forward_linear_flops": 2,
                                        "initial_parameters_sha256": "b" * 64,
                                        "final_parameters_sha256": parameter_hasher.hexdigest(),
                                        "boundary_batch_hashes": boundary,
                                        "checkpoints": checkpoints,
                                        "evaluations": evaluations}}}
        replay_arms[arm] = {"lr": 0.001,
                            "logits_pre_sha256": qualification._array_hash(logits),
                            "logits_final_sha256": qualification._array_hash(logits)}
    manifest = {"config": cfg, "source_repo": str(tmp_path),
                "arms": rows, "research_updates": 4,
                "task_information": {"root": 11, "retained_root": 12},
                "selected_lr": {arm: 0.001 for arm in qualification.ARMS},
                "zero_update_pair_replay": {"seed": 7, "optimization_updates": 0,
                                            "arms": replay_arms}}
    result = analysis.recompute_controls(tmp_path, manifest)
    assert result["status"] == "NUMERIC_GUARDS_PASS_PENDING_REVIEW"
    assert result["iid_adequacy"] and result["ood_headroom"]
    assert "SUPPORTED" not in json.dumps(result)
    tamper_row = rows["SC64"]["0.001"]["7"]["evaluations"]["tune_iid"]
    tamper_path = tmp_path / "SC64" / "0.001" / "7" / "tune_iid.npz"
    with np.load(tamper_path, allow_pickle=False) as original:
        arrays = {key: original[key] for key in original.files}
    arrays["oracle_final"] = np.roll(arrays["oracle_final"], 1, axis=1)
    np.savez_compressed(tamper_path, **arrays)
    tamper_row["predictions_sha256"] = qualification.digest(tamper_path)
    with pytest.raises(ValueError, match="frozen evaluation"):
        analysis.recompute_controls(tmp_path, manifest)
    np.savez_compressed(tamper_path, oracle_pre=posterior, oracle_final=posterior,
                        logits_pre=logits, logits_final=logits)
    tamper_row["predictions_sha256"] = qualification.digest(tamper_path)
    boundary = rows["SC64"]["0.001"]["7"]["boundary_batch_hashes"]["1"]
    original_hash = boundary["x_sha256"]
    boundary["x_sha256"] = "0" * 64
    with pytest.raises(analysis.ControlWeakenedError, match="paired controls"):
        analysis.recompute_controls(tmp_path, manifest)
    boundary["x_sha256"] = original_hash
    rows["SC64"]["0.001"]["7"]["resources"]["effective_parameters"] = 16004
    with pytest.raises(analysis.ControlWeakenedError):
        analysis.recompute_controls(tmp_path, manifest)


def test_worker_failure_fallback_is_inconclusive(tmp_path: Path) -> None:
    (tmp_path / "worker_failure.json").write_text(json.dumps({
        "error": "engineering only", "substatus": "ENGINEERING_ABORT"}))
    result = analysis.analyze(tmp_path)
    assert result["qualification_status"] == "ENGINEERING_ABORT"
    assert result["suggested_decision"] == "INCONCLUSIVE"
    assert result["gate1_credit"] is False
    assert (tmp_path / "analysis.json").is_file()
    (tmp_path / "worker_failure.json").write_text(json.dumps({
        "error": "synthetic timeout", "substatus": "INCOMPLETE_BUDGET"}))
    assert analysis.analyze(tmp_path)["qualification_status"] == "INCOMPLETE_BUDGET"


def test_task_failure_forbids_control_and_candidate_artifacts(tmp_path: Path) -> None:
    manifest = {"research_updates": 0, "rs4_task_updates": 0, "arms": {}}
    analysis._forbid_unregistered_work(tmp_path, manifest, task_failed=True)
    (tmp_path / "SC64").mkdir()
    with pytest.raises(analysis.ProtocolAbortError, match="control training"):
        analysis._forbid_unregistered_work(tmp_path, manifest, task_failed=True)
    (tmp_path / "SC64").rmdir()
    (tmp_path / "RS4").mkdir()
    with pytest.raises(analysis.ProtocolAbortError, match="RS4"):
        analysis._forbid_unregistered_work(tmp_path, manifest, task_failed=False)


def test_committed_blob_and_artifact_path_guards(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Engineering Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "engineering@example.invalid"], cwd=tmp_path, check=True)
    artifact = tmp_path / "sealed.json"
    artifact.write_text('{"status":"PASS"}\n')
    assert not qualification._committed_identical(tmp_path, artifact)
    subprocess.run(["git", "add", "sealed.json"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "seal engineering fixture"], cwd=tmp_path, check=True)
    assert qualification._committed_identical(tmp_path, artifact)
    assert analysis._safe_artifact(tmp_path, artifact) == artifact
    artifact.write_text('{"status":"ALTERED"}\n')
    assert not qualification._committed_identical(tmp_path, artifact)
    external = tmp_path / "outside.json"
    external.write_text("{}")
    escaped = tmp_path / "escaped.json"
    escaped.symlink_to(external)
    with pytest.raises(ValueError, match="escapes run"):
        analysis._safe_artifact(tmp_path, escaped)
    with pytest.raises(ValueError, match="escapes run"):
        analysis._safe_artifact(tmp_path, tmp_path / ".." / tmp_path.name / external.name)
