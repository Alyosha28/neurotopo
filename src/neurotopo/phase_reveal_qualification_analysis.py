"""Raw-array numerical audit for EXP-0013. Independent review follows this stage."""

from __future__ import annotations

import argparse
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .phase_reveal_qualification import ARMS, _array_hash, _bootstrap_lower, _commit_ancestor, _committed_identical, _kl, _source_array_evidence, _wilson_lower, digest, write_json
from .phase_reveal_task import (
    PhaseRevealBatch, cue_only_posterior, last_frame_posterior, make_batch,
    truncated_prehistory_posterior,
)
from .models import recurrent_mask


class ControlWeakenedError(ValueError):
    """Registered single-stream topology or resource identity was weakened."""


class ProtocolAbortError(ValueError):
    """A stopped qualification wrote forbidden candidate/control work."""


def _forbid_unregistered_work(run: Path, manifest: dict[str, Any], *, task_failed: bool) -> None:
    if manifest.get("rs4_task_updates") != 0 or any(
        child.is_dir() and child.name.casefold() == "rs4" for child in run.rglob("*")
    ):
        raise ProtocolAbortError("RS4 task work exists in the qualification run")
    if task_failed and (
        manifest.get("research_updates") != 0 or manifest.get("arms") != {} or
        any((run / arm).exists() for arm in ARMS) or
        any(run.rglob("*.pt"))
    ):
        raise ProtocolAbortError("task-information failure was followed by control training artifacts")


def _probabilities(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[1] != 4 or not np.isfinite(logits).all():
        raise ValueError("invalid raw logits")
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _read_npz(path: Path, expected_sha: str, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    if digest(path) != expected_sha:
        raise ValueError(f"raw array SHA differs: {path}")
    with np.load(path, allow_pickle=False) as data:
        if not all(field in data for field in fields):
            raise ValueError(f"raw array field missing: {path}")
        return {field: data[field] for field in fields}


def _safe_artifact(root: Path, path: Path) -> Path:
    root = root.resolve()
    if (".." in path.parts or not path.is_relative_to(root) or
        not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file()):
        raise ValueError(f"artifact escapes run or is not an ordinary file: {path}")
    if any(part.is_symlink() for part in (path, *path.parents) if part != root and part.is_relative_to(root)):
        raise ValueError(f"artifact has a symlinked path component: {path}")
    return path


def recompute_preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    source = Path(manifest["source_repo"])
    evidence = manifest["engineering_preflight"]
    report_path = _safe_artifact(source, source / evidence["file"])
    receipt_path = _safe_artifact(source, source / evidence["receipt_file"])
    seal_path = _safe_artifact(source, source / evidence["seal_file"])
    if (digest(report_path) != evidence["sha256"] or
        digest(receipt_path) != evidence["receipt_sha256"] or
        digest(seal_path) != evidence["seal_sha256"] or
        not all(_committed_identical(source, p) for p in (report_path, receipt_path, seal_path))):
        raise ValueError("engineering preflight raw artifact hash differs")
    report = json.loads(report_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    seal = json.loads(seal_path.read_text())
    cfg = manifest["config"]
    if (receipt.get("report_sha256") != evidence["sha256"] or
        seal.get("report_sha256") != evidence["sha256"] or
        seal.get("receipt_sha256") != evidence["receipt_sha256"] or
        seal.get("status") != "PASS" or
        seal.get("source_commit") != report.get("source_commit") or
        seal.get("receipt_fsync_seconds", float("inf")) >
            cfg["resource"]["preflight_wall_seconds_cap"] or
        seal.get("receipt_fsync_seconds", -1) < receipt.get("seconds_after_report_fsync", float("inf")) or
        seal.get("receipt_write_seconds", -1) < 0 or
        abs(seal["receipt_fsync_seconds"] - receipt["seconds_after_report_fsync"] -
            seal["receipt_write_seconds"]) > 1e-6 or
        seal.get("write_procedure") != "exclusive_write_flush_fsync" or
        receipt.get("status") != "PASS" or report.get("status") != "PASS" or
        receipt.get("seconds_after_report_fsync", float("inf")) >
            cfg["resource"]["preflight_wall_seconds_cap"] or
        receipt.get("seconds_after_report_fsync", -1) < report.get("seconds", float("inf")) or
        receipt.get("report_write_seconds", -1) < 0 or
        abs(receipt["seconds_after_report_fsync"] - report["seconds"] -
            receipt["report_write_seconds"]) > 1e-6 or
        receipt.get("write_procedure") != {"report": "exclusive_write_flush_fsync",
                                           "receipt": "exclusive_write_flush_fsync"} or
        report.get("kind") != "ENGINEERING_ONLY" or
        report.get("config_sha256") != manifest["frozen_hashes"]["configs/exp0013_v3.json"] or
        evidence.get("source_commit") != report.get("source_commit") or
        receipt.get("source_commit") != report.get("source_commit") or
        report.get("source_commit") == manifest["source_commit"] or
        not _commit_ancestor(source, report.get("source_commit", ""), manifest["source_commit"]) or
        any(report.get(key) != manifest[key] for key in
            ("frozen_hashes", "source_hashes", "dependency_hashes", "lock_sha256")) or
        report["engineering"].get("updates") != cfg["resource"]["engineering_update_cap"] or
        report["engineering"].get("engineering_seed") != cfg["resource"]["engineering_seed"] or
        report["engineering"].get("learning_rate") != max(cfg["qualification"]["lr_menu"]) or
        len(report.get("oracle_checks", [])) != 2 or
        [row.get("switch_step") for row in report.get("oracle_checks", [])] !=
            [cfg["task"]["switch_step"], cfg["task"]["retained_switch_step"]] or
        any(row.get("engineering_root") != 42 or row.get("examples") != 2 or
            row.get("length") != cfg["task"]["ood_length"] or
            row.get("cue_reliability") != reliability or
            any(len(row.get("validation", {}).get(key, "")) != 64 for key in
                ("source_sha256", "full_sha256", "last_frame_sha256", "truncated_sha256")) or
            max(row["validation"].get(name, float("inf")) for name in
                ("full", "last_frame", "truncated")) > 1e-10
            for row, reliability in zip(report.get("oracle_checks", []),
                 (cfg["task"]["cue_reliability"], cfg["task"]["retained_cue_reliability"]),
                 strict=True)) or
        any(report["engineering"].get("arms", {}).get(arm, {}).get("updates") !=
            cfg["resource"]["tiny_overfit_updates_per_arm"] or
            report["engineering"]["arms"][arm].get("ratio", float("inf")) >
            cfg["resource"]["tiny_overfit_final_initial_ratio_max"] for arm in ARMS) or
        report.get("seconds", float("inf")) > cfg["resource"]["preflight_wall_seconds_cap"] or
        not report["engineering"].get("pass")):
        raise ValueError("engineering preflight raw receipt/PASS differs")
    return {"pass": True, "report_sha256": evidence["sha256"],
            "receipt_sha256": evidence["receipt_sha256"],
            "seal_sha256": evidence["seal_sha256"],
            "updates": report["engineering"]["updates"]}


def recompute_task_info(run: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    cfg = manifest["config"]["task_information"]
    task_cfg = manifest["config"]["task"]
    record = manifest["task_information"]
    if record["root"] != cfg["root"] or record["retained_root"] != cfg["retained_root"]:
        raise ValueError("task-information recorded roots differ from frozen config")
    if record["arrays_file"] != "task_information.npz":
        raise ValueError("unexpected task-information array path")
    arrays = _read_npz(_safe_artifact(run, run / record["arrays_file"]), record["arrays_sha256"],
        ("H", "cue", "p_pre", "p_full", "p_cue", "p_last", "p_truncated",
         "retained_H", "retained_cue", "retained_pre", "retained_full", "retained_cue_posterior",
         "retained_last", "retained_truncated"))
    task_cfg = manifest["config"]["task"]
    matrix = Path(manifest["source_repo"]) / manifest["config"]["matrix_file"]
    for prefix, n, length, root, reliability, switch in (
        ("", cfg["examples"], task_cfg["ood_length"], cfg["root"],
         task_cfg["cue_reliability"], task_cfg["switch_step"]),
        ("retained_", cfg["retained_examples"], task_cfg["ood_length"], cfg["retained_root"],
         task_cfg["retained_cue_reliability"], task_cfg["retained_switch_step"]),
    ):
        generated = make_batch(n, length, root, matrix,
                               cue_reliability=reliability, switch_step=switch)
        source_key = "retained" if prefix else "main"
        if record.get("source_arrays", {}).get(source_key) != _source_array_evidence(generated):
            raise ValueError(f"task-information frozen-root source bytes differ: {source_key}")
        expected = {"H": generated.H, "cue": generated.cue,
                    "pre" if prefix else "p_pre": generated.oracle_pre,
                    "full" if prefix else "p_full": generated.oracle_final,
                    "cue_posterior" if prefix else "p_cue": cue_only_posterior(generated.cue, reliability),
                    "last" if prefix else "p_last": last_frame_posterior(
                        generated.u, generated.y, generated.cue, matrix,
                        switch_step=switch, cue_reliability=reliability),
                    "truncated" if prefix else "p_truncated": truncated_prehistory_posterior(
                        generated.u, generated.y, generated.cue, matrix,
                        switch_step=switch, cue_reliability=reliability)}
        for suffix, value in expected.items():
            if not np.array_equal(arrays[prefix + suffix], value):
                raise ValueError(f"task-information frozen-root array differs: {prefix}{suffix}")
    H, cue, pre, full, cue_p = (arrays[k] for k in ("H", "cue", "p_pre", "p_full", "p_cue"))
    if len(H) != cfg["examples"] or len(arrays["retained_H"]) != cfg["retained_examples"]:
        raise ValueError("task-information sample count differs")
    for key in ("p_pre", "p_full", "p_cue", "p_last", "p_truncated",
                "retained_pre", "retained_full", "retained_cue_posterior",
                "retained_last", "retained_truncated"):
        probabilities = arrays[key]
        if probabilities.ndim != 2 or probabilities.shape[1] != 4 or not np.isfinite(probabilities).all() or np.any(probabilities <= 0) or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-10):
            raise ValueError(f"invalid raw task posterior {key}")
    if np.any(H < 0) or np.any(H >= 4) or np.any(cue < 0) or np.any(cue >= 4):
        raise ValueError("invalid raw H/cue")
    for index, cue_key, posterior_key, reliability in (
        (H, "cue", "p_cue", task_cfg["cue_reliability"]),
        (arrays["retained_H"], "retained_cue", "retained_cue_posterior",
         task_cfg["retained_cue_reliability"]),
    ):
        observed_cue = arrays[cue_key]
        if observed_cue.shape != index.shape or np.any(observed_cue < 0) or np.any(observed_cue >= 4):
            raise ValueError("invalid raw cue shape or values")
        expected_prior = np.full((len(index), 4), (1-reliability)/3)
        expected_prior[np.arange(len(index)), observed_cue] = reliability
        if not np.allclose(arrays[posterior_key], expected_prior, rtol=0, atol=1e-12):
            raise ValueError("raw cue posterior differs from frozen reliability")
    frac = (np.bincount(H, minlength=4) / len(H)).tolist()
    hit = float(np.mean(H == cue))
    wrong = H != cue
    corrected = wrong & (np.argmax(pre, axis=1) == cue) & (np.argmax(full, axis=1) == H)
    rng = np.random.Generator(np.random.PCG64(cfg["bootstrap_seed"]))
    shortcut = {}
    for name, key in (("cue", "p_cue"), ("last", "p_last"),
                      ("truncated_prehistory", "p_truncated")):
        values = _kl(full, arrays[key])
        shortcut[name] = {"mean_kl": float(values.mean()),
                          "bootstrap_ci95_lower": _bootstrap_lower(values, cfg["bootstrap_resamples"], rng)}
    rh, rc, rp, rf = (arrays[k] for k in ("retained_H", "retained_cue", "retained_pre", "retained_full"))
    retained_corrected = (rh != rc) & (np.argmax(rp, axis=1) == rc) & (np.argmax(rf, axis=1) == rh)
    result = {"h_fractions": frac, "cue_hit": hit,
              "pre_prior_max_error": float(np.max(np.abs(pre-cue_p))),
              "mean_full_uniform_kl": float(_kl(full, np.full_like(full, 0.25)).mean()),
              "wrong_cue_count": int(wrong.sum()), "correction_count": int(corrected.sum()),
              "correction_wilson_lower": _wilson_lower(int(corrected.sum()), int(wrong.sum())),
              "shortcuts": shortcut,
              "retained_correction_count": int(retained_corrected.sum()),
              "retained_mean_full_cue_kl": float(_kl(rf, arrays["retained_cue_posterior"]).mean())}
    result["pass"] = bool(
        all(cfg["h_fraction_window"][0] <= value <= cfg["h_fraction_window"][1] for value in frac) and
        cfg["cue_hit_window"][0] <= hit <= cfg["cue_hit_window"][1] and
        result["pre_prior_max_error"] <= 1e-10 and
        result["mean_full_uniform_kl"] >= cfg["oracle_uniform_kl_min"] and
        result["correction_count"] >= cfg["wrong_cue_correction_count_min"] and
        result["correction_wilson_lower"] >= cfg["wrong_cue_correction_wilson_lower_min"] and
        all(shortcut[name]["bootstrap_ci95_lower"] >= threshold
            for name, threshold in cfg["shortcut_kl_ci_lower_min"].items()) and
        result["retained_correction_count"] >= cfg["retained_correction_count_min"] and
        result["retained_mean_full_cue_kl"] >= cfg["retained_cue_kl_min"])
    return result


def _read_evaluation(run: Path, arm: str, lr: float, seed: int,
                     split: str, row: dict[str, Any], cfg: dict[str, Any],
                     regenerated: dict[str, Any] | None = None) -> dict[str, Any]:
    evidence = row["evaluations"][split]
    root = cfg["qualification"][split + "_root_base"] + seed
    if evidence["root"] != root or evidence["examples"] != cfg["qualification"]["eval_examples"]:
        raise ValueError(f"wrong {split} root or sample count")
    path = run / arm / str(lr) / str(seed) / evidence["predictions_file"]
    if path.name != split + ".npz":
        raise ValueError("unexpected prediction path")
    arrays = _read_npz(_safe_artifact(run, path), evidence["predictions_sha256"],
                       ("oracle_pre", "oracle_final", "logits_pre", "logits_final"))
    expected = cfg["qualification"]["eval_examples"]
    if any(value.shape != (expected, 4) for value in arrays.values()):
        raise ValueError("wrong raw prediction shape")
    if regenerated is not None and (
        not np.array_equal(arrays["oracle_pre"], regenerated["oracle_pre"]) or
        not np.array_equal(arrays["oracle_final"], regenerated["oracle_final"]) or
        evidence.get("input_sha256") != regenerated["input_sha256"]
    ):
        raise ValueError("frozen evaluation root/input/oracle differs")
    pre = float(_kl(arrays["oracle_pre"], _probabilities(arrays["logits_pre"])).mean())
    final = float(_kl(arrays["oracle_final"], _probabilities(arrays["logits_final"])).mean())
    if not np.isclose(final, evidence["final_kl"], rtol=0, atol=1e-10) or not np.isclose(pre, evidence["pre_kl"], rtol=0, atol=1e-10):
        raise ValueError("raw prediction metric differs from recorded value")
    return {"root": root, "pre_kl": pre, "final_kl": final}


def _guard_control(row: dict[str, Any], arm: str, cfg: dict[str, Any]) -> None:
    resources = row["resources"]
    graph = row["graph"]
    expected_effective = 4484 if arm == "SC64" else 16004
    expected_degree = 16 if arm == "SC64" else 64
    expected_input = 768 if arm == "SC64" else 3072
    expected_recurrent = 3072 if arm == "SC64" else 12288
    if (resources["allocated_parameters"] != 16004 or
        resources["effective_parameters"] != expected_effective or
        resources["active_input_parameters"] != expected_input or
        resources["active_recurrent_parameters"] != expected_recurrent or
        resources["hidden_state_floats"] != 64 or
        resources["pre_readout_cache_floats_per_example"] != 4 or
        resources["internal_steps_per_observation"] != 1 or
        resources["operator_shapes"]["input_linear_per_token"]["weight"] != [192, 16] or
        resources["operator_shapes"]["recurrent_linear_per_token"]["weight"] != [192, 64] or
        resources["operator_shapes"]["query_readout"]["weight"] != [4, 64] or
        graph["diagonal_connections"] != 64 or
        graph["strongly_connected"] is not True or
        graph["seed"] != cfg["model"]["graph_seed_base"] + row["seed"] or
        graph["topology"] != ("random" if arm == "SC64" else "dense") or
        len(graph["mask_sha256"]) != 64 or
        any(d != expected_degree for d in graph["in_degree"] + graph["out_degree"]) or
        graph["successful_swaps"] != (48000 if arm == "SC64" else 0) or
        row["train_examples"] != row["updates"] * cfg["training"]["batch_size"] or
        row["train_observation_steps"] != row["train_examples"] * cfg["task"]["train_length"] or
        row["dense_forward_linear_flops"] != resources["dense_forward_linear_flops_per_batch"] * row["updates"] or
        len(row["initial_parameters_sha256"]) != 64 or
        len(row["final_parameters_sha256"]) != 64):
        raise ControlWeakenedError(f"{arm} topology/resource/update guard failed")


@lru_cache(maxsize=16)
def _registered_sc_mask(graph_seed: int) -> np.ndarray:
    mask, info = recurrent_mask(64, 4, "random", graph_seed)
    if info["successful_swaps"] != 48000 or not info["strongly_connected"]:
        raise ControlWeakenedError("registered SC graph generation failed")
    return mask.numpy()


def _guard_checkpoint_state(path: Path, row: dict[str, Any], arm: str) -> None:
    """Inspect the saved final tensors, not only graph/resource summaries."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["arm"] != arm or checkpoint["seed"] != row["seed"] or
        checkpoint["lr"] != row["lr"] or checkpoint["step"] != row["updates"]):
        raise ControlWeakenedError("final checkpoint identity differs")
    state = checkpoint["state_dict"]
    mask = state["mask"].numpy()
    input_mask = state["input_mask"].numpy()
    if mask.shape != (192, 64) or input_mask.shape != (192, 16):
        raise ControlWeakenedError("saved masked operator shapes differ")
    base = mask[:64]
    inputs = input_mask[:64]
    degree = 16 if arm == "SC64" else 64
    if (not np.array_equal(mask, np.tile(base, (3, 1))) or
        not np.array_equal(input_mask, np.tile(inputs, (3, 1))) or
        not np.all(np.diag(base) == 1) or
        not np.all(base.sum(axis=0) == degree) or
        not np.all(base.sum(axis=1) == degree) or
        sha256(base.astype(np.uint8).tobytes(order="C")).hexdigest() != row["graph"]["mask_sha256"]):
        raise ControlWeakenedError("saved recurrent graph differs")
    for adjacency in (base, base.T):
        seen = {0}
        frontier = [0]
        while frontier:
            current = frontier.pop()
            for neighbor in np.flatnonzero(adjacency[:, current]):
                node = int(neighbor)
                if node not in seen:
                    seen.add(node)
                    frontier.append(node)
        if len(seen) != 64:
            raise ControlWeakenedError("saved recurrent graph is disconnected")
    if arm == "SC64":
        if not np.array_equal(base, _registered_sc_mask(row["graph"]["seed"])):
            raise ControlWeakenedError("saved SC graph differs from registered graph seed")
        expected = np.zeros((64, 16), dtype=np.float32)
        for group in range(4):
            expected[group*16:(group+1)*16, group*4:(group+1)*4] = 1
        if not np.array_equal(inputs, expected):
            raise ControlWeakenedError("SC64 saved routed input mask differs")
    elif not np.all(inputs == 1) or not np.all(base == 1):
        raise ControlWeakenedError("D64_full saved dense masks differ")
    parameter_hasher = sha256()
    for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh",
                 "readout_weight", "readout_bias"):
        parameter_hasher.update(name.encode())
        parameter_hasher.update(state[name].contiguous().numpy().tobytes())
    if parameter_hasher.hexdigest() != row["final_parameters_sha256"]:
        raise ControlWeakenedError("final parameter tensor hash differs")


def recompute_controls(run: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    cfg = manifest["config"]
    qcfg, tcfg = cfg["qualification"], cfg["training"]
    records = manifest["arms"]
    metrics: dict[str, Any] = {}
    updates = 0
    roots: set[int] = set()
    learning_curves: dict[str, Any] = {}
    regenerated: dict[tuple[str, int], dict[str, Any]] = {}
    paired_training_boundaries: dict[tuple[int, int], dict[str, Any]] = {}
    matrix = Path(manifest["source_repo"]) / cfg["matrix_file"]
    def evaluation_batch(split: str, seed: int) -> dict[str, Any]:
        key = (split, seed)
        if key not in regenerated:
            length = cfg["task"]["train_length"] if split.endswith("iid") else cfg["task"]["ood_length"]
            batch = make_batch(qcfg["eval_examples"], length,
                qcfg[split + "_root_base"] + seed, matrix,
                cue_reliability=cfg["task"]["cue_reliability"],
                switch_step=cfg["task"]["switch_step"])
            regenerated[key] = {"oracle_pre": batch.oracle_pre,
                                "oracle_final": batch.oracle_final,
                                "input_sha256": _array_hash(batch.x)}
        return regenerated[key]
    for arm in ARMS:
        metrics[arm] = {}
        learning_curves[arm] = {}
        for lr in qcfg["lr_menu"]:
            metrics[arm][str(lr)] = {}
            learning_curves[arm][str(lr)] = {}
            for seed in qcfg["model_seeds"]:
                row = records[arm][str(lr)][str(seed)]
                if row["arm"] != arm or row["seed"] != seed or row["lr"] != lr or row["status"] != "COMPLETED" or row["updates"] != tcfg["updates_per_arm_seed"]:
                    raise ValueError("incomplete or mislabeled control training")
                _guard_control(row, arm, cfg)
                for step in (1, tcfg["updates_per_arm_seed"]):
                    boundary = row["boundary_batch_hashes"][str(step)]
                    expected_root = tcfg["train_root_base"] + seed * 10_000 + step
                    if boundary["root"] != expected_root or any(
                        len(boundary[name]) != 64 for name in
                        ("x_sha256", "oracle_pre_sha256", "oracle_final_sha256")
                    ):
                        raise ControlWeakenedError("training boundary root/hash differs")
                    identity = (seed, step)
                    if identity in paired_training_boundaries and paired_training_boundaries[identity] != boundary:
                        raise ControlWeakenedError("paired controls received different boundary batches")
                    paired_training_boundaries[identity] = boundary
                updates += row["updates"]
                arm_dir = run / arm / str(lr) / str(seed)
                monitor_values = []
                for checkpoint in row["checkpoints"]:
                    if not np.isfinite(checkpoint["train_loss"]):
                        raise ValueError("non-finite checkpoint training loss")
                    if (checkpoint["checkpoint_file"] != f"checkpoint-{checkpoint['step']}.pt" or
                        checkpoint["prediction_file"] != f"monitor-{checkpoint['step']}.npz"):
                        raise ValueError("unexpected monitor artifact name")
                    expected_monitor_root = tcfg["monitor_root_base"] + seed * 10 + tcfg["monitor_steps"].index(checkpoint["step"])
                    if checkpoint["root"] != expected_monitor_root or digest(_safe_artifact(run, arm_dir / checkpoint["checkpoint_file"])) != checkpoint["checkpoint_sha256"]:
                        raise ValueError("checkpoint hash/step differs")
                    if digest(_safe_artifact(run, arm_dir / checkpoint["prediction_file"])) != checkpoint["prediction_sha256"]:
                        raise ValueError("monitor prediction hash differs")
                    monitor = _read_npz(_safe_artifact(run, arm_dir / checkpoint["prediction_file"]),
                                        checkpoint["prediction_sha256"],
                                        ("oracle_final", "logits_final"))
                    monitor_kl = float(_kl(monitor["oracle_final"],
                                           _probabilities(monitor["logits_final"])).mean())
                    if not np.isclose(monitor_kl, checkpoint["iid_final_kl"], rtol=0, atol=1e-10):
                        raise ValueError("monitor raw KL differs")
                    monitor_values.append(monitor_kl)
                if [cp["step"] for cp in row["checkpoints"]] != tcfg["monitor_steps"]:
                    raise ValueError("missing monitor checkpoint")
                _guard_checkpoint_state(_safe_artifact(run, arm_dir / row["checkpoints"][-1]["checkpoint_file"]), row, arm)
                learning_curves[arm][str(lr)][str(seed)] = {
                    "monitor_iid_kl": monitor_values,
                    "late_iid_kl_improved": bool(monitor_values[-1] < monitor_values[-2])}
                metrics[arm][str(lr)][str(seed)] = {}
                for split in ("tune_iid", "tune_ood"):
                    metric = _read_evaluation(run, arm, lr, seed, split, row, cfg,
                                              evaluation_batch(split, seed))
                    metrics[arm][str(lr)][str(seed)][split] = metric
                    roots.add(metric["root"])
    if updates != cfg["resource"]["research_update_cap"] or updates != manifest["research_updates"]:
        raise ValueError("research update cap/count differs")
    selected: dict[str, float] = {}
    for arm in ARMS:
        candidates = []
        for lr in qcfg["lr_menu"]:
            trials = [metrics[arm][str(lr)][str(seed)] for seed in qcfg["model_seeds"]]
            iid = np.array([trial["tune_iid"]["final_kl"] for trial in trials])
            ood = np.array([trial["tune_ood"]["final_kl"] for trial in trials])
            if np.all(iid < qcfg["iid_kl_per_seed_max_exclusive"]) and iid.mean() < qcfg["iid_kl_mean_max_exclusive"]:
                candidates.append((float(ood.mean()), lr))
        if not candidates:
            return {"status": "LEARNING_FAILURE", "metrics": metrics,
                    "learning_curves": learning_curves, "research_updates": updates}
        selected[arm] = min(candidates)[1]
    if selected != manifest.get("selected_lr"):
        raise ValueError("recorded LR selection differs from raw tuning arrays")
    iid_adequacy = True
    headroom = True
    for arm, lr in selected.items():
        trials = []
        for seed in qcfg["model_seeds"]:
            row = records[arm][str(lr)][str(seed)]
            for split in ("qual_iid", "qual_ood"):
                metric = _read_evaluation(run, arm, lr, seed, split, row, cfg,
                                          evaluation_batch(split, seed))
                metrics[arm][str(lr)][str(seed)][split] = metric
                roots.add(metric["root"])
            trials.append(metrics[arm][str(lr)][str(seed)])
        iid = np.array([trial["qual_iid"]["final_kl"] for trial in trials])
        ood = np.array([trial["qual_ood"]["final_kl"] for trial in trials])
        iid_adequacy &= bool(np.all(iid < qcfg["iid_kl_per_seed_max_exclusive"]) and iid.mean() < qcfg["iid_kl_mean_max_exclusive"])
        low, high = qcfg["ood_mean_windows"][arm]
        seed_low, seed_high = qcfg["ood_per_seed_windows"][arm]
        headroom &= bool(low <= ood.mean() <= high and np.all((ood >= seed_low) & (ood <= seed_high)))
    if manifest["task_information"]["root"] in roots or manifest["task_information"]["retained_root"] in roots:
        raise ValueError("task-info and evaluation roots overlap")
    replay = manifest.get("zero_update_pair_replay")
    if not replay or replay.get("optimization_updates") != 0 or replay.get("seed") != qcfg["model_seeds"][0]:
        raise ValueError("missing matched zero-update replay")
    for arm, lr in selected.items():
        if replay["arms"][arm]["lr"] != lr:
            raise ValueError("replay selected LR differs")
        row = records[arm][str(lr)][str(replay["seed"])]
        path = run / arm / str(lr) / str(replay["seed"]) / row["evaluations"]["qual_ood"]["predictions_file"]
        with np.load(_safe_artifact(run, path), allow_pickle=False) as data:
            for name in ("logits_pre", "logits_final"):
                if _array_hash(data[name]) != replay["arms"][arm][name + "_sha256"]:
                    raise ValueError("zero-update replay logit hash differs")
    return {"status": "NUMERIC_GUARDS_PASS_PENDING_REVIEW" if iid_adequacy and headroom else
            "CONTROL_UNRESOLVED" if not iid_adequacy else "CONTROL_FLOOR_OR_CEILING",
            "metrics": metrics, "learning_curves": learning_curves,
            "paired_boundary_batch_hashes": {f"{seed}:{step}": value for (seed, step), value in paired_training_boundaries.items()},
            "research_updates": updates, "selected_lr": selected,
            "iid_adequacy": bool(iid_adequacy), "ood_headroom": bool(headroom)}


def analyze(run: Path) -> dict[str, Any]:
    run = run.resolve()
    manifest_path = run / "manifest.json"
    failure_path = run / "worker_failure.json"
    failure = json.loads(failure_path.read_text()) if failure_path.is_file() else None
    failure_status = (failure.get("substatus") if failure else None)
    if failure_status not in (None, "INCOMPLETE_BUDGET", "ENGINEERING_ABORT"):
        raise ValueError("unrecognized worker failure substatus")
    if not manifest_path.is_file() and failure_path.is_file():
        report = {"experiment_id": "EXP-0013", "version": "v3",
                  "suggested_decision": "INCONCLUSIVE", "gate1_credit": False,
                  "review_state": "NOT_REVIEWED", "qualification_status": failure_status,
                  "worker_failure_sha256": digest(failure_path)}
        write_json(run / "analysis.json", report)
        return report
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment_id") != "EXP-0013" or manifest.get("version") != "v3" or manifest.get("gate1_claim") is not False:
        raise ValueError("not EXP-0013 v3 qualification evidence")
    report: dict[str, Any] = {"experiment_id": "EXP-0013", "version": "v3",
        "suggested_decision": "INCONCLUSIVE", "gate1_credit": False,
        "review_state": "NOT_REVIEWED", "source_manifest_sha256": digest(manifest_path)}
    if failure_path.is_file():
        report["worker_failure_sha256"] = digest(failure_path)
        report["qualification_status"] = failure_status
        write_json(run / "analysis.json", report)
        return report
    try:
        _forbid_unregistered_work(run, manifest, task_failed=False)
        report["engineering_preflight"] = recompute_preflight(manifest)
        if "task_information" in manifest:
            task = recompute_task_info(run, manifest)
            report["task_information"] = task
            if not task["pass"]:
                if manifest["status"] != "TASK_UNINFORMATIVE":
                    raise ProtocolAbortError("task-information raw failure conflicts with worker status")
                _forbid_unregistered_work(run, manifest, task_failed=True)
                report["qualification_status"] = "TASK_UNINFORMATIVE"
            elif manifest["status"] == "TASK_UNINFORMATIVE":
                raise ProtocolAbortError("task-information raw pass conflicts with worker stop")
            elif manifest["status"] in ("COMPLETED_PENDING_ANALYSIS", "LEARNING_FAILURE"):
                controls = recompute_controls(run, manifest)
                report["controls"] = controls
                report["qualification_status"] = controls["status"]
            else:
                report["qualification_status"] = manifest["status"]
        else:
            report["qualification_status"] = manifest["status"]
    except (OSError, KeyError, ValueError, FloatingPointError) as exc:
        report["qualification_status"] = ("ENGINEERING_ABORT" if isinstance(exc, ProtocolAbortError)
                                          else "CONTROL_WEAKENED" if isinstance(exc, ControlWeakenedError)
                                          else "CONTROL_UNRESOLVED")
        report["evidence_error"] = str(exc)
    deadline = manifest.get("worker_deadline_epoch")
    if deadline is not None and time.time() >= deadline:
        report["qualification_status"] = "INCOMPLETE_BUDGET"
        report["deadline_exceeded_in_analysis"] = True
    write_json(run / "analysis.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run)))


if __name__ == "__main__":
    main()
