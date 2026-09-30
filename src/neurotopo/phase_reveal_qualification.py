"""EXP-0013 v3 control qualification. No candidate arm is constructed here.

The registered entry is deliberately separate from small engineering helpers. A
run writes immutable raw predictions and checkpoints before numerical analysis;
independent scientific review belongs to a later ResearchLoop phase.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .phase_reveal_models import build_model
from .phase_reveal_resources import resource_account
from .phase_reveal_task import (
    PhaseRevealBatch, cue_only_posterior, last_frame_posterior, make_batch,
    truncated_prehistory_posterior, validate_oracles,
)


ARMS = ("SC64", "D64_full")
LOCK_SHA256 = "c5ccad6ebeeeed8870998414326c0dfa476a59ebc427189f2f8bac6bdc24c10e"
LOCK_PATH = "research/registrations/G1-RS-0001_config_lock.json"


class CompletedOptimizerStepError(FloatingPointError):
    """An optimizer step was entered; charge its update even if validation fails."""


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def locked_config(repo: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Check byte-level pre-sample lock and all transitive frozen references."""
    repo = repo.resolve()
    config_path = config_path.resolve()
    if config_path != (repo / "configs/exp0013_v3.json").resolve():
        raise ValueError("EXP-0013 requires the registered v3 config path")
    lock_path = repo / LOCK_PATH
    if digest(lock_path) != LOCK_SHA256:
        raise ValueError("registered G1-RS config lock SHA mismatch")
    lock = json.loads(lock_path.read_text())
    config = json.loads(config_path.read_text())
    if lock.get("id") != "G1-RS-0001_CONFIG_LOCK_V1" or not lock.get("no_candidate_task_training_in_qualification"):
        raise ValueError("wrong qualification config lock")
    hashes = {LOCK_PATH: digest(lock_path)}
    for name, reference in lock["references"].items():
        relative = reference["path"]
        actual = digest(repo / relative)
        if actual != reference["sha256"]:
            raise ValueError(f"frozen {name} differs: {relative}")
        hashes[relative] = actual
    if config["experiment_id"] != "EXP-0013" or config["version"] != "v3" or config["workflow"] != "phase_reveal_qualification":
        raise ValueError("wrong qualification config")
    if tuple(config["arms"]) != ARMS or "RS4" in config["arms"]:
        raise ValueError("qualification arms must be SC64 and D64_full only")
    if config["resource"]["research_update_cap"] != (len(ARMS) *
            len(config["qualification"]["lr_menu"]) *
            len(config["qualification"]["model_seeds"]) *
            config["training"]["updates_per_arm_seed"]):
        raise ValueError("registered research update count differs")
    for key, expected in (("matrix_file", "matrix_sha256"), ("pair_manifest_file", "pair_manifest_sha256")):
        if hashes[config[key]] != config[expected]:
            raise ValueError(f"{key} is not locked")
    if config["registration_lock_file"] != LOCK_PATH:
        raise ValueError("wrong registration lock path")
    pair = json.loads((repo / config["pair_manifest_file"]).read_text())
    if pair.get("id") != "G1-RS-0001_PROTOCOL_PAIR_V2" or pair["qualification"]["experiment_id"] != "EXP-0013":
        raise ValueError("wrong frozen protocol pair")
    for name, reference in pair["references"].items():
        if digest(repo / reference["path"]) != reference["sha256"]:
            raise ValueError(f"protocol-pair {name} byte hash differs")
    receipt = pair["references"]["matrix_receipt"]
    if receipt["sha256"] != config["matrix_receipt_sha256"]:
        raise ValueError("matrix receipt differs from qualification config")
    entry = json.loads((repo / config["entry_contract"]).read_text())
    if (entry["experiment_id"] != "EXP-0013" or entry["gate1_claim"] is not False or
        entry["factor"]["kind"] != "h3_candidate" or
        entry["task_search"]["heldout_rule"].find(LOCK_SHA256) < 0):
        raise ValueError("qualification entry contract differs")
    hashes[config["entry_contract"]] = digest(repo / config["entry_contract"])
    return config, hashes


def configure_cpu() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _loss(logits_pre: Tensor, logits_final: Tensor, pre: Tensor, final: Tensor) -> Tensor:
    return 0.5 * -(pre * F.log_softmax(logits_pre, dim=-1)).sum(dim=-1).mean() + 0.5 * -(final * F.log_softmax(logits_final, dim=-1)).sum(dim=-1).mean()


def _optimizer(model: torch.nn.Module, lr: float, cfg: dict[str, Any]) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=lr, betas=tuple(cfg["betas"]),
                            eps=cfg["adam_eps"], weight_decay=cfg["weight_decay"])


def _update(model: torch.nn.Module, optimizer: torch.optim.Adam, x: Tensor,
            pre: Tensor, final: Tensor, grad_clip: float) -> float:
    optimizer.zero_grad(set_to_none=True)
    lp, lf = model(x, pre_index=8)
    loss = _loss(lp, lf, pre, final)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("non-finite or missing gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    try:
        optimizer.step()
    except Exception as exc:
        raise CompletedOptimizerStepError("optimizer step failed after entry") from exc
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise CompletedOptimizerStepError("non-finite model parameter after optimizer step")
    return float(loss.detach())


def engineering_preflight(cfg: dict[str, Any], deadline: float) -> dict[str, Any]:
    """Only synthetic tensors from engineering seed; never call make_batch."""
    started = time.monotonic()
    rcfg = cfg["resource"]
    rng = np.random.Generator(np.random.PCG64(rcfg["engineering_seed"]))
    x = torch.from_numpy(rng.standard_normal((8, 32, 16)).astype(np.float32))
    pre = torch.from_numpy(rng.dirichlet(np.ones(4), size=8).astype(np.float32))
    final = torch.from_numpy(rng.dirichlet(np.ones(4), size=8).astype(np.float32))
    lr = max(cfg["qualification"]["lr_menu"])
    rows: dict[str, Any] = {}
    updates = 0
    for arm in ARMS:
        model = build_model(arm, rcfg["engineering_seed"], cfg["model"]["graph_seed_base"] + rcfg["engineering_seed"])
        first = model(x, 8)
        repeat = model(x, 8)
        if any(a.shape != (8, 4) or not torch.equal(a, b) or not torch.isfinite(a).all()
               for a, b in zip(first, repeat, strict=True)):
            raise AssertionError(f"{arm} engineering shape/determinism failed")
        initial = float(_loss(*first, pre, final).detach())
        optimizer = _optimizer(model, lr, cfg["training"])
        step_times = []
        for _ in range(rcfg["tiny_overfit_updates_per_arm"]):
            if time.monotonic() > deadline:
                raise TimeoutError("engineering preflight time cap")
            step_start = time.monotonic()
            _update(model, optimizer, x, pre, final, cfg["training"]["grad_clip"])
            step_times.append(time.monotonic() - step_start)
            updates += 1
            if updates > rcfg["engineering_update_cap"]:
                raise RuntimeError("engineering update cap exceeded")
        with torch.no_grad():
            ending = float(_loss(*model(x, 8), pre, final))
        rows[arm] = {"initial_loss": initial, "final_loss": ending,
                     "ratio": ending / initial, "updates": rcfg["tiny_overfit_updates_per_arm"],
                     "step_seconds_p95": float(np.percentile(step_times, 95)),
                     "pass": ending <= initial * rcfg["tiny_overfit_final_initial_ratio_max"]}
    return {"arms": rows, "updates": updates, "seconds": time.monotonic() - started,
            "engineering_seed": rcfg["engineering_seed"], "learning_rate": lr,
            "random_tensor_shape": list(x.shape),
            "input_sha256": _array_hash(x.numpy()),
            "pre_target_sha256": _array_hash(pre.numpy()),
            "final_target_sha256": _array_hash(final.numpy()),
            "pass": all(row["pass"] for row in rows.values())}


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return sha256(np.asarray(contiguous.shape, dtype="<i8").tobytes() + contiguous.tobytes()).hexdigest()


def _source_array_evidence(batch: PhaseRevealBatch) -> dict[str, dict[str, Any]]:
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "sha256": _array_hash(value)}
            for name in ("x", "u", "y", "eps")
            for value in (getattr(batch, name),)}


def _check_features(batch: PhaseRevealBatch) -> None:
    x = batch.x
    steps = x.shape[1]
    if (not np.array_equal(x[:, :, 0::4], batch.u.astype(np.float32)) or
        not np.array_equal(x[:, :, 1::4], batch.y.astype(np.float32)) or
        not np.array_equal(x[:, :, 2::4], np.broadcast_to(
            (batch.cue[:, None, None] == np.arange(4)).astype(np.float32), batch.u.shape)) or
        not np.array_equal(x[:, :, 3::4], np.broadcast_to(
            (np.arange(steps) >= batch.switch_step)[None, :, None], batch.u.shape)) or
        not np.array_equal(batch.y[:, :batch.switch_step], 0.8 * batch.eps[:, :batch.switch_step])):
        raise AssertionError("phase-reveal features, switch, or preswitch targets differ")


def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 4:
        raise ValueError("posterior shapes must match [B,4]")
    if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(p <= 0) or np.any(q <= 0):
        raise FloatingPointError("invalid posterior probabilities")
    return np.sum(p * (np.log(p) - np.log(q)), axis=1)


def _wilson_lower(success: int, total: int) -> float:
    if total == 0:
        return 0.0
    z = 1.959963984540054
    p = success / total
    d = 1 + z * z / total
    return float((p + z * z / (2 * total) - z * np.sqrt(p * (1-p) / total + z*z/(4*total*total))) / d)


def _bootstrap_lower(values: np.ndarray, n: int, rng: np.random.Generator) -> float:
    # Chunking bounds memory even for 16,384 observations and 10,000 resamples.
    means = np.empty(n, dtype=np.float64)
    for start in range(0, n, 64):
        end = min(n, start + 64)
        indices = rng.integers(0, len(values), size=(end-start, len(values)))
        means[start:end] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025))


def task_information(cfg: dict[str, Any], matrix: Path, output: Path,
                     deadline: float) -> dict[str, Any]:
    info, task = cfg["task_information"], cfg["task"]
    started = time.monotonic()
    batch = make_batch(info["examples"], task["ood_length"], info["root"], matrix,
                       cue_reliability=task["cue_reliability"], switch_step=task["switch_step"])
    _check_features(batch)
    if time.monotonic() > deadline:
        raise TimeoutError("qualification wall cap during task information")
    # Small independent enumeration/covariance subset, with hashes retained.
    subset = PhaseRevealBatch(*(getattr(batch, field)[:8] for field in
                                ("x", "oracle_pre", "oracle_final", "H", "cue", "u", "eps", "y")),
                              batch.switch_step, batch.cue_reliability)
    oracle_check = validate_oracles(subset, matrix)
    prior = cue_only_posterior(batch.cue, batch.cue_reliability)
    last = last_frame_posterior(batch.u, batch.y, batch.cue, matrix)
    trunc = truncated_prehistory_posterior(batch.u, batch.y, batch.cue, matrix)
    full = batch.oracle_final
    wrong = batch.cue != batch.H
    corrected = wrong & (np.argmax(batch.oracle_pre, axis=1) == batch.cue) & (np.argmax(full, axis=1) == batch.H)
    uniform_kl = _kl(full, np.full_like(full, 0.25))
    rng = np.random.Generator(np.random.PCG64(info["bootstrap_seed"]))
    shortcuts: dict[str, Any] = {}
    for name, posterior in (("cue", prior), ("last", last), ("truncated_prehistory", trunc)):
        values = _kl(full, posterior)
        shortcuts[name] = {"mean_kl": float(values.mean()),
                           "bootstrap_ci95_lower": _bootstrap_lower(values, info["bootstrap_resamples"], rng),
                           "posterior_sha256": _array_hash(posterior)}
    retained = make_batch(info["retained_examples"], task["ood_length"], info["retained_root"], matrix,
                          cue_reliability=task["retained_cue_reliability"],
                          switch_step=task["retained_switch_step"])
    _check_features(retained)
    retained_prior = cue_only_posterior(retained.cue, retained.cue_reliability)
    retained_last = last_frame_posterior(retained.u, retained.y, retained.cue, matrix,
                                         switch_step=retained.switch_step,
                                         cue_reliability=retained.cue_reliability)
    retained_trunc = truncated_prehistory_posterior(retained.u, retained.y, retained.cue, matrix,
                                                    switch_step=retained.switch_step,
                                                    cue_reliability=retained.cue_reliability)
    retained_corrected = ((retained.cue != retained.H) &
                          (np.argmax(retained.oracle_pre, axis=1) == retained.cue) &
                          (np.argmax(retained.oracle_final, axis=1) == retained.H))
    arrays = output / "task_information.npz"
    np.savez_compressed(arrays, H=batch.H, cue=batch.cue, p_pre=batch.oracle_pre,
                        p_full=full, p_cue=prior, p_last=last, p_truncated=trunc,
                        retained_H=retained.H, retained_cue=retained.cue,
                        retained_pre=retained.oracle_pre, retained_full=retained.oracle_final,
                        retained_cue_posterior=retained_prior,
                        retained_last=retained_last, retained_truncated=retained_trunc)
    report = {"root": info["root"], "retained_root": info["retained_root"],
              "arrays_file": arrays.name, "arrays_sha256": digest(arrays),
              "source_arrays": {"main": _source_array_evidence(batch),
                                "retained": _source_array_evidence(retained)},
              "oracle_check": oracle_check, "h_fractions": np.bincount(batch.H, minlength=4).astype(float).tolist(),
              "cue_hit": float((batch.cue == batch.H).mean()),
              "pre_prior_max_error": float(np.max(np.abs(batch.oracle_pre-prior))),
              "mean_full_uniform_kl": float(uniform_kl.mean()),
              "wrong_cue_count": int(wrong.sum()), "correction_count": int(corrected.sum()),
              "correction_wilson_lower": _wilson_lower(int(corrected.sum()), int(wrong.sum())),
              "shortcuts": shortcuts, "retained_correction_count": int(retained_corrected.sum()),
              "retained_mean_full_cue_kl": float(_kl(retained.oracle_final, retained_prior).mean()),
              "seconds": time.monotonic() - started}
    report["h_fractions"] = (np.asarray(report["h_fractions"]) / len(batch.H)).tolist()
    report["pass"] = bool(
        np.isfinite(full).all() and np.allclose(full.sum(axis=1), 1, rtol=0, atol=1e-10) and
        all(info["h_fraction_window"][0] <= value <= info["h_fraction_window"][1] for value in report["h_fractions"]) and
        info["cue_hit_window"][0] <= report["cue_hit"] <= info["cue_hit_window"][1] and
        report["pre_prior_max_error"] <= 1e-10 and
        report["mean_full_uniform_kl"] >= info["oracle_uniform_kl_min"] and
        report["correction_count"] >= info["wrong_cue_correction_count_min"] and
        report["correction_wilson_lower"] >= info["wrong_cue_correction_wilson_lower_min"] and
        all(shortcuts[name]["bootstrap_ci95_lower"] >= threshold for name, threshold in info["shortcut_kl_ci_lower_min"].items()) and
        report["retained_correction_count"] >= info["retained_correction_count_min"] and
        report["retained_mean_full_cue_kl"] >= info["retained_cue_kl_min"])
    return report


def evaluate(model: torch.nn.Module, batch: PhaseRevealBatch,
             chunk_size: int = 512) -> dict[str, np.ndarray | float]:
    """Return raw oracle/prediction arrays; KL is recomputed independently."""
    pre, final = [], []
    with torch.no_grad():
        for start in range(0, len(batch.x), chunk_size):
            x = torch.from_numpy(batch.x[start:start+chunk_size])
            lp, lf = model(x, pre_index=8)
            pre.append(lp.numpy().astype(np.float64))
            final.append(lf.numpy().astype(np.float64))
    logits_pre = np.concatenate(pre)
    logits_final = np.concatenate(final)
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
        return weights / weights.sum(axis=1, keepdims=True)
    return {"oracle_pre": batch.oracle_pre, "oracle_final": batch.oracle_final,
            "logits_pre": logits_pre, "logits_final": logits_final,
            "pre_kl": float(_kl(batch.oracle_pre, softmax(logits_pre)).mean()),
            "final_kl": float(_kl(batch.oracle_final, softmax(logits_final)).mean())}


def _save_predictions(path: Path, result: dict[str, np.ndarray | float]) -> str:
    np.savez_compressed(path, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    return digest(path)


def _state_hash(model: torch.nn.Module) -> str:
    hasher = sha256()
    for name, tensor in model.state_dict().items():
        hasher.update(name.encode())
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def _parameter_hash(model: torch.nn.Module) -> str:
    hasher = sha256()
    for name, tensor in model.named_parameters():
        hasher.update(name.encode())
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def train_arm(cfg: dict[str, Any], arm: str, seed: int, lr: float,
              matrix: Path, output: Path, deadline: float) -> dict[str, Any]:
    """Exactly the frozen final-weight schedule for one control/seed/LR."""
    if arm not in ARMS or seed not in cfg["qualification"]["model_seeds"] or lr not in cfg["qualification"]["lr_menu"]:
        raise ValueError("unregistered qualification arm, seed, or learning rate")
    tcfg, task = cfg["training"], cfg["task"]
    model = build_model(arm, seed, cfg["model"]["graph_seed_base"] + seed)
    account = resource_account(model, task["train_length"], tcfg["batch_size"])
    initial_hash = _state_hash(model)
    initial_parameters = _parameter_hash(model)
    optimizer = _optimizer(model, lr, tcfg)
    output.mkdir(parents=True, exist_ok=False)
    row: dict[str, Any] = {"arm": arm, "seed": seed, "lr": lr, "updates": 0,
                           "initial_state_sha256": initial_hash,
                           "initial_parameters_sha256": initial_parameters,
                           "graph": model.graph_info,
                           "resources": account, "checkpoints": [],
                           "boundary_batch_hashes": {}, "status": "INCOMPLETE"}
    generation_times: list[float] = []
    step_times: list[float] = []
    try:
        for step in range(1, tcfg["updates_per_arm_seed"] + 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("qualification worker wall cap")
            gen_start = time.monotonic()
            root = tcfg["train_root_base"] + seed * 10_000 + step
            batch = make_batch(tcfg["batch_size"], task["train_length"], root, matrix,
                               cue_reliability=task["cue_reliability"], switch_step=task["switch_step"])
            if step in (1, tcfg["updates_per_arm_seed"]):
                row["boundary_batch_hashes"][str(step)] = {
                    "root": root, "x_sha256": _array_hash(batch.x),
                    "oracle_pre_sha256": _array_hash(batch.oracle_pre),
                    "oracle_final_sha256": _array_hash(batch.oracle_final)}
            generation_times.append(time.monotonic() - gen_start)
            if time.monotonic() >= deadline:
                raise TimeoutError("qualification worker wall cap")
            step_start = time.monotonic()
            loss = _update(model, optimizer, torch.from_numpy(batch.x),
                           torch.from_numpy(batch.oracle_pre.astype(np.float32)),
                           torch.from_numpy(batch.oracle_final.astype(np.float32)), tcfg["grad_clip"])
            step_times.append(time.monotonic() - step_start)
            row["updates"] = step
            row["last_loss"] = loss
            if step in tcfg["monitor_steps"]:
                index = tcfg["monitor_steps"].index(step)
                monitor_root = tcfg["monitor_root_base"] + seed * 10 + index
                monitor = make_batch(512, task["train_length"], monitor_root, matrix)
                result = evaluate(model, monitor)
                checkpoint = output / f"checkpoint-{step}.pt"
                torch.save({"state_dict": model.state_dict(), "arm": arm, "seed": seed,
                            "lr": lr, "step": step}, checkpoint)
                predictions = output / f"monitor-{step}.npz"
                row["checkpoints"].append({"step": step, "root": monitor_root,
                                           "train_loss": loss,
                                           "iid_final_kl": result["final_kl"],
                                           "checkpoint_file": checkpoint.name,
                                           "checkpoint_sha256": digest(checkpoint),
                                           "prediction_file": predictions.name,
                                           "prediction_sha256": _save_predictions(predictions, result)})
            if time.monotonic() >= deadline:
                raise TimeoutError("qualification worker wall cap")
        row["status"] = "COMPLETED"
        row["final_state_sha256"] = _state_hash(model)
        row["final_parameters_sha256"] = _parameter_hash(model)
    except (TimeoutError, FloatingPointError) as exc:
        if isinstance(exc, CompletedOptimizerStepError):
            row["updates"] = max(row["updates"], step)
        row["status"] = "INCOMPLETE_BUDGET" if isinstance(exc, TimeoutError) else "CONTROL_UNRESOLVED"
        row["error"] = str(exc)
        checkpoint = output / "interrupted.pt"
        torch.save({"state_dict": model.state_dict(), "arm": arm, "seed": seed,
                    "lr": lr, "step": row["updates"]}, checkpoint)
        row["interrupted_checkpoint"] = checkpoint.name
        row["interrupted_checkpoint_sha256"] = digest(checkpoint)
    row["train_generation_seconds_p95"] = float(np.percentile(generation_times, 95)) if generation_times else None
    row["train_step_seconds_p95"] = float(np.percentile(step_times, 95)) if step_times else None
    row["train_examples"] = row["updates"] * tcfg["batch_size"]
    row["train_observation_steps"] = row["train_examples"] * task["train_length"]
    row["dense_forward_linear_flops"] = account["dense_forward_linear_flops_per_batch"] * row["updates"]
    write_json(output / "result.json", row)
    return row


def _evaluate_saved(cfg: dict[str, Any], arm: str, seed: int, lr: float,
                    split: str, matrix: Path, arm_dir: Path) -> dict[str, Any]:
    qcfg, task = cfg["qualification"], cfg["task"]
    iid = split in ("tune_iid", "qual_iid")
    root = qcfg[split + "_root_base"] + seed
    length = task["train_length"] if iid else task["ood_length"]
    batch = make_batch(qcfg["eval_examples"], length, root, matrix)
    _check_features(batch)
    model = build_model(arm, seed, cfg["model"]["graph_seed_base"] + seed)
    checkpoint = torch.load(arm_dir / "checkpoint-2400.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    result = evaluate(model, batch)
    path = arm_dir / f"{split}.npz"
    return {"root": root, "examples": len(batch.x), "final_kl": result["final_kl"],
            "pre_kl": result["pre_kl"], "predictions_file": path.name,
            "input_sha256": _array_hash(batch.x),
            "predictions_sha256": _save_predictions(path, result)}


def replay_pair(cfg: dict[str, Any], selected: dict[str, float], matrix: Path,
                output: Path, deadline: float) -> dict[str, Any]:
    """Zero-update reload/replay of the first matched control pair."""
    seed = cfg["qualification"]["model_seeds"][0]
    root = cfg["qualification"]["qual_ood_root_base"] + seed
    batch = make_batch(cfg["qualification"]["eval_examples"], cfg["task"]["ood_length"],
                       root, matrix)
    replay: dict[str, Any] = {"seed": seed, "root": root, "input_sha256": _array_hash(batch.x),
                              "arms": {}, "optimization_updates": 0}
    for arm, lr in selected.items():
        if time.monotonic() >= deadline:
            raise TimeoutError("qualification worker wall cap during replay")
        model = build_model(arm, seed, cfg["model"]["graph_seed_base"] + seed)
        path = output / arm / str(lr) / str(seed)
        checkpoint = torch.load(path / "checkpoint-2400.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        result = evaluate(model, batch)
        with np.load(path / "qual_ood.npz", allow_pickle=False) as saved:
            for key in ("oracle_pre", "oracle_final", "logits_pre", "logits_final"):
                if not np.array_equal(result[key], saved[key]):
                    raise AssertionError(f"zero-update replay differs: {arm}/{key}")
        replay["arms"][arm] = {"lr": lr, "weights_sha256": _state_hash(model),
                               "logits_pre_sha256": _array_hash(result["logits_pre"]),
                               "logits_final_sha256": _array_hash(result["logits_final"])}
    return replay


def _selection(rows: dict[str, Any], cfg: dict[str, Any]) -> dict[str, float] | None:
    qcfg = cfg["qualification"]
    selected: dict[str, float] = {}
    for arm in ARMS:
        candidates = []
        for lr in qcfg["lr_menu"]:
            trials = [rows[arm][str(lr)][str(seed)] for seed in qcfg["model_seeds"]]
            iid = np.array([r["evaluations"]["tune_iid"]["final_kl"] for r in trials])
            if np.any(iid >= qcfg["iid_kl_per_seed_max_exclusive"]) or iid.mean() >= qcfg["iid_kl_mean_max_exclusive"]:
                continue
            ood = np.array([r["evaluations"]["tune_ood"]["final_kl"] for r in trials])
            candidates.append((float(ood.mean()), lr))
        if not candidates:
            return None
        selected[arm] = min(candidates)[1]
    return selected


def _provenance(repo: Path, cfg: dict[str, Any], hashes: dict[str, str]) -> dict[str, Any]:
    """Immutable source identity shared by the preflight and registered run."""
    protected = ["configs/exp0013_v3.json", LOCK_PATH,
                 "research/registrations/G1-RS-0001_protocol_pair_v2.json",
                 "experiments/EXP-0013/v3.md", "experiments/EXP-0014/v3.md",
                 "configs/exp0013_matrices.json", "configs/exp0013_matrices_receipt.json",
                 "research/entries/EXP-0013.json", "src/neurotopo/phase_reveal_qualification.py",
                 "src/neurotopo/phase_reveal_qualification_analysis.py",
                 "src/neurotopo/phase_reveal_task.py", "src/neurotopo/phase_reveal_models.py",
                 "src/neurotopo/phase_reveal_resources.py", "src/neurotopo/models.py",
                 "pyproject.toml", "uv.lock"]
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *protected],
                                    cwd=repo, text=True)
    if dirty:
        raise ValueError("registered qualification files must be committed and clean")
    return {"frozen_hashes": hashes, "lock_sha256": LOCK_SHA256,
        "source_hashes": {f"src/neurotopo/{name}.py": digest(repo / "src/neurotopo" / f"{name}.py")
                          for name in ("models", "phase_reveal_task", "phase_reveal_models",
                                       "phase_reveal_resources", "phase_reveal_qualification",
                                       "phase_reveal_qualification_analysis")},
        "dependency_hashes": {name: digest(repo / name) for name in ("pyproject.toml", "uv.lock")},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                  cwd=repo, text=True).strip()}


def _preflight_paths(repo: Path, cfg: dict[str, Any]) -> tuple[Path, Path]:
    relative = Path("artifacts/validation/exp0013_engineering_preflight.json")
    report = repo / relative
    return report, report.with_name(report.stem + "_receipt.json")


def _seal_path(receipt: Path) -> Path:
    return receipt.with_name(receipt.stem.removesuffix("_receipt") + "_seal.json")


def _committed_identical(repo: Path, path: Path) -> bool:
    """Require an ordinary file whose bytes equal the current HEAD Git blob."""
    repo = repo.resolve()
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(repo):
        return False
    relative = path.resolve().relative_to(repo).as_posix()
    result = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=repo,
                            check=False, capture_output=True)
    return result.returncode == 0 and result.stdout == path.read_bytes()


def _commit_ancestor(repo: Path, earlier: str, later: str) -> bool:
    result = subprocess.run(["git", "merge-base", "--is-ancestor", earlier, later],
                            cwd=repo, check=False, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise ValueError(f"cannot verify preflight source ancestry: {result.stderr.strip()}")
    return result.returncode == 0


def _engineering_oracle_check(cfg: dict[str, Any], matrix: Path,
                              deadline: float) -> list[dict[str, Any]]:
    """Two tiny T48 mathematical references before any registered task root."""
    task = cfg["task"]
    rows = []
    for switch, reliability in ((task["switch_step"], task["cue_reliability"]),
                                (task["retained_switch_step"], task["retained_cue_reliability"])):
        if time.monotonic() >= deadline:
            raise TimeoutError("engineering oracle validation time cap")
        batch = make_batch(2, task["ood_length"], 42, matrix,
                           cue_reliability=reliability, switch_step=switch)
        _check_features(batch)
        evidence = validate_oracles(batch, matrix, atol=1e-10)
        rows.append({"engineering_root": 42, "examples": 2, "length": task["ood_length"],
                     "switch_step": switch, "cue_reliability": reliability,
                     "validation": evidence})
    return rows


def run_engineering_preflight(config_path: Path, *, repo: Path) -> dict[str, Any]:
    """Separate ≤300-second, ≤200-update engineering gate on random tensors only."""
    cfg, hashes = locked_config(repo, config_path)
    provenance = _provenance(repo, cfg, hashes)
    report_path, receipt_path = _preflight_paths(repo, cfg)
    if report_path.exists() or receipt_path.exists() or _seal_path(receipt_path).exists():
        raise FileExistsError("engineering preflight report/receipt/seal already exists")
    configure_cpu()
    started = time.monotonic()
    report: dict[str, Any] = {"experiment_id": "EXP-0013", "version": "v3",
                              "kind": "ENGINEERING_ONLY", "status": "ENGINEERING_ABORT",
                              "config_sha256": digest(config_path), **provenance}
    try:
        result = engineering_preflight(cfg, started + cfg["resource"]["preflight_wall_seconds_cap"])
        report["engineering"] = result
        if result["updates"] > cfg["resource"]["engineering_update_cap"]:
            raise RuntimeError("engineering update cap exceeded")
        if result["pass"]:
            report["oracle_checks"] = _engineering_oracle_check(
                cfg, repo / cfg["matrix_file"],
                started + cfg["resource"]["preflight_wall_seconds_cap"])
        report["status"] = ("PASS" if result["pass"] and
                            time.monotonic() - started <= cfg["resource"]["preflight_wall_seconds_cap"]
                            else "ENGINEERING_ABORT")
    except (TimeoutError, FloatingPointError, AssertionError, RuntimeError) as exc:
        report["error"] = str(exc)
    report["seconds"] = time.monotonic() - started
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    seconds_after_report_fsync = time.monotonic() - started
    receipt_status = ("PASS" if report["status"] == "PASS" and
                      seconds_after_report_fsync <= cfg["resource"]["preflight_wall_seconds_cap"]
                      else "ENGINEERING_ABORT")
    receipt = {"experiment_id": "EXP-0013", "kind": "ENGINEERING_PREFLIGHT_RECEIPT",
               "report_file": str(report_path.relative_to(repo)),
               "report_sha256": digest(report_path), "status": receipt_status,
               "config_sha256": report["config_sha256"],
               "source_commit": provenance["source_commit"],
               "seconds_after_report_fsync": seconds_after_report_fsync,
               "report_write_seconds": seconds_after_report_fsync - report["seconds"],
               "write_procedure": {"report": "exclusive_write_flush_fsync",
                                   "receipt": "exclusive_write_flush_fsync"}}
    with receipt_path.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt_fsync_seconds = time.monotonic() - started
    seal_status = ("PASS" if receipt_status == "PASS" and
                   receipt_fsync_seconds <= cfg["resource"]["preflight_wall_seconds_cap"]
                   else "ENGINEERING_ABORT")
    seal = {"experiment_id": "EXP-0013", "kind": "ENGINEERING_PREFLIGHT_SEAL",
            "status": seal_status, "report_sha256": digest(report_path),
            "receipt_sha256": digest(receipt_path),
            "receipt_fsync_seconds": receipt_fsync_seconds,
            "receipt_write_seconds": receipt_fsync_seconds - seconds_after_report_fsync,
            "source_commit": provenance["source_commit"],
            "write_procedure": "exclusive_write_flush_fsync"}
    with _seal_path(receipt_path).open("x") as stream:
        stream.write(json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {**report, "status": seal_status,
            "seconds_after_report_fsync": seconds_after_report_fsync,
            "receipt_sha256": digest(receipt_path),
            "receipt_fsync_seconds": receipt_fsync_seconds,
            "seal_sha256": digest(_seal_path(receipt_path))}


def _verified_preflight(repo: Path, cfg: dict[str, Any], provenance: dict[str, Any],
                        config_path: Path) -> dict[str, Any]:
    report_path, receipt_path = _preflight_paths(repo, cfg)
    seal_path = _seal_path(receipt_path)
    report = json.loads(report_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    seal = json.loads(seal_path.read_text())
    if (receipt.get("report_sha256") != digest(report_path) or
        not all(_committed_identical(repo, path) for path in (report_path, receipt_path, seal_path)) or
        receipt.get("report_file") != str(report_path.relative_to(repo)) or
        receipt.get("source_commit") != report.get("source_commit") or
        report.get("source_commit") == provenance["source_commit"] or
        not _commit_ancestor(repo, report.get("source_commit", ""), provenance["source_commit"]) or
        seal.get("report_sha256") != digest(report_path) or
        seal.get("receipt_sha256") != digest(receipt_path) or
        seal.get("status") != "PASS" or
        seal.get("source_commit") != report.get("source_commit") or
        seal.get("receipt_fsync_seconds", float("inf")) >
            cfg["resource"]["preflight_wall_seconds_cap"] or
        seal.get("receipt_fsync_seconds", -1) < receipt.get("seconds_after_report_fsync", float("inf")) or
        seal.get("receipt_write_seconds", -1) < 0 or
        abs(seal["receipt_fsync_seconds"] - receipt["seconds_after_report_fsync"] -
            seal["receipt_write_seconds"]) > 1e-6 or
        seal.get("write_procedure") != "exclusive_write_flush_fsync" or
        receipt.get("config_sha256") != digest(config_path) or
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
        report.get("config_sha256") != digest(config_path) or
        any(report.get(key) != value for key, value in provenance.items()
            if key != "source_commit") or
        report.get("engineering", {}).get("updates") != cfg["resource"]["engineering_update_cap"] or
        report.get("engineering", {}).get("engineering_seed") != cfg["resource"]["engineering_seed"] or
        report.get("engineering", {}).get("learning_rate") != max(cfg["qualification"]["lr_menu"]) or
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
        any(report.get("engineering", {}).get("arms", {}).get(arm, {}).get("updates") !=
            cfg["resource"]["tiny_overfit_updates_per_arm"] or
            report["engineering"]["arms"][arm].get("ratio", float("inf")) >
            cfg["resource"]["tiny_overfit_final_initial_ratio_max"] for arm in ARMS) or
        report.get("seconds", float("inf")) > cfg["resource"]["preflight_wall_seconds_cap"] or
        not report.get("engineering", {}).get("pass")):
        raise ValueError("engineering preflight report/receipt provenance or PASS differs")
    return {"file": str(report_path.relative_to(repo)), "sha256": digest(report_path),
            "receipt_file": str(receipt_path.relative_to(repo)),
            "receipt_sha256": digest(receipt_path),
            "seal_file": str(seal_path.relative_to(repo)), "seal_sha256": digest(seal_path),
            "source_commit": report["source_commit"], "engineering": report["engineering"]}


def run(config_path: Path, *, repo: Path, output: Path | None = None) -> dict[str, Any]:
    """Registered research worker; preflight is already sealed, never rerun here."""
    cfg, hashes = locked_config(repo, config_path)
    provenance = _provenance(repo, cfg, hashes)
    preflight = _verified_preflight(repo, cfg, provenance, config_path)
    output = output or repo / cfg["output"]
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"qualification output already exists: {output}")
    configure_cpu()
    started = time.monotonic()
    manifest: dict[str, Any] = {"experiment_id": "EXP-0013", "version": "v3",
        "status": "TASK_INFORMATION", "config": cfg, **provenance,
        "source_repo": str(repo.resolve()),
        "engineering_preflight": preflight,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "platform": platform.platform(),
                    "machine": platform.machine(), "processor": platform.processor(),
                    "torch_build": torch.__config__.show(), "threads": torch.get_num_threads()},
        "arms": {}, "research_updates": 0, "rs4_task_updates": 0,
        "generation_p95_seconds": None, "step_p95_seconds": None, "gate1_claim": False}
    output.mkdir(parents=True)
    matrix = repo / cfg["matrix_file"]
    deadline = started + cfg["resource"]["worker_wall_seconds_cap"]
    manifest["worker_wall_seconds_cap"] = cfg["resource"]["worker_wall_seconds_cap"]
    manifest["worker_started_epoch"] = time.time()
    manifest["worker_deadline_epoch"] = manifest["worker_started_epoch"] + cfg["resource"]["worker_wall_seconds_cap"]
    write_json(output / "manifest.json", manifest)
    try:
        info = task_information(cfg, matrix, output, deadline)
        manifest["task_information"] = info
        if not info["pass"]:
            manifest["status"] = "TASK_UNINFORMATIVE"
            return manifest
        if time.monotonic() >= deadline:
            raise TimeoutError("qualification worker wall cap")
        manifest["status"] = "TRAINING"
        write_json(output / "manifest.json", manifest)
        rows = manifest["arms"]
        for arm in ARMS:
            rows[arm] = {}
            for lr in cfg["qualification"]["lr_menu"]:
                by_seed = rows[arm][str(lr)] = {}
                for seed in cfg["qualification"]["model_seeds"]:
                    if manifest["research_updates"] + cfg["training"]["updates_per_arm_seed"] > cfg["resource"]["research_update_cap"]:
                        raise RuntimeError("research update cap would be exceeded")
                    arm_dir = output / arm / str(lr) / str(seed)
                    row = train_arm(cfg, arm, seed, lr, matrix, arm_dir, deadline)
                    by_seed[str(seed)] = row
                    manifest["research_updates"] += row["updates"]
                    generation = [r["train_generation_seconds_p95"]
                                  for by_lr in rows.values() for by_seed2 in by_lr.values()
                                  for r in by_seed2.values()
                                  if r.get("train_generation_seconds_p95") is not None]
                    manifest["generation_p95_seconds"] = max(generation) if generation else None
                    steps = [r["train_step_seconds_p95"]
                             for by_lr in rows.values() for by_seed2 in by_lr.values()
                             for r in by_seed2.values()
                             if r.get("train_step_seconds_p95") is not None]
                    manifest["step_p95_seconds"] = max(steps) if steps else None
                    write_json(output / "manifest.json", manifest)
                    if row["status"] != "COMPLETED":
                        manifest["status"] = row["status"]
                        return manifest
                    row["evaluations"] = {}
                    for split in ("tune_iid", "tune_ood"):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("qualification worker wall cap")
                        row["evaluations"][split] = _evaluate_saved(cfg, arm, seed, lr, split, matrix, arm_dir)
                    write_json(arm_dir / "result.json", row)
                    write_json(output / "manifest.json", manifest)
        selected = _selection(rows, cfg)
        manifest["selected_lr"] = selected
        if selected is None:
            manifest["status"] = "LEARNING_FAILURE"
            return manifest
        for arm, lr in selected.items():
            for seed in cfg["qualification"]["model_seeds"]:
                row = rows[arm][str(lr)][str(seed)]
                arm_dir = output / arm / str(lr) / str(seed)
                for split in ("qual_iid", "qual_ood"):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("qualification worker wall cap")
                    row["evaluations"][split] = _evaluate_saved(cfg, arm, seed, lr, split, matrix, arm_dir)
                write_json(arm_dir / "result.json", row)
                write_json(output / "manifest.json", manifest)
        manifest["zero_update_pair_replay"] = replay_pair(cfg, selected, matrix, output, deadline)
        if time.monotonic() >= deadline:
            raise TimeoutError("qualification worker wall cap after replay")
        manifest["status"] = "COMPLETED_PENDING_ANALYSIS"
        return manifest
    except TimeoutError as exc:
        manifest["status"] = "INCOMPLETE_BUDGET"
        manifest["error"] = str(exc)
        return manifest
    except (FloatingPointError, AssertionError) as exc:
        manifest["status"] = "CONTROL_UNRESOLVED"
        manifest["error"] = str(exc)
        return manifest
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        manifest["process_peak_rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        manifest["process_peak_rss_unit"] = "bytes on macOS; KiB on Linux"
        write_json(output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp0013_v3.json"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engineering-preflight", action="store_true",
                        help="only random-tensor engineering gate; writes a sealed report and receipt")
    args = parser.parse_args()
    if args.engineering_preflight:
        if args.output is not None:
            parser.error("--output is only for registered research runs")
        result = run_engineering_preflight(args.config, repo=args.repo.resolve())
        print(json.dumps({"status": result["status"], "engineering_updates": result.get("engineering", {}).get("updates", 0)}))
        if result["status"] != "PASS":
            raise SystemExit(1)
    else:
        result = run(args.config, repo=args.repo.resolve(), output=args.output)
        print(json.dumps({"status": result["status"], "research_updates": result["research_updates"]}))


if __name__ == "__main__":
    main()
