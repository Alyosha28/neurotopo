"""Registered EXP-0011 staged field-reference study with a hard control gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .pfstyle_models import build_model
from .pfstyle_resources import resource_account
from .pfstyle_preflight import classify_resource
from .pfstyle_task import PfstyleBatch, make_batch
from .run import configure, digest, git_output, source_fingerprints, write_json


CONFIG_ID = ("EXP-0011", "v6", "pfstyle_field")
ARMS = ("K4_score", "S68_sparse", "S68_dense", "S64_score", "K4_unscored")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch_sha(batch: PfstyleBatch) -> str:
    return digest(b"".join((batch.x.numpy().tobytes(), batch.pre_target.numpy().tobytes(),
                            batch.final_target.numpy().tobytes(), batch.truth.numpy().tobytes(),
                            batch.mode_means.numpy().tobytes())))


def _loss(model: nn.Module, batch: PfstyleBatch) -> Tensor:
    pre, final = model(batch.x, batch.pre_index)
    targets = (batch.pre_target.to(torch.float32), batch.final_target.to(torch.float32))
    return 0.5 * sum((-(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)).mean()
                     for logits, target in zip((pre, final), targets))


def _kl_rows(target: Tensor, logits: Tensor) -> Tensor:
    truth = target.to(torch.float64)
    log_model = F.log_softmax(logits.to(torch.float64), dim=-1)
    log_truth = torch.where(truth > 0, truth.clamp_min(torch.finfo(torch.float64).tiny).log(), 0)
    return torch.where(truth > 0, truth * (log_truth - log_model), 0).sum(dim=-1)


@torch.no_grad()
def evaluate(model: nn.Module, batch: PfstyleBatch) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    pre_logits, final_logits = [], []
    for offset in range(0, len(batch.x), 128):
        pre, final = model(batch.x[offset:offset + 128], batch.pre_index)
        pre_logits.append(pre)
        final_logits.append(final)
    pre = torch.cat(pre_logits)
    final = torch.cat(final_logits)
    pre_kl = _kl_rows(batch.pre_target, pre)
    final_kl = _kl_rows(batch.final_target, final)
    revise = batch.pre_target.argmax(-1) != batch.final_target.argmax(-1)
    predicted = final.argmax(-1)
    probabilities = final.to(torch.float64).softmax(-1)
    entropy = -(probabilities * probabilities.clamp_min(torch.finfo(torch.float64).tiny).log()).sum(-1)
    result = {
        "examples": len(batch.x),
        "pre_joint_kl_to_oracle": pre_kl.mean().item(),
        "final_joint_kl_to_oracle": final_kl.mean().item(),
        "final_oracle_map_agreement": (predicted == batch.final_target.argmax(-1)).double().mean().item(),
        "final_true_map_accuracy": (predicted == batch.truth).double().mean().item(),
        "predicted_mode_map_fraction": [(predicted == mode).double().mean().item()
                                        for mode in range(4)],
        "predicted_entropy_final": entropy.mean().item(),
        "oracle_revision_count": int(revise.sum().item()),
        "oracle_revision_subset_final_kl": final_kl[revise].mean().item() if revise.any() else None,
        "dataset_sha256": _batch_sha(batch),
    }
    if getattr(model, "score_state_floats", 0):
        mu_error_sum = sigma_sum = sigma_error_sum = entropy_early_sum = entropy_final_sum = 0.0
        mu_error_by_mode = np.zeros(4, dtype=np.float64)
        sigma_by_mode = np.zeros(4, dtype=np.float64)
        cosine_sum = 0.0
        seen = 0
        for offset in range(0, len(batch.x), 128):
            stop = min(offset + 128, len(batch.x))
            trace = model.diagnostic_trace(batch.x[offset:stop],
                                           include_hidden=getattr(model, "name", "") == "K4_score")
            mu = trace["mu"].to(torch.float64)
            sigma = trace["sigma"].to(torch.float64)
            expected_mu = batch.mode_means[offset:stop]
            count = stop - offset
            mu_error_sum += (mu - expected_mu).square().mean(dim=(1, 2)).sum().item()
            sigma_sum += sigma.mean(dim=(1, 2)).sum().item()
            sigma_error_sum += (sigma - 0.8).abs().mean(dim=(1, 2)).sum().item()
            mu_error_by_mode += (mu - expected_mu).square().mean(dim=1).sum(dim=0).numpy()
            sigma_by_mode += sigma.mean(dim=1).sum(dim=0).numpy()
            for index, key in ((batch.pre_index - 1, "early"), (-1, "final")):
                logw = trace["logw"][:, index].to(torch.float64)
                p = logw.exp()
                value = -(p * logw).sum(-1).sum().item()
                if key == "early":
                    entropy_early_sum += value
                else:
                    entropy_final_sum += value
            if "hidden" in trace:
                chunks = trace["hidden"][:, -1].reshape(count, 4, 16).to(torch.float64)
                normalized = F.normalize(chunks, dim=-1)
                similarity = torch.einsum("nki,nji->nkj", normalized, normalized).abs()
                upper = torch.triu_indices(4, 4, offset=1)
                cosine_sum += similarity[:, upper[0], upper[1]].mean(-1).sum().item()
            seen += count
        result["score_diagnostics"] = {
            "gaussian_head_mu_mse_vs_known_mode": mu_error_sum / seen,
            "gaussian_head_mu_mse_by_mode": (mu_error_by_mode / seen).tolist(),
            "gaussian_head_sigma_mean": sigma_sum / seen,
            "gaussian_head_sigma_mean_by_mode": (sigma_by_mode / seen).tolist(),
            "gaussian_head_sigma_mae_vs_true_0_8": sigma_error_sum / seen,
            "score_entropy_t8": entropy_early_sum / seen,
            "score_entropy_final": entropy_final_sum / seen,
            "state_pair_abs_cosine_final": cosine_sum / seen if getattr(model, "name", "") == "K4_score" else None,
            "scope": "Descriptive fixed-mode head calibration and state similarity, not causal specialization",
        }
    arrays = {"pre_logits": pre.numpy(), "final_logits": final.numpy(),
              "pre_target": batch.pre_target.numpy(),
              "final_target": batch.final_target.numpy(), "truth": batch.truth.numpy()}
    result["prediction_sha256"] = digest(b"".join(arrays[key].tobytes() for key in sorted(arrays)))
    return result, arrays


def task_information(config: dict[str, Any]) -> dict[str, Any]:
    task = config["task"]
    batch = make_batch(task["task_information_examples"], task["ood_observations"],
                       task["task_information_root"], Path(config["matrix_file"]))
    final = batch.final_target.numpy()
    pre = batch.pre_target.numpy()
    truth = batch.truth.numpy()
    map_accuracy = float(np.mean(np.argmax(final, axis=1) == truth))
    revision_rate = float(np.mean(np.argmax(pre, axis=1) != np.argmax(final, axis=1)))
    uniform_kl = float(np.mean(np.sum(np.where(final > 0,
                                             final * (np.log(np.clip(final, np.finfo(np.float64).tiny, 1))
                                                      + np.log(task["modes"])), 0), axis=1)))
    passed = (map_accuracy >= task["oracle_true_map_accuracy_min"] and
              revision_rate >= task["oracle_revision_rate_min"] and
              uniform_kl >= task["uniform_to_oracle_kl_min"])
    prefix = make_batch(task["task_information_examples"], task["iid_observations"],
                        task["task_information_root"], Path(config["matrix_file"]))
    if (not torch.equal(batch.x[:, :task["iid_observations"]], prefix.x)
            or not torch.equal(batch.posterior_history[:, :task["iid_observations"]],
                               prefix.posterior_history)):
        raise RuntimeError("EXP-0011 fixed task prefix mismatch")
    return {"status": "PASS" if passed else "TASK_UNINFORMATIVE",
            "ood32_oracle_true_map_accuracy": map_accuracy,
            "ood32_oracle_revision_rate_t8_to_t32": revision_rate,
            "ood32_uniform_to_oracle_kl": uniform_kl,
            "ood32_dataset_sha256": _batch_sha(batch),
            "iid16_prefix_dataset_sha256": _batch_sha(prefix),
            "prefix_equal": True, "examples": len(batch.x)}


def _training_batches(config: dict[str, Any], seed: int) -> tuple[list[PfstyleBatch], str]:
    train = config["training"]
    batches = []
    hasher = hashlib.sha256()
    for step in range(1, train["updates_per_arm_seed"] + 1):
        root = train["train_root_base"] + seed * 10_000 + step
        batch = make_batch(train["batch_size"], config["task"]["iid_observations"], root,
                           Path(config["matrix_file"]))
        hasher.update(bytes.fromhex(_batch_sha(batch)))
        batches.append(batch)
    return batches, hasher.hexdigest()


def _seal_test_streams(output: Path, phase: str, seed: int,
                       tests: dict[str, PfstyleBatch], root: int) -> dict[str, Any]:
    """Publish fixed final-stream hashes before this seed's first model update."""
    destination = output / phase / f"seed-{seed}"
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"phase": phase, "seed": seed, "root": root,
                "streams": {label: {"examples": len(batch.x), "length": batch.x.shape[1],
                                    "dataset_sha256": _batch_sha(batch)}
                            for label, batch in tests.items()},
                "sealed_before_training": True}
    with (destination / "test_stream_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def _parameter_sha(model: nn.Module) -> str:
    return digest(b"".join(parameter.detach().cpu().numpy().tobytes()
                           for parameter in model.parameters()))


def _verify_preflight(config: dict[str, Any], config_path: Path,
                      preflight: dict[str, Any], preflight_path: Path,
                      source_hashes: dict[str, str]) -> dict[str, Any]:
    receipt_path = preflight_path.with_name(preflight_path.stem + "_receipt.json")
    receipt = json.loads(receipt_path.read_text())
    if (receipt["preflight_report_sha256"] != _file_sha(preflight_path)
            or receipt["budget_status"] not in {"PASS", "INCOMPLETE_BUDGET"}
            or receipt["seconds_after_report_fsync"] < 0
            or (receipt["budget_status"] == "PASS" and
                receipt["seconds_after_report_fsync"] > config["resource"]["preflight_wall_seconds_cap"])):
        raise RuntimeError("EXP-0011 preflight receipt or total preflight time invalid")
    commit = preflight["git_commit"]
    if (not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit)
            or subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                              check=False).returncode):
        raise RuntimeError("EXP-0011 preflight commit is not an ancestor of run source")
    if (preflight["source_sha256"] != source_hashes
            or preflight["config_sha256"] != _file_sha(config_path)
            or preflight["matrix_json_sha256"] != config["matrix_json_sha256"]
            or preflight["lock_sha256"] != _file_sha(Path("uv.lock"))
            or not preflight["engineering_valid"]
            or preflight["engineering_updates"] > config["resource"]["engineering_update_cap"]):
        raise RuntimeError("EXP-0011 preflight provenance or engineering budget mismatch")
    if preflight["resource_status"] == "INCOMPLETE_BUDGET":
        if receipt["budget_status"] != "INCOMPLETE_BUDGET":
            raise RuntimeError("incomplete preflight must have an over-budget receipt")
        return receipt
    accounts = preflight["resource_accounts"]
    timing = preflight["timing"]
    if set(accounts) != set(config["models"]) or set(timing) != set(config["models"]):
        raise RuntimeError("EXP-0011 preflight arm inventory incomplete")
    if set(preflight["operator_profiles"]) != set(config["models"]):
        raise RuntimeError("EXP-0011 five-arm operator profile missing")
    for profile in preflight["operator_profiles"].values():
        if any(not profile.get(key) for key in ("training16", "inference16", "inference32")):
            raise RuntimeError("EXP-0011 registered operator path absent")
    ratios = []
    for block in range(config["resource"]["timing_blocks"]):
        candidate, control = timing["K4_score"][block], timing["S68_sparse"][block]
        for path in ("training16_seconds", "inference16_seconds", "inference32_seconds"):
            values_a, values_b = candidate[path], control[path]
            if (len(values_a) != config["resource"]["timing_repetitions_per_block"]
                    or len(values_b) != config["resource"]["timing_repetitions_per_block"]
                    or not np.isfinite(values_a).all() or not np.isfinite(values_b).all()
                    or np.median(values_b) <= 0):
                raise RuntimeError("EXP-0011 preflight timing samples invalid")
            ratios.append(float(np.median(values_a) / np.median(values_b)))
    if not np.allclose(ratios, preflight["candidate_control_p50_ratios"], rtol=0, atol=1e-12):
        raise RuntimeError("EXP-0011 preflight reported p50 ratios differ from raw timings")
    statuses = {str(length): classify_resource(accounts["K4_score"][str(length)],
                                               accounts["S68_sparse"][str(length)], ratios, config)
                for length in (16, 32)}
    resource_status = ("RESOURCE_MISMATCH" if "RESOURCE_MISMATCH" in statuses.values()
                       else "RESOURCE_UNRESOLVED" if "RESOURCE_UNRESOLVED" in statuses.values()
                       else "PASS")
    if (statuses != preflight["resource_status_by_length"]
            or resource_status != preflight["resource_status"]):
        raise RuntimeError("EXP-0011 resource status does not follow registered rules")
    return receipt


def _save_arm(output: Path, *, config: dict[str, Any], seed: int, arm: str,
              batches: list[PfstyleBatch], training_data_sha: str,
              tests: dict[str, PfstyleBatch], phase: str, started: float,
              on_update: Callable[[], None]) -> dict[str, Any]:
    train = config["training"]
    configure(seed, train["threads"])
    model = build_model(arm, seed, train["graph_seed_base"] + seed)
    initial_sha = _parameter_sha(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=train["learning_rate"])
    destination = output / phase / f"seed-{seed}" / arm
    destination.mkdir(parents=True, exist_ok=False)
    losses = []
    monitors = []
    train_started = time.monotonic()
    try:
        for step, batch in enumerate(batches, 1):
            if step % 50 == 0 and time.monotonic() - started >= config["resource"]["worker_wall_seconds_cap"] - 60:
                raise TimeoutError("EXP-0011 reserved worker time for analysis")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss {phase}/{seed}/{arm}/{step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
            on_update()
            if step in train["monitor_steps"]:
                root = train["monitor_root_base"] + seed * 1_000 + step
                monitor_batch = make_batch(train["monitor_examples"], config["task"]["iid_observations"],
                                           root, Path(config["matrix_file"]))
                metric, _ = evaluate(model, monitor_batch)
                monitors.append({"step": step, "metrics": metric})
    except BaseException as error:
        partial = destination / "partial.pt"
        torch.save(model.state_dict(), partial)
        write_json(destination / "partial.json", {
            "phase": phase, "seed": seed, "arm": arm, "completed_updates": len(losses),
            "losses": losses, "error": f"{type(error).__name__}: {error}",
            "partial_checkpoint_sha256": _file_sha(partial),
        })
        raise
    training_seconds = time.monotonic() - train_started
    metrics = {}
    predictions = {}
    inference_seconds = {}
    for label, batch in tests.items():
        eval_started = time.monotonic()
        metric, arrays = evaluate(model, batch)
        inference_seconds[label] = time.monotonic() - eval_started
        metrics[label] = metric
        predictions.update({f"{label}_{key}": array for key, array in arrays.items()})
    checkpoint = destination / "final.pt"
    torch.save(model.state_dict(), checkpoint)
    np.savez_compressed(destination / "predictions.npz", **predictions)
    resource = {label: resource_account(model, batch.x.shape[1], len(batch.x))
                for label, batch in tests.items()}
    result = {
        "phase": phase, "seed": seed, "arm": arm, "status": "COMPLETED",
        "train_updates": len(batches), "train_examples": len(batches) * train["batch_size"],
        "train_tokens": sum(batch.x.shape[0] * batch.x.shape[1] for batch in batches),
        "training_forward_linear_flops_proxy": resource_account(
            model, config["task"]["iid_observations"], train["batch_size"]
        )["dense_forward_linear_flops_per_batch"] * len(batches),
        "training_data_sha256": training_data_sha,
        "initial_parameters_sha256": initial_sha,
        "final_parameters_sha256": _parameter_sha(model),
        "graph": model.graph_info,
        "graph_mask_sha256": digest(model.mask.detach().cpu().numpy().tobytes()),
        "resources": resource, "metrics": metrics,
        "monitoring": monitors, "train_losses": losses,
        "training_seconds_including_monitor": training_seconds,
        "inference_seconds_by_condition": inference_seconds,
        "checkpoint_file": str(checkpoint.relative_to(output)),
        "checkpoint_sha256": _file_sha(checkpoint),
        "predictions_file": str((destination / "predictions.npz").relative_to(output)),
        "predictions_sha256": _file_sha(destination / "predictions.npz"),
    }
    write_json(destination / "result.json", result)
    return result


def _qualification(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    q = config["qualification"]
    by_arm_seed = {(row["arm"], row["seed"]): row for row in rows}
    learned = all(by_arm_seed[arm, seed]["metrics"]["iid16"]["final_joint_kl_to_oracle"] <
                  q["iid_kl_each_seed_max_exclusive"]
                  for seed in q["seeds"] for arm in q["controls"])
    values = {arm: [by_arm_seed[arm, seed]["metrics"]["ood32"]["final_joint_kl_to_oracle"]
                    for seed in q["seeds"]] for arm in q["controls"]}
    def in_window(value: float, limits: list[float]) -> bool:
        return limits[0] <= value <= limits[1]
    headroom = all(
        in_window(float(np.mean(values[arm])), q[f"{prefix}_ood_mean_kl_window"])
        and all(in_window(value, q[f"{prefix}_ood_each_kl_window"]) for value in values[arm])
        for arm, prefix in (("S68_sparse", "sparse"), ("S68_dense", "dense"))
    )
    return {"status": "PASS" if learned and headroom else
            "LEARNING_FAILURE" if not learned else "CONTROL_FLOOR_OR_CEILING",
            "iid_learning_ok": learned, "ood_headroom_ok": headroom,
            "ood_kl_by_arm": values}


def run(config_path: Path, output_override: Path | None = None) -> None:
    config = json.loads(config_path.read_text())
    if (config["experiment_id"], config["version"], config["workflow"]) != CONFIG_ID:
        raise ValueError("wrong registered EXP-0011 config")
    if config["models"] != list(ARMS):
        raise ValueError("registered model list changed")
    if git_output("status", "--porcelain"):
        raise RuntimeError("formal EXP-0011 study requires clean committed source")
    if _file_sha(Path(config["matrix_file"])) != config["matrix_json_sha256"]:
        raise RuntimeError("registered matrix JSON differs")
    if _file_sha(Path("artifacts/validation/exp0011_matrix_verify.json")) != config["matrix_verify_report_sha256"]:
        raise RuntimeError("registered matrix independent verification differs")
    suite = json.loads(Path("artifacts/validation/suite.json").read_text())
    preflight_path = Path("artifacts/validation/exp0011_preflight.json")
    preflight = json.loads(preflight_path.read_text())
    current_source = source_fingerprints()
    if (not suite["pass"] or suite["source_sha256"] != current_source
            or preflight["source_sha256"] != current_source
            or preflight["config_sha256"] != _file_sha(config_path)
            or preflight["matrix_json_sha256"] != config["matrix_json_sha256"]
            or not preflight["engineering_valid"]):
        raise RuntimeError("EXP-0011 source/config validation mismatch")
    preflight_receipt = _verify_preflight(config, config_path, preflight,
                                          preflight_path, current_source)
    output = output_override or Path(config["output"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    manifest: dict[str, Any] = {
        "experiment_id": "EXP-0011", "version": "v6", "workflow": "pfstyle_field",
        "status": "RUNNING", "stage": "initial", "config": config,
        "config_sha256": _file_sha(config_path),
        "matrix_json_sha256": config["matrix_json_sha256"],
        "preflight": preflight, "preflight_sha256": _file_sha(preflight_path),
        "preflight_receipt": preflight_receipt,
        "preflight_receipt_sha256": _file_sha(preflight_path.with_name(preflight_path.stem + "_receipt.json")),
        "validation": suite, "source_sha256": current_source,
        "git_commit": git_output("rev-parse", "HEAD"), "git_status": "",
        "lock_sha256": _file_sha(Path("uv.lock")),
        "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "torch": torch.__version__, "numpy": np.__version__,
        "packages": dict(sorted((distribution.metadata["Name"], distribution.version)
                                for distribution in importlib.metadata.distributions()
                                if distribution.metadata["Name"])),
        "research_updates": 0, "records": [],
    }
    write_json(output / "manifest.json", manifest)
    def record_update() -> None:
        manifest["research_updates"] += 1
        if manifest["research_updates"] > config["resource"]["total_update_cap"]:
            raise RuntimeError("EXP-0011 total update cap exceeded")
        if manifest["research_updates"] % 50 == 0:
            write_json(output / "manifest.json", manifest)
    try:
        effective_resource_status = ("INCOMPLETE_BUDGET" if
                                     preflight_receipt["budget_status"] != "PASS" else
                                     preflight["resource_status"])
        if effective_resource_status != "PASS":
            manifest.update(status="RESOURCE_STOP", stage="resource",
                            stop_reason=effective_resource_status)
            return
        manifest["stage"] = "task_information"
        info = task_information(config)
        write_json(output / "task_information.json", info)
        manifest["task_information"] = info
        if info["status"] != "PASS":
            manifest.update(status="TASK_STOP", stop_reason=info["status"])
            return
        manifest["stage"] = "qualification"
        q = config["qualification"]
        qual_results = []
        for seed in q["seeds"]:
            tests = {label: make_batch(q["test_examples_per_length"], length,
                                       q["test_root_base"] + seed, Path(config["matrix_file"]))
                     for label, length in (("iid16", 16), ("ood32", 32))}
            _seal_test_streams(output, "qualification", seed, tests, q["test_root_base"] + seed)
            batches, batch_sha = _training_batches(config, seed)
            for arm in q["controls"]:
                result = _save_arm(output, config=config, seed=seed, arm=arm, batches=batches,
                                   training_data_sha=batch_sha, tests=tests,
                                   phase="qualification", started=started, on_update=record_update)
                qual_results.append(result)
                manifest["records"].append({"phase": "qualification", "seed": seed, "arm": arm})
                write_json(output / "manifest.json", manifest)
        qualification = _qualification(config, qual_results)
        write_json(output / "qualification.json", qualification)
        manifest["qualification"] = qualification
        if qualification["status"] != "PASS":
            manifest.update(status="QUALIFICATION_STOP", stop_reason=qualification["status"])
            return
        for phase, section in (("formal", "formal"), ("replication", "replication")):
            manifest["stage"] = phase
            block = config["formal"] if phase == "formal" else config["replication"]
            for seed in block["seeds"]:
                tests = {label: make_batch(block["test_examples_per_length"], length,
                                           block["test_root_base"] + seed, Path(config["matrix_file"]))
                         for label, length in (("iid16", 16), ("ood32", 32))}
                _seal_test_streams(output, phase, seed, tests, block["test_root_base"] + seed)
                batches, batch_sha = _training_batches(config, seed)
                offset = seed % len(ARMS)
                arms = ARMS[offset:] + ARMS[:offset]
                for arm in arms:
                    result = _save_arm(output, config=config, seed=seed, arm=arm, batches=batches,
                                       training_data_sha=batch_sha, tests=tests,
                                       phase=phase, started=started, on_update=record_update)
                    manifest["records"].append({"phase": phase, "seed": seed, "arm": arm})
                    write_json(output / "manifest.json", manifest)
        if manifest["research_updates"] != 97_200:
            raise RuntimeError("EXP-0011 study update count differs from frozen protocol")
        manifest.update(status="COMPLETED", stage="done")
    except BaseException as error:
        manifest.update(status="INCOMPLETE_BUDGET" if isinstance(error, TimeoutError) else "ENGINEERING_ABORT",
                        error=f"{type(error).__name__}: {error}")
    finally:
        manifest["wall_seconds_before_manifest_fsync"] = time.monotonic() - started
        write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "stage": manifest["stage"],
                      "research_updates": manifest["research_updates"],
                      "output": str(output)}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
