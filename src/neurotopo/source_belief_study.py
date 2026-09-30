"""Frozen EXP-0009 matched recurrent-structure posterior-transfer runner."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .models import RecurrentModel
from .resources import resource_account
from .run import configure, digest, git_output, source_fingerprints, write_json
from .source_belief_task import (HYPOTHESES, INPUT_SIZE, REGIMES,
                                 SourceBeliefBatch, make_source_belief_batch)


def schedule(seed: int, updates: int) -> list[int]:
    if updates != 1500:
        raise ValueError("Registered schedule requires 1500 updates")
    rng = np.random.default_rng(seed)
    values: list[int] = []
    for _ in range(5):
        block = np.repeat(np.arange(3), 100)
        rng.shuffle(block)
        values.extend(int(d) for d in block)
    return values


def build_arm(name: str, config: dict[str, Any], seed: int) -> RecurrentModel:
    topology, groups = {"multi_gru": ("block", 4), "random_gru": ("random", 4),
                        "gru_wide": ("dense", 1)}[name]
    configure(seed, config["threads"])
    return RecurrentModel(INPUT_SIZE, config["hidden_size"], HYPOTHESES * REGIMES,
                          groups=groups, topology=topology,
                          graph_seed=config["graph_seed_offset"] + seed)


def parameter_sha256(model: RecurrentModel) -> str:
    return digest(b"".join(parameter.detach().numpy().tobytes() for parameter in model.parameters()))


def update_batch_hash(hasher: Any, batch: SourceBeliefBatch) -> None:
    for field in ("x", "pre_joint", "final_joint", "truth_h", "truth_r"):
        hasher.update(getattr(batch, field).numpy().tobytes())


def batch_sha256(batch: SourceBeliefBatch) -> str:
    hasher = hashlib.sha256()
    update_batch_hash(hasher, batch)
    return hasher.hexdigest()


def paired_evaluation_batches(config: dict[str, Any]) -> dict[int, SourceBeliefBatch]:
    batches = {d: make_source_belief_batch(config["test_examples"],
                                            np.random.default_rng(config["test_seed"]),
                                            distractors_per_evidence=d)
               for d in config["eval_distractors_per_evidence"]}
    first = batches[config["eval_distractors_per_evidence"][0]]
    for d, batch in batches.items():
        for field in ("truth_h", "truth_r", "evidence_cues", "evidence_sources", "pre_joint", "final_joint"):
            if not torch.equal(getattr(first, field), getattr(batch, field)):
                raise ValueError(f"Cross-delay pairing failed: d={d} field={field}")
    return batches


def cross_block_edges(model: RecurrentModel) -> int:
    base = model.mask[:model.hidden].bool().numpy()
    width = model.hidden // 4
    blocks = np.arange(model.hidden) // width
    return int((base & (blocks[:, None] != blocks[None, :])).sum())


def loss_on(model: RecurrentModel, batch: SourceBeliefBatch) -> torch.Tensor:
    pre, final = model(batch.x, batch.pre_index)
    return 0.5 * (-(batch.pre_joint * F.log_softmax(pre, dim=1)).sum(1).mean()
                  - (batch.final_joint * F.log_softmax(final, dim=1)).sum(1).mean())


@torch.no_grad()
def evaluate(model: RecurrentModel, batch: SourceBeliefBatch) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    pre_rows, final_rows = [], []
    for start in range(0, len(batch.x), 256):
        pre, final = model(batch.x[start:start + 256], batch.pre_index)
        pre_rows.append(pre)
        final_rows.append(final)
    pre_logits = torch.cat(pre_rows).numpy()
    final_logits = torch.cat(final_rows).numpy()
    pre = F.softmax(torch.from_numpy(pre_logits), dim=1).numpy().astype(np.float64)
    final = F.softmax(torch.from_numpy(final_logits), dim=1).numpy().astype(np.float64)
    target_pre = batch.pre_joint.numpy().astype(np.float64)
    target_final = batch.final_joint.numpy().astype(np.float64)

    def kl(target: np.ndarray, predicted: np.ndarray) -> float:
        return float((target * (np.log(np.clip(target, 1e-12, 1)) -
                                np.log(np.clip(predicted, 1e-12, 1)))).sum(1).mean())

    truth_h, truth_r = batch.truth_h.numpy(), batch.truth_r.numpy()
    joint_truth = truth_h * REGIMES + truth_r
    marginals = final.reshape(-1, HYPOTHESES, REGIMES)
    result = {
        "pre_joint_kl_to_oracle": kl(target_pre, pre),
        "final_joint_kl_to_oracle": kl(target_final, final),
        "pre_joint_oracle_map_agreement": float((pre.argmax(1) == target_pre.argmax(1)).mean()),
        "final_joint_oracle_map_agreement": float((final.argmax(1) == target_final.argmax(1)).mean()),
        "final_h_true_accuracy": float((marginals.sum(2).argmax(1) == truth_h).mean()),
        "final_r_true_accuracy": float((marginals.sum(1).argmax(1) == truth_r).mean()),
        "final_joint_true_accuracy": float((final.argmax(1) == joint_truth).mean()),
        "final_joint_true_nll": float(-np.log(np.clip(final[np.arange(len(joint_truth)), joint_truth], 1e-12, 1)).mean()),
    }
    arrays = {"pre_logits": pre_logits, "final_logits": final_logits,
              "pre_target": batch.pre_joint.numpy(), "final_target": batch.final_joint.numpy(),
              "truth_h": truth_h, "truth_r": truth_r}
    result["prediction_sha256"] = digest(b"".join(arrays[key].tobytes() for key in sorted(arrays)))
    return result, arrays


def run_seed(config: dict[str, Any], seed: int, output: Path,
             evaluation: dict[int, SourceBeliefBatch], preflight: dict[str, Any]) -> None:
    models = {name: build_arm(name, config, seed) for name in config["models"]}
    initial = {name: parameter_sha256(model) for name, model in models.items()}
    if initial["multi_gru"] != initial["random_gru"]:
        raise ValueError(f"Primary initialization mismatch seed={seed}")
    expected = next(row for row in preflight["seeds"] if row["seed"] == seed)
    for name, model in models.items():
        if initial[name] != expected["models"][name]["initial_parameters_sha256"]:
            raise ValueError(f"Preflight initial-parameter mismatch seed={seed} model={name}")
        mask_sha = digest(model.mask.numpy().tobytes())
        if mask_sha != expected["models"][name]["mask_sha256"]:
            raise ValueError(f"Preflight mask mismatch seed={seed} model={name}")
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
                  for name, model in models.items()}
    monitoring = {name: {} for name in models}
    training_seconds = {name: 0.0 for name in models}
    training_tokens = {name: 0 for name in models}
    training_flops = {name: 0 for name in models}
    data_hasher = hashlib.sha256()
    validation = make_source_belief_batch(config["monitor_examples"],
                                          np.random.default_rng(config["monitor_seed"]))
    validation_hash = batch_sha256(validation)
    delays = schedule(config["schedule_seed"], config["updates"])
    schedule_sha = digest(np.asarray(delays, dtype=np.int8).tobytes())
    if schedule_sha != preflight["schedule_sha256"] or validation_hash != preflight["monitor_dataset_sha256"]:
        raise ValueError("Preflight schedule/monitor data mismatch")

    for step, d in enumerate(delays, 1):
        batch = make_source_belief_batch(config["batch_size"],
                                         np.random.default_rng(config["train_seed_base"] + seed * 10_000 + step),
                                         distractors_per_evidence=d)
        update_batch_hash(data_hasher, batch)
        for name, model in models.items():
            model.train()
            started = time.perf_counter()
            optimizers[name].zero_grad(set_to_none=True)
            loss = loss_on(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss seed={seed} model={name} step={step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
            optimizers[name].step()
            training_seconds[name] += time.perf_counter() - started
            training_tokens[name] += len(batch.x) * batch.x.shape[1]
            training_flops[name] += resource_account(model, batch.x.shape[1], len(batch.x))[
                "dense_forward_linear_flops_per_batch"]
            if step in config["monitor_steps"]:
                metrics, _ = evaluate(model, validation)
                monitoring[name][str(step)] = {"training_d": d, "train_batch_loss": float(loss.item()),
                                               "validation": metrics, "dataset_sha256": validation_hash}
                print(json.dumps({"seed": seed, "model": name, "step": step,
                                  "validation_final_kl": metrics["final_joint_kl_to_oracle"]}), flush=True)
    if data_hasher.hexdigest() != expected["training_dataset_sha256"]:
        raise ValueError(f"Preflight training stream mismatch seed={seed}")
    if len(set(training_tokens.values())) != 1 or len(set(training_flops.values())) != 1:
        raise ValueError(f"Training resource mismatch seed={seed}")
    for name, model in models.items():
        destination = output / f"seed-{seed}" / name
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "checkpoints").mkdir()
        checkpoint = destination / "checkpoints" / "final.pt"
        torch.save(model.state_dict(), checkpoint)
        metrics, data_hashes, resources, timings, arrays = {}, {}, {}, {}, {}
        for d, batch in evaluation.items():
            started = time.perf_counter()
            scores, predictions = evaluate(model, batch)
            key = f"d{d}"
            metrics[key] = scores
            timings[key] = time.perf_counter() - started
            data_hashes[key] = batch_sha256(batch)
            resources[key] = resource_account(model, batch.x.shape[1], config["batch_size"])
            arrays.update({f"{key}_{part}": value for part, value in predictions.items()})
        if data_hashes != preflight["final_dataset_sha256"]:
            raise ValueError(f"Preflight final stream mismatch seed={seed} model={name}")
        np.savez_compressed(destination / "predictions.npz", **arrays)
        result = {
            "status": "COMPLETED", "seed": seed, "model": name,
            "initial_parameters_sha256": initial[name],
            "final_parameters_sha256": parameter_sha256(model),
            "mask_sha256": digest(model.mask.numpy().tobytes()),
            "graph": model.graph_info, "cross_original_block_edges": cross_block_edges(model),
            "training_dataset_sha256": data_hasher.hexdigest(),
            "training_schedule_sha256": schedule_sha,
            "train_updates": config["updates"], "train_examples": config["updates"] * config["batch_size"],
            "train_tokens": training_tokens[name], "training_forward_linear_flops": training_flops[name],
            "training_seconds": training_seconds[name], "monitoring": monitoring[name],
            "metrics": metrics, "dataset_sha256": data_hashes, "resources": resources,
            "inference_seconds": timings,
            "checkpoint_sha256": digest(checkpoint.read_bytes()),
            "checkpoint_file": str(checkpoint.relative_to(output)),
            "predictions_sha256": digest((destination / "predictions.npz").read_bytes()),
            "compute_scope": "Analytical dense forward linear FLOPs; full training FLOPs and peak memory not measured.",
        }
        write_json(destination / "result.json", result)


def run(config_path: Path, output_override: Path | None = None, subset_seed: int | None = None) -> None:
    config = json.loads(config_path.read_text())
    if (config.get("experiment_id"), config.get("version"), config.get("workflow")) != (
            "EXP-0009", "v1", "source_belief") or config["device"] != "cpu" or config["threads"] != 1:
        raise ValueError("Expected frozen EXP-0009 v1 CPU source-belief configuration")
    if config["models"] != ["multi_gru", "random_gru", "gru_wide"] or config["seeds"] != list(range(9401, 9411)):
        raise ValueError("Registered models and seeds changed")
    if git_output("status", "--porcelain"):
        raise RuntimeError("Formal run requires a clean committed source checkout")
    suite = json.loads(Path("artifacts/validation/suite.json").read_text())
    preflight = json.loads(Path("artifacts/validation/exp0009_preflight.json").read_text())
    source_hashes = source_fingerprints()
    config_hash = digest(config_path.read_bytes())
    if (not suite["pass"] or suite["source_sha256"] != source_hashes or
            not preflight["pass"] or preflight["source_sha256"] != source_hashes or
            preflight["config_sha256"] != config_hash or
            preflight["script_sha256"] != digest(Path("scripts/preflight_exp0009.py").read_bytes())):
        raise RuntimeError("Current source/config does not match validation and preflight")
    output = output_override or Path(config["output"])
    if output.exists():
        raise FileExistsError(output)
    seeds = config["seeds"] if subset_seed is None else [subset_seed]
    if any(seed not in config["seeds"] for seed in seeds):
        raise ValueError("Unregistered seed")
    configure(config["seeds"][0], config["threads"])
    evaluation = paired_evaluation_batches(config)
    if {f"d{d}": batch_sha256(batch) for d, batch in evaluation.items()} != preflight["final_dataset_sha256"]:
        raise RuntimeError("Final test stream does not match preflight")
    manifest = {
        "status": "RUNNING", "experiment_id": config["experiment_id"], "version": config["version"],
        "config": config, "config_path": str(config_path), "config_sha256": config_hash,
        "run_kind": "REGISTERED" if subset_seed is None else "REPRODUCTION_SUBSET",
        "seeds_run": seeds, "git_commit": git_output("rev-parse", "HEAD"), "git_status": "",
        "source_sha256": source_hashes, "validation": suite,
        "preflight_sha256": digest(Path("artifacts/validation/exp0009_preflight.json").read_bytes()),
        "preflight": preflight, "lock_sha256": digest(Path("uv.lock").read_bytes()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "threads": 1,
        "packages": dict(sorted((distribution.metadata["Name"], distribution.version)
                                for distribution in importlib.metadata.distributions()
                                if distribution.metadata["Name"])),
        "test_dataset_sha256": preflight["final_dataset_sha256"],
    }
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        for seed in seeds:
            run_seed(config, seed, output, evaluation, preflight)
        manifest["status"] = "COMPLETED"
    except BaseException as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - started
        write_json(output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, help="Declared reproduction subset; not a full study decision")
    args = parser.parse_args()
    run(args.config, args.output, args.seed)


if __name__ == "__main__":
    main()
