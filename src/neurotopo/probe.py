"""Fixed linear probes of frozen pre-evidence recurrent states."""
import argparse
from datetime import datetime, timezone
import hashlib
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
from .replay import load_verified_model
from .resources import resource_account
from .run import digest, evaluate, git_output, source_fingerprints, write_json
from .study import batch_digest
from .task import Batch, make_batch


def split_batch(pairs: int, delay: int, seed: int) -> Batch:
    return make_batch(pairs, delay, np.random.default_rng(seed + delay))


def unique_prefix_hash(batch: Batch) -> str:
    return digest(batch.x[::2, :batch.pre_index + 1].numpy().tobytes() +
                  batch.values[::2].numpy().tobytes() + batch.prior[::2].numpy().tobytes())


def check_disjoint_prefixes(splits: dict[str, Batch]) -> None:
    seen: set[bytes] = set()
    for kind in ("train", "validation", "test"):
        batch = splits[kind]
        for prefix in batch.x[::2, :batch.pre_index + 1].numpy():
            key = hashlib.sha256(prefix.tobytes()).digest()
            if key in seen:
                raise ValueError(f"Duplicate counterfactual prefix within/across splits: {kind}")
            seen.add(key)


@torch.no_grad()
def hidden_at_pre_query(model: RecurrentModel, batch: Batch, batch_size: int = 128) -> torch.Tensor:
    """One hidden state per pair; no post-query evidence or suffix is consumed."""
    x = batch.x[::2]
    pieces = []
    weight = model.weight_hh * model.mask
    for start in range(0, len(x), batch_size):
        chunk = x[start:start + batch_size]
        h = chunk.new_zeros((len(chunk), model.hidden))
        for t in range(batch.pre_index + 1):
            h = model.step(chunk[:, t], h, weight)
        pieces.append(h)
    return torch.cat(pieces)


def fit_ridge(states: torch.Tensor, labels: torch.Tensor, penalty: float, std_floor: float):
    if states.ndim != 2 or labels.shape != (len(states), 4) or states.shape[1] != 64:
        raise ValueError("Expected [N,64] states and [N,4] candidate labels")
    x = states.double()
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(std_floor)
    z = (x - mean) / scale
    design = torch.cat([z, torch.ones((len(z), 1), dtype=z.dtype)], dim=1)
    target = F.one_hot(labels.long(), num_classes=4).reshape(len(labels), 16).double()
    gram = design.T @ design
    ridge = torch.eye(65, dtype=gram.dtype) * penalty
    ridge[-1, -1] = 0.0  # unpenalized intercept
    coefficients = torch.linalg.solve(gram + ridge, design.T @ target)
    if not all(torch.isfinite(t).all() for t in (mean, scale, coefficients)):
        raise FloatingPointError("non-finite linear probe")
    return mean, scale, coefficients


def probe_scores(states: torch.Tensor, labels: torch.Tensor, prior: torch.Tensor,
                 mean: torch.Tensor, scale: torch.Tensor, coefficients: torch.Tensor) -> dict[str, float]:
    z = (states.double() - mean) / scale
    design = torch.cat([z, torch.ones((len(z), 1), dtype=z.dtype)], dim=1)
    logits = (design @ coefficients).reshape(len(states), 4, 4)
    guesses = logits.argmax(-1)
    correct = guesses.eq(labels)
    selected = correct.gather(1, prior.long().view(-1, 1)).squeeze(1)
    uncued = (correct.sum(1) - selected.long()).double().sum() / (3 * len(states))
    target = F.one_hot(labels.long(), num_classes=4).double()
    return {"uncued_accuracy": float(uncued.item()), "prior_accuracy": float(selected.double().mean().item()),
            "all_candidate_accuracy": float(correct.double().mean().item()),
            "all_four_correct": float(correct.all(1).double().mean().item()),
            "mean_squared_error": float(((logits - target) ** 2).mean().item())}


@torch.no_grad()
def behavior_on_test(model: RecurrentModel, batch: Batch) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics, arrays = evaluate(model, batch)
    pre_ok = arrays["pre_logits"].argmax(-1) == arrays["pre_target"]
    final_ok = arrays["final_logits"].argmax(-1) == arrays["final_target"]
    pair_early = pre_ok.reshape(-1, 2).all(axis=1)
    pair_final = final_ok.reshape(-1, 2).all(axis=1)
    return metrics, {"early": pair_early, "late": pair_final}


def run_one(model_name: str, seed: int, config: dict[str, Any], copied: dict[str, Any],
            source: dict[str, Any], output: Path) -> dict[str, Any]:
    replay_config = {**config, "replay_delays": source["config"]["eval_delays"],
                     "test_pairs": source["config"]["test_pairs"]}
    model, source_result, initial_hash, checkpoint_hash = load_verified_model(model_name, seed, replay_config, copied, source)
    record: dict[str, Any] = {"model": model_name, "seed": seed, "status": "COMPLETED",
        "source_replay": "EXACT", "source_commit": config["source_commit"],
        "checkpoint_sha256": checkpoint_hash, "initial_parameters_sha256": initial_hash,
        "mask_sha256": source_result["mask_sha256"], "delays": {}}
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    for delay in config["delays"]:
        splits = {kind: split_batch(config[f"{kind}_pairs"], delay, config[f"{kind}_seed"])
                  for kind in ("train", "validation", "test")}
        check_disjoint_prefixes(splits)
        extraction_start = time.perf_counter()
        states = {kind: hidden_at_pre_query(model, batch) for kind, batch in splits.items()}
        extraction_seconds = time.perf_counter() - extraction_start
        for kind, batch in splits.items():
            if len(states[kind]) != config[f"{kind}_pairs"]:
                raise ValueError("Counterfactual prefix was not deduplicated")
            if not torch.equal(torch.sort(batch.values[::2].long(), dim=1).values,
                               torch.arange(4).expand(len(states[kind]), 4)):
                raise ValueError("Candidate labels are not permutations")
        # Check every distinct prefix across all three splits, with identical
        # chunk shape and source weights for exact floating-point comparison.
        parity_start = time.perf_counter()
        for kind, batch in splits.items():
            unique = batch.x[::2]
            for start in range(0, len(unique), 128):
                original_pre, _ = model(unique[start:start + 128], batch.pre_index)
                extracted = model.readout(states[kind][start:start + 128])
                if not torch.equal(original_pre, extracted):
                    raise ValueError(f"Extracted state/readout mismatch: {kind}/{start}")
        parity_seconds = time.perf_counter() - parity_start
        labels = {kind: batch.values[::2].long() for kind, batch in splits.items()}
        prior = {kind: batch.prior[::2].long() for kind, batch in splits.items()}
        fit_start = time.perf_counter()
        mean, scale, coefficients = fit_ridge(states["train"], labels["train"],
                                                config["ridge_lambda"], config["std_floor"])
        fit_seconds = time.perf_counter() - fit_start
        scores = {kind: probe_scores(states[kind], labels[kind], prior[kind], mean, scale, coefficients)
                  for kind in splits}
        behavior_start = time.perf_counter()
        behavior, pair_bits = behavior_on_test(model, splits["test"])
        behavior_seconds = time.perf_counter() - behavior_start
        diag = {"train": {"prefix_sha256": unique_prefix_hash(splits["train"]),
                           "full_task_sha256": batch_digest(splits["train"])},
                "validation": {"prefix_sha256": unique_prefix_hash(splits["validation"]),
                               "full_task_sha256": batch_digest(splits["validation"])},
                "test": {"prefix_sha256": unique_prefix_hash(splits["test"]),
                         "full_task_sha256": batch_digest(splits["test"])}}
        probe_file = output / f"delay-{delay}-probe.npz"
        np.savez_compressed(probe_file, mean=mean.numpy(), scale=scale.numpy(), coefficients=coefficients.numpy())
        record["delays"][str(delay)] = {"scores": scores, "behavior": behavior,
            "split_unique_prefix_counts": {kind: len(states[kind]) for kind in splits},
            "timing_seconds": {"state_extraction": extraction_seconds,
                "readout_equivalence": parity_seconds, "ridge_fit": fit_seconds,
                "behavior_evaluation": behavior_seconds},
            "test_pair_early_sha256": digest(pair_bits["early"].tobytes()),
            "test_pair_late_sha256": digest(pair_bits["late"].tobytes()),
            "test_pair_bits": {"early": pair_bits["early"].astype(np.uint8).tolist(),
                               "late": pair_bits["late"].astype(np.uint8).tolist()},
            "datasets": diag, "probe_parameters": int(coefficients.numel()),
            "probe_coefficients_sha256": digest(coefficients.numpy().tobytes()),
            "probe_file_sha256": digest(probe_file.read_bytes()),
            "resources": resource_account(model, splits["test"].x.shape[1], len(splits["test"].x)),
            "train_only_scaling": True, "ridge_lambda": config["ridge_lambda"],
            "std_floor": config["std_floor"],
            "linear_fit_macs_proxy": config["train_pairs"] * (65 * 65 + 65 * 16),
            "state_extraction_forward_linear_flops_proxy": 2 * sum(len(states[kind]) *
                (batch.pre_index + 1) * model.gates * model.hidden * (model.input_size + model.hidden)
                for kind, batch in splits.items()),
            "linear_fit_proxy_scope": "Fit proxy covers XtX and XtY MACs; extraction proxy counts dense recurrent/input matrix FLOPs on unique prefixes. Both exclude nonlinearities and state/readout checks; wall-clock components recorded separately."}
        print(json.dumps({"model": model_name, "seed": seed, "delay": delay,
                          "uncued_test": scores["test"]["uncued_accuracy"],
                          "uncued_validation": scores["validation"]["uncued_accuracy"]}), flush=True)
    record["runtime_seconds"] = time.perf_counter() - started
    write_json(output / "result.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, help="registered subset reproduction")
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    if c["experiment_id"] != "EXP-0006" or c["version"] != "v2":
        raise ValueError("EXP-0006 v1 superseded before run; use v2")
    if git_output("status", "--porcelain"):
        raise RuntimeError("Commit registered probe and source-hashed validation before execution")
    validation = json.loads(Path("artifacts/validation/suite.json").read_text())
    if not validation["pass"] or validation["source_sha256"] != source_fingerprints():
        raise RuntimeError("Current source-hashed validation required")
    output = args.output or Path(c["output"])
    if output.exists():
        raise FileExistsError(output)
    source_path = Path("artifacts") / c["source_experiment"] / "manifest.json"
    source = json.loads(source_path.read_text())
    if source["status"] != "COMPLETED" or not source["git_commit"].startswith(c["source_commit"]):
        raise ValueError("Source experiment identity mismatch")
    for name in ("models.py", "task.py", "resources.py"):
        path = Path("src/neurotopo") / name
        if digest(path.read_bytes()) != source["source_sha256"][str(path)]:
            raise ValueError(f"Frozen source drift: {path}")
    copied = json.loads(Path(c["checkpoint_manifest"]).read_text())["copies"]
    seeds = c["seeds"] if args.seed is None else [args.seed]
    if any(seed not in c["seeds"] for seed in seeds):
        raise ValueError("Unregistered seed")
    manifest = {"status": "RUNNING", "experiment_id": c["experiment_id"], "version": c["version"],
        "config": c, "config_path": str(args.config), "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": "", "run_kind": "REGISTERED" if args.seed is None else "REPRODUCTION_SUBSET",
        "seeds_run": seeds, "source_sha256": source_fingerprints(),
        "config_sha256": digest(args.config.read_bytes()),
        "checkpoint_manifest_sha256": digest(Path(c["checkpoint_manifest"]).read_bytes()),
        "source_run_manifest_sha256": digest(source_path.read_bytes()),
        "validation": validation, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
        "threads": source["base_config"]["threads"], "lock_sha256": digest(Path("uv.lock").read_bytes())}
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        for seed in seeds:
            names = c["models"]
            for name in names[seed % len(names):] + names[:seed % len(names)]:
                run_one(name, seed, c, copied, source, output / f"seed-{seed}" / name)
        manifest["status"] = "COMPLETED"
    except BaseException as exc:
        manifest.update(status="FAILED", error=repr(exc))
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - started
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
