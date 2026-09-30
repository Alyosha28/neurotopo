"""Immutable registered CPU experiment runner; final checkpoint only."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .models import RecurrentModel, build_model
from .resources import resource_account
from .task import Batch, make_batch


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_fingerprints() -> dict[str, str]:
    return {str(path): digest(path.read_bytes()) for path in sorted(Path("src").rglob("*.py"))
            if not path.name.startswith("._")}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def configure(seed: int, threads: int = 1) -> None:
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def loss_on(model: RecurrentModel, batch: Batch) -> torch.Tensor:
    pre, final = model(batch.x, batch.pre_index)
    return 0.5 * (F.cross_entropy(pre, batch.pre_target) + F.cross_entropy(final, batch.final_target))


@torch.no_grad()
def evaluate(model: RecurrentModel, batch: Batch) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    pres, finals = [], []
    for start in range(0, len(batch.x), 128):
        pre, final = model(batch.x[start:start + 128], batch.pre_index)
        pres.append(pre)
        finals.append(final)
    pre, final = torch.cat(pres), torch.cat(finals)
    pred_pre, pred_final = pre.argmax(-1), final.argmax(-1)
    early = pred_pre == batch.pre_target
    late = pred_final == batch.final_target
    revise = torch.arange(len(early)) % 2 == 1
    joint = early.reshape(-1, 2).all(-1) & late.reshape(-1, 2).all(-1)
    conditional = revise & early
    result = {
        "pairs": len(joint), "joint_pair_accuracy": joint.float().mean().item(),
        "pre_accuracy": early.float().mean().item(),
        "final_accuracy": late.float().mean().item(),
        "confirm_accuracy": late[~revise].float().mean().item(),
        "revise_accuracy": late[revise].float().mean().item(),
        "revise_given_pre_correct": late[conditional].float().mean().item() if conditional.any() else None,
        "revise_given_pre_correct_count": conditional.sum().item(),
        "prior_consistent_error_on_revise": (pred_final[revise] == batch.pre_target[revise]).float().mean().item(),
        "paired_final_accuracy": late.reshape(-1, 2).all(-1).float().mean().item(),
        "evidence_sensitivity": (pred_final[::2] != pred_final[1::2]).float().mean().item(),
        "pre_nll": F.cross_entropy(pre, batch.pre_target).item(),
        "confirm_nll": F.cross_entropy(final[~revise], batch.final_target[~revise]).item(),
        "revise_nll": F.cross_entropy(final[revise], batch.final_target[revise]).item(),
    }
    arrays = {"pre_logits": pre.numpy(), "final_logits": final.numpy(),
              "pre_target": batch.pre_target.numpy(), "final_target": batch.final_target.numpy()}
    result["prediction_sha256"] = digest(b"".join(arrays[k].tobytes() for k in sorted(arrays)))
    return result, arrays


@torch.no_grad()
def latency(model: RecurrentModel, batch: Batch) -> dict[str, float]:
    x = batch.x[:64]
    for _ in range(3):
        model(x, batch.pre_index)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        model(x, batch.pre_index)
        times.append(time.perf_counter() - start)
    median = float(np.median(times))
    return {"median_batch_seconds": median, "batch_size": len(x), "repetitions": 10,
            "tokens_per_second": len(x) * x.shape[1] / median}


def run_one(name: str, seed: int, config: dict[str, Any], destination: Path) -> dict[str, Any]:
    configure(seed, config["threads"])
    start = time.perf_counter()
    model = build_model(name, config, seed)
    construction = time.perf_counter() - start
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    rng = np.random.default_rng(config["train_seed_offset"] + seed)
    losses = []
    train_tokens, linear_flops = 0, 0
    train_start = time.perf_counter()
    model.train()
    for step in range(config["updates"]):
        delay = config["train_delays"][step % len(config["train_delays"])]
        batch = make_batch(config["batch_size"] // 2, delay, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_on(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at {name}/{seed}/{step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.item())
        train_tokens += len(batch.x) * batch.x.shape[1]
        linear_flops += resource_account(model, batch.x.shape[1], len(batch.x))["dense_forward_linear_flops_per_batch"]
    train_seconds = time.perf_counter() - train_start
    metrics, accounts, timing, arrays, data_hashes = {}, {}, {}, {}, {}
    for delay in config["eval_delays"]:
        batch = make_batch(config["eval_pairs"], delay, np.random.default_rng(config["eval_seed"] + delay))
        key = str(delay)
        data_hashes[key] = digest(batch.x.numpy().tobytes() + batch.pre_target.numpy().tobytes() + batch.final_target.numpy().tobytes())
        metrics[key], predictions = evaluate(model, batch)
        arrays.update({f"delay_{delay}_{k}": v for k, v in predictions.items()})
        accounts[key] = resource_account(model, batch.x.shape[1])
        timing[key] = latency(model, batch)
    result = {
        "model": name, "seed": seed, "status": "COMPLETED", "metrics": metrics,
        "resources": accounts, "graph": model.graph_info,
        "mask_sha256": digest(model.mask.numpy().tobytes()), "dataset_sha256": data_hashes,
        "train_losses": losses, "final_train_loss": losses[-1],
        "train_seconds": train_seconds, "construction_seconds": construction,
        "inference_timing": timing, "train_examples": config["updates"] * config["batch_size"],
        "train_tokens": train_tokens, "training_forward_linear_flops": linear_flops,
        "estimated_train_linear_flops_forward_backward": 3 * linear_flops,
        "training_compute_note": "3x forward is a rough backward estimate, not measured FLOPs; optimizer/nonlinear/mask work excluded.",
    }
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "final.pt")
    result["checkpoint_sha256"] = digest((checkpoint_dir / "final.pt").read_bytes())
    np.savez_compressed(destination / "predictions.npz", **arrays)
    write_json(destination / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, help="Subset for explicit reproduction, not a changed scientific protocol")
    parser.add_argument("--allow-dirty", action="store_true", help="Engineering validation only; prevents scientific status")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["version"] != "v2":
        raise ValueError("v1 was superseded before execution; this source implements v2 diagonal-matched controls")
    if config["device"] != "cpu" or config["batch_size"] % 2:
        raise ValueError("Reference protocol requires CPU and even counterfactual batch size")
    status = git_output("status", "--porcelain")
    if status and not args.allow_dirty:
        raise RuntimeError("Commit registered code/config before a formal run; working tree is dirty")
    output = args.output or Path(config["output"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    seeds = config["seeds"] if args.seed is None else [args.seed]
    if any(seed not in config["seeds"] for seed in seeds):
        raise ValueError("Reproduction seed must belong to registered seed list")
    sources = source_fingerprints()
    validation_path = Path("artifacts/validation/suite.json")
    validation = json.loads(validation_path.read_text())
    if not validation["pass"] or validation["source_sha256"] != sources:
        raise RuntimeError("Run scripts/verify.py on the current source before scientific execution")
    manifest = {
        "experiment_id": config["experiment_id"], "version": config["version"], "config": config,
        "config_sha256": digest(args.config.read_bytes()), "source_sha256": sources,
        "git_commit": git_output("rev-parse", "HEAD"), "git_status": status,
        "validation": validation, "validation_sha256": digest(validation_path.read_bytes()),
        "run_kind": "ENGINEERING_ONLY" if status else ("REPRODUCTION_SUBSET" if args.seed is not None else "REGISTERED"),
        "seeds_run": seeds, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "cpu_count": os.cpu_count(),
        "torch": torch.__version__, "numpy": np.__version__, "device": config["device"],
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"])),
        "torch_build": torch.__config__.show(), "uv_lock_sha256": digest(Path("uv.lock").read_bytes()),
        "status": "RUNNING",
    }
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        for seed in seeds:
            # Rotate deterministic order to reduce systematic order/thermal confounding.
            models = config["models"]
            order = models[seed % len(models):] + models[:seed % len(models)]
            for name in order:
                result = run_one(name, seed, config, output / f"seed-{seed}" / name)
                scores = {d: round(m["joint_pair_accuracy"], 4) for d, m in result["metrics"].items()}
                print(json.dumps({"seed": seed, "model": name, "joint_pair_accuracy": scores,
                                  "train_seconds": round(result["train_seconds"], 2)}), flush=True)
    except BaseException as exc:
        manifest.update(status="FAILED", error=repr(exc), wall_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        raise
    manifest.update(status="COMPLETED", wall_seconds=time.perf_counter() - started)
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
