"""Registered curriculum/task-condition suites without changing historical runners."""
import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .models import build_model
from .resources import resource_account
from .run import configure, digest, evaluate, git_output, latency, loss_on, source_fingerprints, write_json
from .task import Batch, DISTRACTOR, make_batch


def variant_batch(pairs: int, delay: int, rng: np.random.Generator, payload: str) -> Batch:
    if payload not in {"noise", "full", "blank", "repeat", "id_only", "value_only"}:
        raise ValueError(payload)
    batch = make_batch(pairs, delay, rng)
    if payload in {"noise", "full"}:
        return batch

    nuisance = batch.x[:, :, 8 + DISTRACTOR].bool()
    distractor_payload = batch.x[:, :, :8]
    if payload == "blank":
        distractor_payload[nuisance] = 0
    elif payload == "repeat":
        # Each generated sequence has at least the two post-evidence distractors.
        first_index = nuisance.to(torch.int64).argmax(dim=1)
        rows = torch.arange(batch.x.shape[0])
        first_payload = distractor_payload[rows, first_index]
        repeated_payload = first_payload[:, None, :].expand(-1, batch.x.shape[1], -1)
        distractor_payload[nuisance] = repeated_payload[nuisance]
    elif payload == "id_only":
        distractor_payload[:, :, 4:8][nuisance] = 0
    elif payload == "value_only":
        distractor_payload[:, :, :4][nuisance] = 0
    return batch


def schedule_delays(schedule: list[list[int]]) -> list[int]:
    if not schedule or any(delay < 0 or count <= 0 for delay, count in schedule):
        raise ValueError("positive update counts and nonnegative delays required")
    return [delay for delay, count in schedule for _ in range(count)]


def batch_digest(batch: Batch) -> str:
    return digest(batch.x.numpy().tobytes() + batch.pre_target.numpy().tobytes() + batch.final_target.numpy().tobytes())


def run_arm(c: dict[str, Any], base: dict[str, Any], arm: dict[str, Any], seed: int, output: Path) -> dict[str, Any]:
    configure(seed, base["threads"])
    model = build_model(arm["model"], base, seed)
    initial_hash = digest(b"".join(p.detach().numpy().tobytes() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=base["learning_rate"])
    rng = np.random.default_rng(c["train_seed_offset"] + seed)
    delays = schedule_delays(arm["schedule"])
    losses, monitoring = [], {}
    tokens, flops = 0, 0
    start = time.perf_counter()
    for step, delay in enumerate(delays, 1):
        model.train()
        batch = variant_batch(base["batch_size"] // 2, delay, rng, arm["payload"])
        optimizer.zero_grad(set_to_none=True)
        loss = loss_on(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{arm['name']}/{seed}/{step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), base["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.item())
        tokens += batch.x.shape[0] * batch.x.shape[1]
        flops += resource_account(model, batch.x.shape[1], len(batch.x))["dense_forward_linear_flops_per_batch"]
        if step in c["checkpoints"]:
            monitoring_batch = variant_batch(c["monitor_pairs"], delay, np.random.default_rng(c["monitor_seed"] + delay), arm["payload"])
            metrics, _ = evaluate(model, monitoring_batch)
            monitoring[str(step)] = {"training_delay": delay, "metrics": metrics, "dataset_sha256": batch_digest(monitoring_batch)}
            print(json.dumps({"arm": arm["name"], "seed": seed, "update": step, "delay": delay,
                              "monitor_pre": metrics["pre_accuracy"], "monitor_joint": metrics["joint_pair_accuracy"]}), flush=True)
    training_seconds = time.perf_counter() - start
    final, data_hashes, resources, timing, arrays = {}, {}, {}, {}, {}
    for delay in c["eval_delays"]:
        for payload in c["eval_payloads"]:
            key = f"{payload}_delay{delay}"
            batch = variant_batch(c["test_pairs"], delay, np.random.default_rng(c["test_seed"] + delay), payload)
            final[key], predictions = evaluate(model, batch)
            data_hashes[key] = batch_digest(batch)
            resources[key] = resource_account(model, batch.x.shape[1])
            timing[key] = latency(model, batch)
            arrays.update({f"{key}_{k}": v for k, v in predictions.items()})
    result = {"arm": arm, "seed": seed, "status": "COMPLETED", "train_losses": losses,
              "monitoring": monitoring, "metrics": final, "dataset_sha256": data_hashes,
              "resources": resources, "inference_timing": timing,
              "initial_parameters_sha256": initial_hash, "mask_sha256": digest(model.mask.numpy().tobytes()),
              "graph": model.graph_info, "train_examples": len(delays) * base["batch_size"],
              "train_updates": len(delays), "train_tokens": tokens, "training_forward_linear_flops": flops,
              "training_linear_flops_estimate_3x_forward": 3 * flops,
              "runtime_seconds_including_monitor": training_seconds,
              "compute_scope": "Analytical linear operations, not full measured FLOPs; excludes optimizer/nonlinear work."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    torch.save(model.state_dict(), output / "checkpoints/final.pt")
    np.savez_compressed(output / "predictions.npz", **arrays)
    write_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    if c["experiment_id"] == "EXP-0003" and c["version"] != "v2":
        raise ValueError("EXP-0003 v1 superseded before run; use corrected v2")
    base = json.loads(Path(c["base_config"]).read_text())
    if git_output("status", "--porcelain"):
        raise RuntimeError("Commit registered study and validated source before running")
    suite = json.loads(Path("artifacts/validation/suite.json").read_text())
    if not suite["pass"] or suite["source_sha256"] != source_fingerprints():
        raise RuntimeError("Run scripts/verify.py on current source")
    output = args.output or Path(c["output"])
    if output.exists():
        raise FileExistsError(output)
    seeds = c["seeds"] if args.seed is None else [args.seed]
    if any(seed not in c["seeds"] for seed in seeds):
        raise ValueError("unregistered seed")
    import importlib.metadata
    import platform
    import sys
    manifest = {"status": "RUNNING", "config": c, "base_config": base,
                "git_commit": git_output("rev-parse", "HEAD"), "git_status": "",
                "run_kind": "REGISTERED" if args.seed is None else "REPRODUCTION_SUBSET", "seeds_run": seeds,
                "source_sha256": source_fingerprints(), "config_sha256": digest(args.config.read_bytes()),
                "base_config_sha256": digest(Path(c["base_config"]).read_bytes()),
                "lock_sha256": digest(Path("uv.lock").read_bytes()), "validation": suite,
                "platform": platform.platform(), "python": sys.version, "machine": platform.machine(),
                "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"])),
                "device": "cpu", "threads": base["threads"], "torch_build": torch.__config__.show()}
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        for index, seed in enumerate(seeds):
            arms = c["arms"]
            for arm in arms[index % len(arms):] + arms[:index % len(arms)]:
                run_arm(c, base, arm, seed, output / f"seed-{seed}" / arm["name"])
        manifest["status"] = "COMPLETED"
    except BaseException as exc:
        manifest.update(status="FAILED", error=repr(exc))
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - started
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
