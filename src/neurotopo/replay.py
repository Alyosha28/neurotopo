"""Frozen-checkpoint input interventions for EXP-0005."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch

from .models import build_model
from .resources import resource_account
from .run import configure, digest, evaluate, git_output, latency, source_fingerprints, write_json
from .study import batch_digest, variant_batch
from .task import DISTRACTOR, Batch


def target_digest(batch: Batch) -> str:
    return digest(batch.pre_target.numpy().tobytes() + batch.final_target.numpy().tobytes() +
                  batch.prior.numpy().tobytes() + batch.selected.numpy().tobytes() + batch.values.numpy().tobytes())


def pair_joint(predictions: dict[str, np.ndarray]) -> np.ndarray:
    pre = predictions["pre_logits"].argmax(-1) == predictions["pre_target"]
    final = predictions["final_logits"].argmax(-1) == predictions["final_target"]
    return np.all((pre & final).reshape(-1, 2), axis=1)


def verify_variant(reference: Batch, batch: Batch) -> None:
    if reference.pre_index != batch.pre_index or reference.x.shape != batch.x.shape:
        raise ValueError("variant changed sequence length or query time")
    if target_digest(reference) != target_digest(batch):
        raise ValueError("variant changed targets or candidate mapping")
    if not torch.equal(reference.x[:, :, 8:], batch.x[:, :, 8:]):
        raise ValueError("variant changed event tags")
    nuisance = reference.x[:, :, 8 + DISTRACTOR].bool()
    if not torch.equal(reference.x[:, :, :8][~nuisance], batch.x[:, :, :8][~nuisance]):
        raise ValueError("variant changed a non-distractor input")
    if not torch.equal(batch.x[::2, :batch.pre_index + 1], batch.x[1::2, :batch.pre_index + 1]):
        raise ValueError("counterfactual pair prefix mismatch")


def load_verified_model(name: str, seed: int, config: dict[str, Any], copied: dict[str, Any],
                        source_manifest: dict[str, Any]):
    base = source_manifest["base_config"]
    configure(seed, base["threads"])
    model = build_model(name, base, seed)
    initial_hash = digest(b"".join(p.detach().numpy().tobytes() for p in model.parameters()))
    source_result = json.loads((Path("artifacts/EXP-0004/v1") / f"seed-{seed}" / name / "result.json").read_text())
    if initial_hash != source_result["initial_parameters_sha256"]:
        raise ValueError(f"initial parameter mismatch: {name}/{seed}")
    entry = copied[str(seed)][name]
    checkpoint = Path(entry["path"])
    raw = checkpoint.read_bytes()
    if len(raw) != entry["size"] or digest(raw) != entry["sha256"]:
        raise ValueError(f"checkpoint byte mismatch: {name}/{seed}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if digest(model.mask.numpy().tobytes()) != source_result["mask_sha256"]:
        raise ValueError(f"mask mismatch: {name}/{seed}")
    model.eval()
    # Exact old-stream replay authenticates the copied weights against the
    # complete original model result, independent of ZIP serialization bytes.
    for delay in config["replay_delays"]:
        batch = variant_batch(config["test_pairs"], delay,
                              np.random.default_rng(source_manifest["config"]["test_seed"] + delay), "full")
        key = f"noise_delay{delay}"
        metrics, _ = evaluate(model, batch)
        if batch_digest(batch) != source_result["dataset_sha256"][key] or metrics != source_result["metrics"][key]:
            raise ValueError(f"original-score replay failed: {name}/{seed}/delay{delay}")
    return model, source_result, initial_hash, entry["sha256"]


def run_one(name: str, seed: int, config: dict[str, Any], copied: dict[str, Any],
            source_manifest: dict[str, Any], destination: Path) -> dict[str, Any]:
    model, source_result, initial_hash, checkpoint_hash = load_verified_model(name, seed, config, copied, source_manifest)
    result: dict[str, Any] = {"model": name, "seed": seed, "status": "COMPLETED", "metrics": {},
        "input_sha256": {}, "target_sha256": {}, "resources": {}, "inference_timing": {},
        "pair_overlap_with_full": {}, "source_replay": "EXACT", "source_commit": config["source_commit"],
        "checkpoint_sha256": checkpoint_hash, "initial_parameters_sha256": initial_hash,
        "mask_sha256": source_result["mask_sha256"]}
    arrays: dict[str, np.ndarray] = {}
    start = time.perf_counter()
    for delay in config["delays"]:
        reference = variant_batch(config["test_pairs"], delay,
                                  np.random.default_rng(config["test_seed"] + delay), "full")
        full_joint: np.ndarray | None = None
        for payload in config["payloads"]:
            batch = variant_batch(config["test_pairs"], delay,
                                  np.random.default_rng(config["test_seed"] + delay), payload)
            verify_variant(reference, batch)
            key = f"{payload}_delay{delay}"
            metrics, predictions = evaluate(model, batch)
            joint = pair_joint(predictions)
            if payload == "full":
                full_joint = joint
            assert full_joint is not None
            result["metrics"][key] = metrics
            result["input_sha256"][key] = digest(batch.x.numpy().tobytes())
            result["target_sha256"][key] = target_digest(batch)
            result["resources"][key] = resource_account(model, batch.x.shape[1], len(batch.x))
            result["inference_timing"][key] = latency(model, batch)
            result["pair_overlap_with_full"][key] = {
                "both_correct": int((full_joint & joint).sum()),
                "rescued": int((~full_joint & joint).sum()),
                "harmed": int((full_joint & ~joint).sum()),
                "both_wrong": int((~full_joint & ~joint).sum()),
            }
            arrays.update({f"{key}_{field}": value for field, value in predictions.items()})
    result["runtime_seconds"] = time.perf_counter() - start
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / "predictions.npz", **arrays)
    write_json(destination / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, help="registered subset reproduction")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["experiment_id"] != "EXP-0005" or config["version"] != "v2":
        raise ValueError("EXP-0005 v1 superseded before run; use v2")
    if git_output("status", "--porcelain"):
        raise RuntimeError("Commit registered replay and validation before formal execution")
    validation = json.loads(Path("artifacts/validation/suite.json").read_text())
    if not validation["pass"] or validation["source_sha256"] != source_fingerprints():
        raise RuntimeError("Current source-hashed validation is required")
    output = args.output or Path(config["output"])
    if output.exists():
        raise FileExistsError(output)
    source_manifest = json.loads((Path("artifacts") / config["source_experiment"] / "manifest.json").read_text())
    if source_manifest["status"] != "COMPLETED" or not source_manifest["git_commit"].startswith(config["source_commit"]):
        raise ValueError("Unexpected source experiment identity")
    for source in ("models.py", "task.py", "resources.py"):
        path = Path("src/neurotopo") / source
        if digest(path.read_bytes()) != source_manifest["source_sha256"][str(path)]:
            raise ValueError(f"Frozen scientific source drifted: {path}")
    copied = json.loads(Path(config["checkpoint_manifest"]).read_text())["copies"]
    seeds = config["seeds"] if args.seed is None else [args.seed]
    if any(seed not in config["seeds"] for seed in seeds):
        raise ValueError("unregistered seed")
    manifest = {"status": "RUNNING", "experiment_id": config["experiment_id"], "version": config["version"],
        "config": config, "config_path": str(args.config),
        "git_commit": git_output("rev-parse", "HEAD"), "git_status": "",
        "run_kind": "REGISTERED" if args.seed is None else "REPRODUCTION_SUBSET", "seeds_run": seeds,
        "source_sha256": source_fingerprints(), "config_sha256": digest(args.config.read_bytes()),
        "checkpoint_manifest_sha256": digest(Path(config["checkpoint_manifest"]).read_bytes()),
        "source_run_commit": source_manifest["git_commit"],
        "source_run_manifest_sha256": digest((Path("artifacts") / config["source_experiment"] / "manifest.json").read_bytes()),
        "lock_sha256": digest(Path("uv.lock").read_bytes()), "validation": validation,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "machine": platform.machine(), "device": "cpu",
        "torch": torch.__version__, "numpy": np.__version__, "threads": source_manifest["base_config"]["threads"],
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"]))}
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        for seed in seeds:
            order = config["models"][seed % len(config["models"]):] + config["models"][:seed % len(config["models"])]
            for name in order:
                record = run_one(name, seed, config, copied, source_manifest, output / f"seed-{seed}" / name)
                print(json.dumps({"seed": seed, "model": name,
                    "full32": round(record["metrics"]["full_delay32"]["joint_pair_accuracy"], 4),
                    "blank32": round(record["metrics"]["blank_delay32"]["joint_pair_accuracy"], 4)}), flush=True)
        manifest["status"] = "COMPLETED"
    except BaseException as exc:
        manifest.update(status="FAILED", error=repr(exc))
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - started
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
