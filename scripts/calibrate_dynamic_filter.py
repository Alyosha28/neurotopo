"""Frozen CAL-0006 v2 single-stream benchmark qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from neurotopo.dynamic_filter_task import (INPUT_SIZE, MODES, DynamicFilterBatch,
                                           make_dynamic_filter_batch)
from neurotopo.models import RecurrentModel
from neurotopo.resources import resource_account
from neurotopo.run import configure, digest, git_output, source_fingerprints


SEEDS = (601, 602, 603)
UPDATES = 1200
BATCH_SIZE = 128
EVAL_EXAMPLES = 8192
WALL_LIMIT_SECONDS = 900.0
PROJECT_ROOT = Path(".")
OUTPUT = PROJECT_ROOT / "artifacts/calibration/CAL-0006_v3.json"
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def train_seed(model_seed: int, step: int) -> int:
    return 6_000_000 + model_seed * 10_000 + step


def prefix_digest(batch: DynamicFilterBatch) -> str:
    return digest(b"".join((batch.truth.numpy().tobytes(),
                            batch.controls[:, :8].numpy().tobytes(),
                            batch.observations[:, :8].numpy().tobytes())))


def paired_evaluation() -> tuple[DynamicFilterBatch, DynamicFilterBatch, str]:
    iid = make_dynamic_filter_batch(EVAL_EXAMPLES, 8, 6_400_001)
    ood = make_dynamic_filter_batch(EVAL_EXAMPLES, 12, 6_400_001)
    if not (torch.equal(iid.truth, ood.truth)
            and torch.equal(iid.controls, ood.controls[:, :8])
            and torch.equal(iid.observations, ood.observations[:, :8])
            and torch.equal(iid.x[:, :9], ood.x[:, :9])
            and torch.equal(iid.pre_joint, ood.pre_joint)
            and torch.equal(iid.posterior_history, ood.posterior_history[:, :8])
            and prefix_digest(iid) == prefix_digest(ood)):
        raise ValueError("CAL-0006 IID/OOD evaluation pairing failed")
    return iid, ood, prefix_digest(iid)


def oracle_metrics(batch: DynamicFilterBatch) -> dict[str, float]:
    pre, final = batch.pre_joint.numpy(), batch.final_joint.numpy()
    truth = batch.truth.numpy()
    entropy = -(final * np.log(np.clip(final, 1e-12, 1))).sum(1)
    result = {
        "final_true_map_accuracy": float((final.argmax(1) == truth).mean()),
        "pre_true_map_accuracy": float((pre.argmax(1) == truth).mean()),
        "pre_final_map_change_rate": float((pre.argmax(1) != final.argmax(1)).mean()),
        "final_entropy_nats": float(entropy.mean()),
        "uniform_kl_to_oracle_nats": float(np.log(MODES) - entropy.mean()),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite CAL-0006 oracle metric")
    return result


def loss_on(model: RecurrentModel, batch: DynamicFilterBatch) -> torch.Tensor:
    pre, final = model(batch.x, batch.pre_index)
    return 0.5 * (-(batch.pre_joint * F.log_softmax(pre, dim=1)).sum(1).mean()
                  - (batch.final_joint * F.log_softmax(final, dim=1)).sum(1).mean())


@torch.no_grad()
def model_metrics(model: RecurrentModel, batch: DynamicFilterBatch) -> dict[str, float | int | None]:
    model.eval()
    pre_rows, final_rows = [], []
    forward_seconds = 0.0
    for start in range(0, len(batch.x), 256):
        then = time.perf_counter()
        pre, final = model(batch.x[start:start + 256], batch.pre_index)
        forward_seconds += time.perf_counter() - then
        pre_rows.append(F.softmax(pre, dim=1))
        final_rows.append(F.softmax(final, dim=1))
    pre = torch.cat(pre_rows).numpy().astype(np.float64)
    final = torch.cat(final_rows).numpy().astype(np.float64)
    target_pre = batch.pre_joint.numpy().astype(np.float64)
    target_final = batch.final_joint.numpy().astype(np.float64)

    def per_example_kl(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        return (target * (np.log(np.clip(target, 1e-12, 1)) -
                          np.log(np.clip(prediction, 1e-12, 1)))).sum(1)

    changed = target_pre.argmax(1) != target_final.argmax(1)
    final_kl = per_example_kl(target_final, final)
    result = {
        "pre_joint_kl_to_oracle": float(per_example_kl(target_pre, pre).mean()),
        "final_joint_kl_to_oracle": float(final_kl.mean()),
        "final_joint_oracle_map_agreement": float((final.argmax(1) == target_final.argmax(1)).mean()),
        "final_true_map_accuracy": float((final.argmax(1) == batch.truth.numpy()).mean()),
        "oracle_revision_count": int(changed.sum()),
        "oracle_revision_subset_final_kl": float(final_kl[changed].mean()) if changed.any() else None,
        "model_forward_seconds": forward_seconds,
    }
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise FloatingPointError("Nonfinite CAL-0006 model metric")
    return result


def model_digest(model: RecurrentModel) -> str:
    return digest(b"".join(parameter.detach().numpy().tobytes() for parameter in model.parameters()))


def build_arm(name: str, seed: int) -> RecurrentModel:
    configure(seed, 1)
    if name == "dense":
        return RecurrentModel(INPUT_SIZE, 64, MODES, groups=1, topology="dense")
    if name == "random":
        return RecurrentModel(INPUT_SIZE, 64, MODES, groups=4, topology="random",
                              graph_seed=6_500_000 + seed)
    raise ValueError(name)


def tiny_overfit() -> dict[str, float | int]:
    """Engineering preflight on an independent 32-example stream."""
    batch = make_dynamic_filter_batch(32, 8, 9_000_001)
    model = build_arm("dense", 9601)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(600):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_on(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    score = model_metrics(model, batch)
    if score["pre_joint_kl_to_oracle"] >= 0.03 or score["final_joint_kl_to_oracle"] >= 0.03:
        raise RuntimeError(f"Tiny overfit failed: {score}")
    dense = resource_account(model, 10, BATCH_SIZE)
    random = resource_account(build_arm("random", 9601), 10, BATCH_SIZE)
    if not (dense["hidden_state_floats"] == random["hidden_state_floats"] == 64
            and dense["pre_readout_cache_floats_per_example"] == 4
            and dense["dense_forward_linear_flops_per_batch"] == random["dense_forward_linear_flops_per_batch"]
            and dense["effective_parameters"] > random["effective_parameters"]):
        raise RuntimeError("Resource sanity failed")
    return {"examples": 32, "steps": 600,
            "pre_joint_kl_to_oracle": float(score["pre_joint_kl_to_oracle"]),
            "final_joint_kl_to_oracle": float(score["final_joint_kl_to_oracle"]),
            "dense_effective_parameters": dense["effective_parameters"],
            "random_effective_parameters": random["effective_parameters"]}


def publish_report(output: Path, report: dict, started: float) -> dict:
    """Publish once via a same-filesystem staging file and exclusive hard link."""
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.{os.getpid()}.staging")
    if staging.exists():
        raise FileExistsError(staging)

    def safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return f"NONFINITE:{value}"
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [safe(item) for item in value]
        return value

    try:
        for attempt in range(3):
            report["elapsed_seconds"] = time.perf_counter() - started
            candidate = safe(report) if report.get("status") in {"ENGINEERING_ABORT", "INCOMPLETE_BUDGET"} else report
            payload = json.dumps(candidate, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
            with staging.open("xb" if attempt == 0 else "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            measured = time.perf_counter() - started
            if measured > WALL_LIMIT_SECONDS and report.get("status") != "INCOMPLETE_BUDGET":
                report["status_before_budget_limit"] = report.get("status")
                report["status"] = "INCOMPLETE_BUDGET"
                report["qualified_for_design_review"] = False
                continue
            report["staging_fsync_checked_seconds"] = measured
            if measured > WALL_LIMIT_SECONDS or attempt == 2:
                break
            # Refresh the elapsed field after the first fsync; the next fsync is checked too.
            if attempt == 0:
                continue
            break
        os.link(staging, output)
        report["hardlink_publication_seconds"] = time.perf_counter() - started - report["staging_fsync_checked_seconds"]
    finally:
        staging.unlink(missing_ok=True)
    return report


def publish_timing_sidecar(output: Path, report: dict) -> None:
    path = output.with_name("CAL-0006_v3_publication.json")
    value = {"report_sha256": digest(output.read_bytes()),
             "staging_fsync_checked_seconds": report["staging_fsync_checked_seconds"],
             "hardlink_publication_seconds": report["hardlink_publication_seconds"],
             "budget_excludes_hardlink_metadata": True}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_publication(output: Path) -> dict:
    """Require the publication sidecar before interpreting any report status."""
    sidecar = output.with_name("CAL-0006_v3_publication.json")
    report = json.loads(output.read_text(encoding="utf-8"))
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata["report_sha256"] != digest(output.read_bytes()) or not metadata["budget_excludes_hardlink_metadata"]:
        raise ValueError("CAL-0006 report/publication sidecar mismatch")
    for field in ("staging_fsync_checked_seconds", "hardlink_publication_seconds"):
        if not math.isfinite(metadata[field]) or metadata[field] < 0:
            raise ValueError(f"Invalid CAL-0006 publication timing: {field}")
    return report


def finish_report(output: Path, report: dict, started: float) -> dict:
    result = publish_report(output, report, started)
    publish_timing_sidecar(output, result)
    verify_publication(output)
    return result


def publish_checkpoint(model: RecurrentModel, checkpoint: Path) -> str:
    """Publish a finished checkpoint without replacing a prior target."""
    staging = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.staging")
    if staging.exists():
        raise FileExistsError(staging)
    try:
        torch.save(model.state_dict(), staging)
        os.link(staging, checkpoint)
    finally:
        staging.unlink(missing_ok=True)
    return digest(checkpoint.read_bytes())


def _run_locked(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if Path(sys.executable).resolve() != Path("/private/tmp/neurotopo-venv/bin/python").resolve():
        raise RuntimeError("Formal CAL-0006 requires the frozen external interpreter")
    if git_output("status", "--porcelain"):
        raise RuntimeError("Formal CAL-0006 requires a clean committed checkout")
    checkpoint_dir = output.parent / "checkpoints" / "CAL-0006_v3"
    if checkpoint_dir.exists():
        raise FileExistsError(checkpoint_dir)
    configure(601, 1)
    started = time.perf_counter()
    report: dict = {"kind": "BENCHMARK_QUALIFICATION_NOT_ARCHITECTURE_EVIDENCE",
                    "protocol": "research/design/CAL-0006_dynamic_counterfactual_qualification_v3.md",
                    "output": str(output), "completion_requires_publication_sidecar": True,
                    "stage": "initializing", "model_results": []}
    current_pair: dict | None = None

    def check_budget() -> None:
        if time.perf_counter() - started > WALL_LIMIT_SECONDS:
            raise TimeoutError("CAL-0006 total wall budget exceeded")

    try:
        report["git_commit"] = git_output("rev-parse", "HEAD")
        report["script_sha256"] = digest(Path(__file__).read_bytes())
        report["protocol_sha256"] = digest(Path(report["protocol"]).read_bytes())
        report["source_sha256"] = source_fingerprints()
        report["source_checkout"] = str(SOURCE_ROOT)
        report["project_root"] = str(PROJECT_ROOT)
        report["import_paths"] = {name: str(Path(sys.modules[name].__file__).resolve()) for name in
                                  ("neurotopo.dynamic_filter_task", "neurotopo.models",
                                   "neurotopo.resources", "neurotopo.run")}
        report["versions"] = {"torch": torch.__version__, "numpy": np.__version__,
                              "python_executable": sys.executable}
        report["seeds"] = {"model": list(SEEDS), "train_formula": "6000000+model_seed*10000+step",
                           "validation_formula": "6300000+model_seed", "evaluation": 6_400_001,
                           "random_graph_formula": "6500000+model_seed"}
        report["configuration"] = {"updates_per_arm": UPDATES, "batch_size": BATCH_SIZE,
                                   "train_observations": 8, "iid_observations": 8,
                                   "ood_observations": 12, "hidden": 64, "random_groups": 4,
                                   "optimizer": "Adam", "learning_rate": 0.003, "grad_clip": 1.0,
                                   "evaluation_examples": EVAL_EXAMPLES}
        report["budget"] = {"max_total_train_examples": len(SEEDS) * 2 * UPDATES * BATCH_SIZE,
                            "wall_limit_seconds": WALL_LIMIT_SECONDS}
        report["configuration_sha256"] = digest(json.dumps(
            {key: report[key] for key in ("configuration", "seeds", "budget")},
            sort_keys=True).encode("utf-8"))
        check_budget()
        report["stage"] = "evaluation_generation"
        iid, ood, paired_hash = paired_evaluation()
        check_budget()
        report["stage"] = "oracle_guard"
        oracle = {"iid8": oracle_metrics(iid), "ood12": oracle_metrics(ood)}
        report["evaluation"] = {"examples_per_length": EVAL_EXAMPLES, "prefix_sha256": paired_hash,
                                "iid_input_sha256": digest(iid.x.numpy().tobytes()),
                                "ood_input_sha256": digest(ood.x.numpy().tobytes())}
        report["oracle"] = oracle
        check_budget()
        if (oracle["iid8"]["final_true_map_accuracy"] < 0.55
                or oracle["ood12"]["pre_final_map_change_rate"] < 0.08):
            report.update(status="TASK_UNINFORMATIVE", qualified_for_design_review=False,
                          stage="finalizing_task_guard")
            check_budget()
            return finish_report(output, report, started)

        report["stage"] = "checkpoint_directory"
        checkpoint_dir.mkdir(parents=True)
        check_budget()
        for seed in SEEDS:
            report["stage"] = f"seed_{seed}_construction"
            validation = make_dynamic_filter_batch(1000, 8, 6_300_000 + seed)
            models = {name: build_arm(name, seed) for name in ("dense", "random")}
            optimizers = {name: torch.optim.Adam(model.parameters(), lr=0.003)
                          for name, model in models.items()}
            current_pair = {"seed": seed, "completed_updates": 0, "arms": {}}
            train_stream = hashlib.sha256()
            for name, model in models.items():
                current_pair["arms"][name] = {
                    "initial_parameters_sha256": model_digest(model),
                    "mask_sha256": digest(model.mask.numpy().tobytes()),
                    "graph_info": model.graph_info,
                    "completed_updates": 0, "train_seconds": 0.0, "train_tokens": 0,
                    "train_forward_linear_flops": 0, "train_examples": 0,
                    "step_losses": [], "monitors": [],
                    "iid_resources": resource_account(model, 10, BATCH_SIZE),
                    "ood_resources": resource_account(model, 14, BATCH_SIZE),
                }
            dense_resources = current_pair["arms"]["dense"]["iid_resources"]
            random_resources = current_pair["arms"]["random"]["iid_resources"]
            for field in ("allocated_parameters", "hidden_state_floats",
                          "pre_readout_cache_floats_per_example", "internal_steps_per_input",
                          "dense_forward_linear_flops_per_batch"):
                if dense_resources[field] != random_resources[field]:
                    raise ValueError(f"CAL-0006 screen resource mismatch: {field}")
            check_budget()
            for step in range(1, UPDATES + 1):
                report["stage"] = f"seed_{seed}_step_{step}_batch"
                batch = make_dynamic_filter_batch(BATCH_SIZE, 8, train_seed(seed, step))
                for tensor in (batch.x, batch.pre_joint, batch.final_joint):
                    train_stream.update(tensor.numpy().tobytes())
                check_budget()
                for name, model in models.items():
                    report["stage"] = f"seed_{seed}_step_{step}_{name}"
                    arm = current_pair["arms"][name]
                    then = time.perf_counter()
                    model.train()
                    optimizers[name].zero_grad(set_to_none=True)
                    loss = loss_on(model, batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss seed={seed} arm={name} step={step}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizers[name].step()
                    arm["train_seconds"] += time.perf_counter() - then
                    arm["train_examples"] += len(batch.x)
                    arm["train_tokens"] += len(batch.x) * batch.x.shape[1]
                    arm["train_forward_linear_flops"] += resource_account(
                        model, batch.x.shape[1], len(batch.x))["dense_forward_linear_flops_per_batch"]
                    arm["completed_updates"] = step
                    arm["step_losses"].append(float(loss.item()))
                    check_budget()
                    if step in (300, 1200):
                        report["stage"] = f"seed_{seed}_step_{step}_{name}_validation"
                        arm["monitors"].append({"step": step,
                                                "validation": model_metrics(model, validation),
                                                "train_batch_loss": float(loss.item())})
                        check_budget()
                current_pair["completed_updates"] = step
                if step % 300 == 0:
                    print(json.dumps({"seed": seed, "step": step,
                                      "validation_final_kl": {
                                          name: current_pair["arms"][name]["monitors"][-1]["validation"]["final_joint_kl_to_oracle"]
                                          for name in models if current_pair["arms"][name]["monitors"]
                                          and current_pair["arms"][name]["monitors"][-1]["step"] == step}}), flush=True)
                check_budget()
            current_pair["train_stream_sha256"] = train_stream.hexdigest()
            dense_arm, random_arm = current_pair["arms"]["dense"], current_pair["arms"]["random"]
            for field in ("train_examples", "train_tokens", "train_forward_linear_flops"):
                if dense_arm[field] != random_arm[field]:
                    raise ValueError(f"CAL-0006 paired training resource mismatch: {field}")
            for name, model in models.items():
                report["stage"] = f"seed_{seed}_{name}_checkpoint"
                arm = current_pair["arms"][name]
                checkpoint = checkpoint_dir / f"seed-{seed}-{name}.pt"
                arm["checkpoint_planned"] = str(checkpoint)
                arm["checkpoint_sha256"] = publish_checkpoint(model, checkpoint)
                arm["checkpoint"] = str(checkpoint)
                arm["final_parameters_sha256"] = model_digest(model)
                check_budget()
                for label, batch in (("iid8", iid), ("ood12", ood)):
                    report["stage"] = f"seed_{seed}_{name}_{label}_evaluation"
                    then = time.perf_counter()
                    arm[label] = model_metrics(model, batch)
                    arm[f"{label}_model_forward_seconds"] = arm[label]["model_forward_seconds"]
                    arm[f"{label}_metric_eval_seconds"] = time.perf_counter() - then
                    check_budget()
            report["model_results"].append(current_pair)
            current_pair = None
        report["stage"] = "decision"
        iid_ok = all(row["arms"][name]["iid8"]["final_joint_kl_to_oracle"] < 0.10
                     for row in report["model_results"] for name in ("dense", "random"))
        ood_values = [row["arms"][name]["ood12"]["final_joint_kl_to_oracle"]
                      for row in report["model_results"] for name in ("dense", "random")]
        if not iid_ok:
            status = "LEARNING_FLOOR"
        elif any(value < 0.08 for value in ood_values):
            status = "CEILING"
        elif any(value > 0.50 for value in ood_values):
            status = "OOD_FLOOR"
        else:
            status = "SCREEN_PASSED_FOR_DESIGN_REVIEW"
        report.update(status=status, qualified_for_design_review=status == "SCREEN_PASSED_FOR_DESIGN_REVIEW",
                      stage="finalizing")
        check_budget()
        return finish_report(output, report, started)
    except TimeoutError as error:
        report.update(status="INCOMPLETE_BUDGET", qualified_for_design_review=False,
                      error=str(error), incomplete_pair=current_pair,
                      published_checkpoints=sorted(str(path) for path in checkpoint_dir.glob("*.pt")))
    except Exception as error:
        if output.exists():
            raise
        report.update(status="ENGINEERING_ABORT", qualified_for_design_review=False,
                      error=f"{type(error).__name__}: {error}", incomplete_pair=current_pair,
                      published_checkpoints=sorted(str(path) for path in checkpoint_dir.glob("*.pt")))
        finish_report(output, report, started)
        raise
    return finish_report(output, report, started)


def run() -> dict:
    if Path.cwd().resolve() != SOURCE_ROOT:
        raise RuntimeError("Run CAL-0006 from the source checkout root")
    if Path(git_output("rev-parse", "--show-toplevel")).resolve() != SOURCE_ROOT:
        raise RuntimeError("CAL-0006 source and Git checkout roots differ")
    if git_output("status", "--porcelain"):
        raise RuntimeError("Formal CAL-0006 requires a clean committed checkout")
    for name in ("neurotopo.dynamic_filter_task", "neurotopo.models",
                 "neurotopo.resources", "neurotopo.run"):
        if not Path(sys.modules[name].__file__).resolve().is_relative_to(SOURCE_ROOT):
            raise RuntimeError(f"CAL-0006 import outside the clean source checkout: {name}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lock = PROJECT_ROOT / "artifacts/auto_research/live/CAL-0006_v3.run.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "source_checkout": str(SOURCE_ROOT),
                   "source_commit": git_output("rev-parse", "HEAD")}, stream)
        stream.write("\n")
    try:
        return _run_locked(OUTPUT)
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(tiny_overfit(), sort_keys=True), flush=True)
        return
    result = run()
    print(json.dumps({"status": result["status"], "output": str(OUTPUT),
                      "qualified_for_design_review": result["qualified_for_design_review"]}), flush=True)


if __name__ == "__main__":
    main()
