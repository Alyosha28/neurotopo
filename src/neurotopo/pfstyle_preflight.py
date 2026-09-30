"""EXP-0011 engineering-only resource preflight on isolated synthetic tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .pfstyle_models import ARM_NAMES, build_model
from .pfstyle_resources import resource_account
from .run import configure, git_output, source_fingerprints


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_resource(candidate: dict[str, Any], control: dict[str, Any],
                      timing_ratios: list[float], config: dict[str, Any]) -> str:
    """Apply the registered static and three-block timing rules without re-measuring."""
    limits = config["resource"]
    if candidate["total_dynamic_state_floats"] != control["total_dynamic_state_floats"]:
        return "RESOURCE_MISMATCH"
    for field, window in (("allocated_parameters", "parameter_allocated_control_ratio_window"),
                          ("effective_parameters", "parameter_effective_control_ratio_window"),
                          ("dense_forward_linear_macs_per_example", "forward_linear_mac_control_ratio_window")):
        ratio = control[field] / candidate[field]
        low, high = limits[window]
        if not low <= ratio <= high:
            return "RESOURCE_MISMATCH"
    if len(timing_ratios) != limits["timing_blocks"] * 3 or not np.isfinite(timing_ratios).all():
        return "RESOURCE_UNRESOLVED"
    if any(ratio > limits["p50_candidate_control_fail_above"] for ratio in timing_ratios):
        return "RESOURCE_MISMATCH"
    if any(ratio > limits["p50_candidate_control_pass_max"] for ratio in timing_ratios):
        return "RESOURCE_UNRESOLVED"
    return "PASS"


def _soft_loss(model: torch.nn.Module, x: Tensor, pre_target: Tensor,
               final_target: Tensor) -> Tensor:
    pre, final = model(x, 8)
    return 0.5 * (-(pre_target * F.log_softmax(pre, -1)).sum(-1).mean()
                  - (final_target * F.log_softmax(final, -1)).sum(-1).mean())


def _time_training(model: torch.nn.Module, x: Tensor, targets: tuple[Tensor, Tensor],
                   config: dict[str, Any]) -> tuple[list[float], torch.optim.Optimizer]:
    resource = config["resource"]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    times = []
    for trial in range(resource["warmup_repetitions_per_block"] +
                       resource["timing_repetitions_per_block"]):
        begin = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _soft_loss(model, x, *targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite engineering-only loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"],
                                       error_if_nonfinite=True)
        optimizer.step()
        elapsed = time.perf_counter() - begin
        if trial >= resource["warmup_repetitions_per_block"]:
            times.append(elapsed)
    return times, optimizer


@torch.no_grad()
def _time_inference(model: torch.nn.Module, x: Tensor,
                    config: dict[str, Any]) -> list[float]:
    resource = config["resource"]
    model.eval()
    times = []
    for trial in range(resource["warmup_repetitions_per_block"] +
                       resource["timing_repetitions_per_block"]):
        begin = time.perf_counter()
        model(x, 8)
        elapsed = time.perf_counter() - begin
        if trial >= resource["warmup_repetitions_per_block"]:
            times.append(elapsed)
    return times


def _operator_profile(model: torch.nn.Module, x32: Tensor, targets: tuple[Tensor, Tensor],
                      config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    from torch.profiler import ProfilerActivity, profile

    def rows(session: Any) -> list[dict[str, Any]]:
        return [{"operator": event.key, "calls": event.count,
                 "input_shapes": event.input_shapes,
                 "self_cpu_time_us": event.self_cpu_time_total,
                 "profiler_flops_if_available": event.flops}
                for event in sorted(session.key_averages(group_by_input_shape=True),
                                    key=lambda item: (item.key, str(item.input_shapes)))]

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as session:
        optimizer.zero_grad(set_to_none=True)
        loss = _soft_loss(model, x32[:, :16], *targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"],
                                       error_if_nonfinite=True)
        optimizer.step()
    result = {"training16": rows(session)}
    model.eval()
    for length in (16, 32):
        with torch.no_grad(), profile(activities=[ProfilerActivity.CPU], record_shapes=True,
                                      with_flops=True) as session:
            model(x32[:, :length], 8)
        result[f"inference{length}"] = rows(session)
    return result


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    receipt_path = output.with_name(output.stem + "_receipt.json")
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    config = json.loads(config_path.read_text())
    if (config["experiment_id"], config["version"], config["workflow"]) != (
            "EXP-0011", "v6", "pfstyle_field"):
        raise ValueError("wrong registered preflight config")
    if git_output("status", "--porcelain"):
        raise RuntimeError("EXP-0011 engineering preflight requires a clean committed checkout")
    if _sha(Path(config["matrix_file"])) != config["matrix_json_sha256"]:
        raise RuntimeError("frozen matrix digest changed")
    started = time.monotonic()
    configure(config["resource"]["engineering_seed"], config["training"]["threads"])
    rng = torch.Generator(device="cpu").manual_seed(config["resource"]["engineering_seed"])
    x32 = torch.randn(128, 32, 2, generator=rng)
    targets = (F.softmax(torch.randn(128, 4, generator=rng), -1),
               F.softmax(torch.randn(128, 4, generator=rng), -1))
    account = {}
    graphs = {}
    timing: dict[str, list[dict[str, Any]]] = {name: [] for name in ARM_NAMES}
    updates = 0
    timed_out = False
    for block in range(config["resource"]["timing_blocks"]):
        offset = block % len(ARM_NAMES)
        ordered = ARM_NAMES[offset:] + ARM_NAMES[:offset]
        for name in ordered:
            model = build_model(name, config["resource"]["engineering_seed"] + block,
                                config["training"]["graph_seed_base"] +
                                config["resource"]["engineering_seed"] + block)
            if block == 0:
                account[name] = {str(length): resource_account(model, length, 128)
                                 for length in (16, 32)}
                graphs[name] = model.graph_info
            training_times, _ = _time_training(model, x32[:, :16], targets, config)
            updates += config["resource"]["warmup_repetitions_per_block"] + config["resource"]["timing_repetitions_per_block"]
            if updates > config["resource"]["engineering_update_cap"]:
                raise RuntimeError("engineering update budget exceeded")
            inference16 = _time_inference(model, x32[:, :16], config)
            inference32 = _time_inference(model, x32, config)
            timing[name].append({"block": block, "training16_seconds": training_times,
                                 "inference16_seconds": inference16,
                                 "inference32_seconds": inference32})
            if time.monotonic() - started > config["resource"]["preflight_wall_seconds_cap"]:
                timed_out = True
                break
        if timed_out:
            break
    profiles = {}
    if not timed_out:
        for name in ARM_NAMES:
            model = build_model(name, config["resource"]["engineering_seed"] + 10,
                                config["training"]["graph_seed_base"] +
                                config["resource"]["engineering_seed"] + 10)
            profiles[name] = _operator_profile(model, x32, targets, config)
            updates += 1
            if time.monotonic() - started > config["resource"]["preflight_wall_seconds_cap"]:
                timed_out = True
                break
    ratios = []
    if not timed_out:
        for block in range(config["resource"]["timing_blocks"]):
            candidate, control = timing["K4_score"][block], timing["S68_sparse"][block]
            for path in ("training16_seconds", "inference16_seconds", "inference32_seconds"):
                ratios.append(float(np.median(candidate[path]) / np.median(control[path])))
        length_statuses = {str(length): classify_resource(account["K4_score"][str(length)],
                                                          account["S68_sparse"][str(length)],
                                                          ratios, config)
                           for length in (16, 32)}
        resource_status = ("RESOURCE_MISMATCH" if "RESOURCE_MISMATCH" in length_statuses.values()
                           else "RESOURCE_UNRESOLVED" if "RESOURCE_UNRESOLVED" in length_statuses.values()
                           else "PASS")
    else:
        length_statuses = {}
        resource_status = "INCOMPLETE_BUDGET"
    passed = updates <= config["resource"]["engineering_update_cap"]
    report = {
        "experiment_id": "EXP-0011", "version": "v6", "engineering_valid": passed,
        "resource_status": resource_status,
        "config_sha256": _sha(config_path),
        "matrix_json_sha256": _sha(Path(config["matrix_file"])),
        "source_sha256": source_fingerprints(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "lock_sha256": _sha(Path("uv.lock")),
        "engineering_seed": config["resource"]["engineering_seed"],
        "engineering_updates": updates,
        "preflight_seconds_before_report_fsync": time.monotonic() - started,
        "resource_accounts": account,
        "graphs": graphs,
        "timing": timing,
        "candidate_control_p50_ratios": ratios,
        "resource_status_by_length": length_statuses,
        "operator_profiles": profiles,
        "scope": "Synthetic tensors only; no task/qualification/formal seed or task generator accessed",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    after_report_fsync = time.monotonic() - started
    receipt = {
        "id": "EXP-0011_PREFLIGHT_RECEIPT",
        "preflight_report_sha256": _sha(output),
        "seconds_after_report_fsync": after_report_fsync,
        "budget_status": "PASS" if after_report_fsync <= config["resource"]["preflight_wall_seconds_cap"]
                         else "INCOMPLETE_BUDGET",
        "publication_metadata_fsync_excluded": True,
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"engineering_valid": passed, "resource_status": resource_status,
                      "engineering_updates": updates, "output": str(output),
                      "receipt": str(receipt_path), "budget_status": receipt["budget_status"]}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation/exp0011_preflight.json"))
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
