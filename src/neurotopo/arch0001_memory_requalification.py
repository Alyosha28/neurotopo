"""EXP0031 v1: corrected-endpoint re-qualification of the DELAY-d task family.

Same task, same training, same seeds as EXP0030 (module src/neurotopo/arch0001_memory_task.py is imported
unchanged); this module adds position-restricted evaluation, an expanded memoryless reference set, an
explicit supervision tie-break and a byte-level reproduction check against the EXP0030 archive.
See experiments/EXP-0031/v1.md for the frozen decision table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .arch0001_memory_task import (evaluate_memory, evaluation_stream, fit_memory, generate_memory,
                                   validate_config as validate_parent_training_config)
from .frozen_binary_probe import save_json
from .order_observation import sha
from .run import git_output, source_fingerprints


def meaningful_slice(d: int, length: int) -> slice:
    """Readout positions r in [d+1, T]: targets are real commands there."""
    return slice(d + 1, length + 1)


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def meaningful_metrics(parts: dict[str, torch.Tensor], data: dict[str, np.ndarray], d: int) -> dict[str, Any]:
    logits = parts["logits"].detach().numpy().astype(np.float64)
    target = data["targets"] >= .5
    length = data["commands"].shape[1]
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite logits")
    correct = (logits >= 0) == target
    sel = meaningful_slice(d, length)
    branch = parts["branch"].detach().numpy().astype(np.float64)
    branch_ok = (branch >= 0) == (data["branch_targets"] >= .5)
    return dict(meaningful_accuracy=float(correct[:, sel].mean()),
                accuracy_all=float(correct.mean()),
                final_accuracy=float(correct[:, -1].mean()),
                tail_mean=float(correct[:, 6:].mean()),
                per_position=[float(x) for x in correct[:, sel].mean(axis=0)],
                branch_joint_accuracy=float(branch_ok.all(axis=2).mean()),
                logits_sha256=_hash(parts["logits"].detach().numpy().tobytes()),
                branch_sha256=_hash(parts["branch"].detach().numpy().tobytes()),
                examples=int(len(logits)), steps=int(length))


def reference_predictions(data: dict[str, np.ndarray], d: int, kind: str,
                          calibration: dict[str, np.ndarray] | None = None) -> np.ndarray:
    length = data["commands"].shape[1]
    n = len(data["b0"])
    out = np.zeros((n, length + 1), bool)
    if kind == "constant_zero":
        return out
    if kind == "constant_one":
        return ~out
    if kind == "pass_through":
        out[:, 1:] = data["commands"] == 1
        return out
    if kind == "position_majority":
        assert calibration is not None
        target = calibration["targets"] >= .5
        majority = target.mean(axis=0) >= .5
        return np.repeat(majority[None, :], n, axis=0)
    raise ValueError(kind)


def reference_scores(data: dict[str, np.ndarray], d: int,
                     calibration: dict[str, np.ndarray] | None = None) -> dict[str, float]:
    target = data["targets"] >= .5
    sel = meaningful_slice(d, data["commands"].shape[1])
    scores = {}
    for kind in ("constant_zero", "constant_one", "pass_through", "position_majority"):
        pred = reference_predictions(data, d, kind, calibration)
        scores[kind] = float((pred[:, sel] == target[:, sel]).mean())
    return scores


def oracle_score(data: dict[str, np.ndarray], d: int) -> float:
    """Explicit shift register over the observed command buffer, scored on meaningful positions."""
    u = data["commands"]
    n, length = u.shape
    pred = np.zeros((n, length + 1), bool)
    for t in range(d, length):
        pred[:, t + 1] = u[:, t - d] == 1
    target = data["targets"] >= .5
    sel = meaningful_slice(d, length)
    return float((pred[:, sel] == target[:, sel]).mean())


def calibration_stream(cfg: dict[str, Any], d: int) -> dict[str, np.ndarray]:
    length = cfg["evaluation"]["primary_length"]
    return generate_memory(cfg["calibration"]["examples"], length,
                           [cfg["calibration"]["namespace"], d, length, 0], d,
                           cfg["training"]["common_amplitude"], cfg["training"]["leak"], cfg["training"]["sigma"])


def run_condition(cfg: dict[str, Any], item: dict[str, Any], seed: int, out: Path) -> dict[str, Any]:
    d, sup = item["d"], item["sup"]
    length = cfg["evaluation"]["primary_length"]
    data = evaluation_stream(cfg, seed, d, length)
    result = fit_memory(seed, d, sup, cfg, monitor=data)
    record = result["record"]
    record["primary"] = meaningful_metrics(evaluate_memory(result["model"], data), data, d)
    for extra in cfg["evaluation"]["secondary_lengths"]:
        stream = evaluation_stream(cfg, seed, d, extra)
        record[f"secondary_{extra}"] = meaningful_metrics(evaluate_memory(result["model"], stream), stream, d)
    record["references"] = reference_scores(data, d, calibration_stream(cfg, d))
    record["oracle"] = oracle_score(data, d)
    parent = Path(cfg["parent_artifacts"]) / "runs" / f"{item['id']}_seed{seed}.json"
    record["parent_record"] = str(parent)
    record["parent_reproduced"] = bool(parent.exists()
                                       and json.loads(parent.read_text())["final_hash"] == record["final_hash"]
                                       and json.loads(parent.read_text())["training_data_sha256"] == record["training_data_sha256"])
    save_json(out / "runs" / f"{item['id']}_seed{seed}.json", record)
    return record


def summarize_scan(cfg: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    band = cfg["band"]
    by_d: dict[int, Any] = {}
    for d in sorted({item["d"] for item in cfg["menu"]}):
        variants = {}
        for sup in ("PLAIN", "BRANCH"):
            rows = [r for r in records if r["d"] == d and r["sup"] == sup]
            if not rows:
                continue
            acc = [r["primary"]["meaningful_accuracy"] for r in rows]
            variants[sup] = dict(seeds=[r["seed"] for r in rows], accuracy=acc, mean=float(np.mean(acc)),
                                 min=float(np.min(acc)), max=float(np.max(acc)),
                                 reproduced=all(r["parent_reproduced"] for r in rows), rows=rows)
        if not variants:
            continue
        if len(variants) == 2 and abs(variants["PLAIN"]["mean"] - variants["BRANCH"]["mean"]) <= 1e-9:
            best_sup = "PLAIN"
        else:
            best_sup = max(variants, key=lambda s: variants[s]["mean"])
        best = variants[best_sup]
        refs = [r["references"] for r in best["rows"]]
        memoryless_best = {kind: max(r[kind] for r in refs) for kind in refs[0]}
        memoryless_max = max(memoryless_best.values())
        oracle_min = min(r["oracle"] for r in best["rows"])
        learn = (best["mean"] >= max(band["learn_mean_min"], memoryless_max + band["memoryless_margin_mean"])
                 and best["min"] >= max(band["learn_seed_min"], memoryless_max + band["memoryless_margin_seed"]))
        nonceiling = best["mean"] <= band["nonceiling_mean_max"] and best["max"] <= band["nonceiling_seed_max"]
        oracle_ok = oracle_min >= band["oracle_min"]
        by_d[d] = dict(d=d, best_sup=best_sup, variants=variants, learn=bool(learn), nonceiling=bool(nonceiling),
                       memoryless_best=memoryless_best, memoryless_max=memoryless_max, oracle_min=oracle_min,
                       reproduced=best["reproduced"],
                       qualified=bool(learn and nonceiling and oracle_ok and best["reproduced"]))
    qualified = None
    for d in sorted(by_d):
        if by_d[d]["qualified"]:
            qualified = dict(d=d, sup=by_d[d]["best_sup"])
            break
    return dict(by_d=by_d, qualified=qualified, band=band, tie_break=cfg["tie_break"])


def run_scan(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    started = time.monotonic()
    records = []
    for item in cfg["menu"]:
        for seed in cfg["seeds"]["scan"]:
            if time.monotonic() - started > cfg["resource"]["wall_seconds_cap"]:
                raise TimeoutError("Shared EXP-0031 wall budget")
            records.append(run_condition(cfg, item, seed, out))
    summary = summarize_scan(cfg, records)
    save_json(out / "scan_summary.json", summary)
    return summary


def run_confirm(cfg: dict[str, Any], out: Path, qualified: dict[str, Any]) -> dict[str, Any]:
    item = dict(id=f"D{qualified['d']}_{qualified['sup']}", d=qualified["d"], sup=qualified["sup"])
    records = [run_condition(cfg, item, seed, out) for seed in cfg["seeds"]["confirm"]]
    band = cfg["band"]
    acc = [r["primary"]["meaningful_accuracy"] for r in records]
    refs = [r["references"] for r in records]
    memoryless_best = {kind: max(r[kind] for r in refs) for kind in refs[0]}
    memoryless_max = max(memoryless_best.values())
    oracle_min = min(r["oracle"] for r in records)
    learn = (float(np.mean(acc)) >= max(band["learn_mean_min"], memoryless_max + band["memoryless_margin_mean"])
             and float(np.min(acc)) >= max(band["learn_seed_min"], memoryless_max + band["memoryless_margin_seed"]))
    nonceiling = float(np.mean(acc)) <= band["nonceiling_mean_max"] and float(np.max(acc)) <= band["nonceiling_seed_max"]
    confirm = dict(item=item, seeds=cfg["seeds"]["confirm"], accuracy=acc, mean=float(np.mean(acc)),
                   min=float(np.min(acc)), max=float(np.max(acc)), learn=bool(learn), nonceiling=bool(nonceiling),
                   memoryless_best=memoryless_best, memoryless_max=memoryless_max, oracle_min=oracle_min,
                   reproduced=all(r["parent_reproduced"] for r in records), rows=records)
    confirm["reproduced"] = bool(confirm["reproduced"] and learn and nonceiling and oracle_min >= band["oracle_min"])
    save_json(out / "confirm_summary.json", confirm)
    return confirm


def validate_config(c: dict[str, Any]) -> None:
    assert c["experiment_id"] == "EXP-0031" and c["version"] == "v1"
    assert [m["id"] for m in c["menu"]] == ["D1_PLAIN", "D1_BRANCH", "D2_PLAIN", "D2_BRANCH", "D3_PLAIN", "D3_BRANCH"]
    assert c["seeds"] == dict(scan=[29101, 29102], confirm=[29111, 29112, 29113])
    assert c["calibration"] == dict(namespace=294001, examples=4096)
    assert c["band"] == dict(learn_mean_min=.65, learn_seed_min=.58, memoryless_margin_mean=.05,
                             memoryless_margin_seed=.02, nonceiling_mean_max=.95, nonceiling_seed_max=.97,
                             oracle_min=.99)
    assert c["calibration"] == dict(namespace=294001, examples=4096)
    assert c["resource"] == dict(max_updates=30000, wall_seconds_cap=1800, replay_wall_seconds_cap=600)
    parent = json.loads(Path(c["parent_config"]).read_text())
    validate_parent_training_config(parent)
    for key in ("control", "menu", "seeds", "training", "streams", "evaluation"):
        assert c[key] == parent[key], f"EXP-0031 must keep the parent {key} unchanged"


def manifest(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    status = git_output("status", "--porcelain")
    return dict(experiment_id="EXP-0031", version="v1", gate="A2_TASK_QUALIFICATION_CORRECTED", config=cfg,
                git_commit=git_output("rev-parse", "HEAD"), git_dirty=bool(status.strip()),
                git_status=status, source_sha256=source_fingerprints(),
                frozen_hashes={str(p): sha(p) for p in (config_path, Path(cfg["protocol"]),
                                                        Path(cfg["training_module"]), Path(cfg["parent_config"]),
                                                        Path("uv.lock"), Path("artifacts/validation/suite.json"),
                                                        Path("src/neurotopo/arch0001_memory_requalification.py"),
                                                        Path("tests/test_arch0001_memory_requalification.py"))},
                python=sys.version, torch=torch.__version__, numpy=np.__version__,
                platform=platform.platform(),
                scope="Corrected-endpoint re-qualification only; no candidate training, no A2 effect claim.")


def run(config_path: Path, output: Path, phases: str = "all") -> dict[str, Any]:
    cfg = json.loads(config_path.read_text())
    validate_config(cfg)
    payload = json.loads(Path("artifacts/validation/suite.json").read_text())
    if not payload["pass"] or payload["source_sha256"] != source_fingerprints():
        raise RuntimeError("Validation suite does not match current sources; run scripts/verify.py")
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "manifest.json").exists():
        save_json(output / "manifest.json", manifest(cfg, config_path))
    started = time.monotonic()
    todo = ["scan", "confirm", "analyze"] if phases == "all" else phases.split(",")
    scan = json.loads((output / "scan_summary.json").read_text()) if (output / "scan_summary.json").exists() else None
    confirm = json.loads((output / "confirm_summary.json").read_text()) if (output / "confirm_summary.json").exists() else None
    for phase in todo:
        if time.monotonic() - started > cfg["resource"]["wall_seconds_cap"]:
            raise TimeoutError("Shared EXP-0031 wall budget")
        if phase == "scan":
            scan = run_scan(cfg, output)
        elif phase == "confirm":
            scan = scan or json.loads((output / "scan_summary.json").read_text())
            if scan["qualified"] is not None:
                confirm = run_confirm(cfg, output, scan["qualified"])
        elif phase == "analyze":
            scan = scan or json.loads((output / "scan_summary.json").read_text())
            if confirm is None and (output / "confirm_summary.json").exists():
                confirm = json.loads((output / "confirm_summary.json").read_text())
            reproduced = all(v["reproduced"] for v in scan["by_d"].values())
            if not reproduced:
                decision, substatus = "INCONCLUSIVE", "ENGINEERING_OR_REPRODUCTION_FAILURE"
            elif scan["qualified"] is None:
                decision, substatus = "INCONCLUSIVE", "NO_QUALIFIED_MEMORY_CONDITION"
            elif confirm is None:
                decision, substatus = "INCONCLUSIVE", "CONFIRMATION_MISSING"
            elif confirm["reproduced"]:
                decision, substatus = "SUPPORTED", "A2_TASK_QUALIFIED_CORRECTED"
            else:
                decision, substatus = "INCONCLUSIVE", "QUALIFICATION_NOT_REPRODUCED"
            analysis = dict(experiment_id="EXP-0031", version="v1", gate="A2_TASK_QUALIFICATION_CORRECTED",
                            decision=decision, substatus=substatus, scan=scan, confirm=confirm,
                            seconds=time.monotonic() - started,
                            scope=("Corrected-endpoint re-qualification only; no candidate run, no A2 effect, "
                                   "no A1 or old Gate 1 credit."))
            save_json(output / "analysis.json", analysis)
            return analysis
        else:
            raise ValueError(phase)
    return dict(phases=todo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phases", default="all")
    args = parser.parse_args()
    outcome = run(args.config, args.output, args.phases)
    print(json.dumps({k: v for k, v in outcome.items() if k not in ("scan", "confirm")}, ensure_ascii=False, default=str))
