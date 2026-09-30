"""EXP0032 v1: control-menu and budget sweep on the corrected DELAY-d endpoint.

Imports the frozen EXP0030 training and EXP0031 evaluation without modifying them. See
experiments/EXP-0032/v1.md for the frozen staged menu, selection rule and decision table.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .arch0001_core import op_estimate, param_count
from .arch0001_memory_requalification import (calibration_stream, meaningful_metrics, oracle_score, reference_scores)
from .arch0001_memory_task import evaluate_memory, evaluation_stream, fit_memory
from .frozen_binary_probe import save_json
from .run import git_output, source_fingerprints

CAND_INFERENCE_OPS = op_estimate("CAND")["inference"]


def training_cfg(cfg: dict[str, Any], budget: int, curriculum: list[list[int]],
                 width: int | None = None, blocks: int | None = None) -> dict[str, Any]:
    """Copy the config, set the stage budget/curriculum and inject the swept control width/blocks.

    fit_memory() builds the model from cfg['control'], so the sweep must override it here; forgetting this
    makes every menu item train the same model (EXP0032 v1 defect, see the protocol addendum).
    """
    out = dict(cfg)
    out["training"] = dict(cfg["training"], updates=budget, curriculum=curriculum)
    out["control"] = dict(cfg["control"],
                          width=cfg["control"]["width"] if width is None else width,
                          blocks=cfg["control"]["blocks"] if blocks is None else blocks)
    return out


def run_one(cfg: dict[str, Any], width: int, blocks: int, d: int, seed: int, budget: int,
            curriculum: list[list[int]], out: Path, stage: str) -> dict[str, Any]:
    path = out / stage / f"w{width}_b{blocks}_d{d}_seed{seed}_u{budget}.json"
    if path.exists():
        return json.loads(path.read_text())
    c = training_cfg(cfg, budget, curriculum, width=width, blocks=blocks)
    result = fit_memory(seed, d, cfg["supervision"], c)
    data = evaluation_stream(c, seed, d, cfg["evaluation"]["primary_length"])
    metrics = meaningful_metrics(evaluate_memory(result["model"], data), data, d)
    record = dict(stage=stage, width=width, blocks=blocks, d=d, seed=seed, budget=budget,
                  primary=metrics, references=reference_scores(data, d, calibration_stream(c, d)),
                  oracle=oracle_score(data, d),
                  parameters=param_count("INV_COMP", width, blocks),
                  inference_ops=op_estimate("INV_COMP", width, blocks)["inference"],
                  raw_inference_ops=op_estimate("INV_COMP", width, blocks)["inference"],
                  seconds=result["record"]["seconds"], final_hash=result["record"]["final_hash"],
                  training_data_sha256=result["record"]["training_data_sha256"])
    save_json(path, record)
    return record


def stage_a(cfg: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    records = []
    stage = cfg["stage_a"]
    for d in cfg["d_values"]:
        for width, blocks in cfg["menu"]:
            records.append(run_one(cfg, width, blocks, d, stage["seeds"][0], stage["budget"],
                                   stage["curriculum"], out, "stage_a"))
            print(json.dumps(dict(stage="A", d=d, w=width, b=blocks,
                                  meaningful=round(records[-1]["primary"]["meaningful_accuracy"], 4)), ensure_ascii=False), flush=True)
    return records


def in_band(cfg: dict[str, Any], record: dict[str, Any]) -> bool:
    band = cfg["band"]
    acc = record["primary"]["meaningful_accuracy"]
    return band["learn_mean_min"] <= acc <= band["nonceiling_mean_max"]


def select(cfg: dict[str, Any], records: list[dict[str, Any]]) -> tuple[int, dict[str, Any]] | None:
    """First d in order with a non-empty S_d; within it, the compute-closest config (ties: fewer params)."""
    band = cfg["band"]
    for d in cfg["d_values"]:
        candidates = [r for r in records if r["d"] == d and in_band(cfg, r)]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: (abs(np.log(r["inference_ops"] / CAND_INFERENCE_OPS)), r["parameters"]))
        return d, best
    return None


def stage_b(cfg: dict[str, Any], out: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage = cfg["stage_b"]
    extra = []
    for d in cfg["d_values"]:
        ranked = sorted([r for r in records if r["d"] == d], key=lambda r: -r["primary"]["meaningful_accuracy"])[:stage["top_k"]]
        for r in ranked:
            extra.append(run_one(cfg, r["width"], r["blocks"], d, stage["seeds"][0], stage["budget"],
                                 stage["curriculum"], out, "stage_b"))
            print(json.dumps(dict(stage="B", d=d, w=r["width"], b=r["blocks"],
                                  meaningful=round(extra[-1]["primary"]["meaningful_accuracy"], 4)), ensure_ascii=False), flush=True)
    return extra


def stage_c(cfg: dict[str, Any], out: Path, d: int, selected: dict[str, Any]) -> dict[str, Any]:
    records = [selected]
    for seed in cfg["stage_c"]["seeds"]:
        records.append(run_one(cfg, selected["width"], selected["blocks"], d, seed, selected["budget"],
                               cfg["stage_a"]["curriculum"] if selected["budget"] == cfg["stage_a"]["budget"] else cfg["stage_b"]["curriculum"],
                               out, "stage_c"))
        print(json.dumps(dict(stage="C", d=d, seed=seed,
                              meaningful=round(records[-1]["primary"]["meaningful_accuracy"], 4)), ensure_ascii=False), flush=True)
    band = cfg["band"]
    acc = [r["primary"]["meaningful_accuracy"] for r in records]
    refs = [r["references"] for r in records]
    memoryless_max = max(max(r.values()) for r in refs)
    oracle_min = min(r["oracle"] for r in records)
    summary = dict(selected=dict(width=selected["width"], blocks=selected["blocks"], budget=selected["budget"],
                                 d=d, parameters=selected["parameters"], inference_ops=selected["inference_ops"]),
                   seeds=[r["seed"] for r in records], accuracy=acc, mean=float(np.mean(acc)),
                   min=float(np.min(acc)), max=float(np.max(acc)), memoryless_max=memoryless_max, oracle_min=oracle_min,
                   learn=bool(float(np.mean(acc)) >= band["learn_mean_min"] and float(np.min(acc)) >= band["learn_seed_min"]),
                   nonceiling=bool(float(np.mean(acc)) <= band["nonceiling_mean_max"] and float(np.max(acc)) <= band["nonceiling_seed_max"]),
                   rows=records)
    summary["qualified"] = bool(summary["learn"] and summary["nonceiling"]
                                and memoryless_max <= band["memoryless_max"] and oracle_min >= band["oracle_min"])
    save_json(out / "stage_c_summary.json", summary)
    return summary


def validate_config(c: dict[str, Any]) -> None:
    assert c["experiment_id"] == "EXP-0032" and c["version"] in ("v1", "v2")
    assert c["menu"] == [[32, 1], [32, 2], [32, 3], [48, 1], [48, 2], [48, 3],
                         [64, 1], [64, 2], [64, 3], [96, 1], [96, 2], [96, 3]]
    assert c["d_values"] == [2, 3] and c["supervision"] == "PLAIN"
    assert c["stage_a"] == dict(budget=2000, curriculum=[[500, 2], [1000, 4], [1500, 8], [2000, 12]], seeds=[29101])
    assert c["stage_b"] == dict(budget=6000, curriculum=[[1500, 2], [3000, 4], [4500, 8], [6000, 12]], top_k=3, seeds=[29101])
    assert c["stage_c"] == dict(seeds=[29102, 29111, 29112, 29113])
    assert c["band"] == dict(learn_mean_min=.65, learn_seed_min=.58, memoryless_margin_mean=.05,
                             memoryless_margin_seed=.02, nonceiling_mean_max=.95, nonceiling_seed_max=.97,
                             oracle_min=.99)
    parent = json.loads(Path(c["parent_config"]).read_text())
    for key in ("control", "streams", "evaluation"):
        assert c[key] == parent[key], f"EXP-0032 must keep the parent {key} unchanged"
    for key in ("batch_size", "lr", "betas", "eps", "clip", "branch_weight", "common_amplitude", "leak", "sigma"):
        assert c["training"][key] == parent["training"][key], key


def manifest(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    status = git_output("status", "--porcelain")
    from .order_observation import sha
    frozen = [config_path, Path(cfg["protocol"]), Path(cfg["training_module"]), Path(cfg["eval_module"]),
              Path(cfg["parent_config"]), Path("uv.lock"), Path("artifacts/validation/suite.json"),
              Path("src/neurotopo/arch0001_control_sweep.py"), Path("tests/test_arch0001_control_sweep.py")]
    return dict(experiment_id="EXP-0032", version=cfg["version"], gate="A2_TASK_CARRIER", config=cfg,
                git_commit=git_output("rev-parse", "HEAD"), git_dirty=bool(status.strip()),
                git_status=status, source_sha256=source_fingerprints(),
                frozen_hashes={str(p): sha(p) for p in frozen},
                python=sys.version, torch=torch.__version__, numpy=np.__version__,
                platform=platform.platform(),
                scope="Control-side task-carrier sweep only; no candidate training, no A2 effect claim.")


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
    todo = ["sweep", "analyze"] if phases == "all" else phases.split(",")
    for phase in todo:
        if time.monotonic() - started > cfg["resource"]["wall_seconds_cap"]:
            raise TimeoutError("Shared EXP-0032 wall budget")
        if phase == "sweep":
            records = stage_a(cfg, output)
            selected = select(cfg, records)
            stage_b_records: list[dict[str, Any]] = []
            if selected is None:
                stage_b_records = stage_b(cfg, output, records)
                selected = select(cfg, records + stage_b_records)
            summary = dict(stage_a=[r for r in records],
                           stage_b=stage_b_records,
                           selected=(None if selected is None else dict(d=selected[0], record=selected[1])))
            save_json(output / "sweep_summary.json", summary)
        elif phase == "analyze":
            summary = json.loads((output / "sweep_summary.json").read_text())
            confirm = None
            if summary["selected"] is not None:
                d, record = summary["selected"]["d"], summary["selected"]["record"]
                confirm_path = output / "stage_c_summary.json"
                if confirm_path.exists():
                    confirm = json.loads(confirm_path.read_text())
                else:
                    confirm = stage_c(cfg, output, d, record)
            if summary["selected"] is None:
                decision, substatus = "INCONCLUSIVE", "CONTROL_MENU_EXHAUSTED_NO_BAND"
            elif confirm is None or not confirm["qualified"]:
                decision, substatus = "INCONCLUSIVE", "QUALIFICATION_NOT_REPRODUCED"
            else:
                decision, substatus = "SUPPORTED", "A2_TASK_CARRIER_QUALIFIED"
            analysis = dict(experiment_id="EXP-0032", version=cfg["version"], gate="A2_TASK_CARRIER",
                            decision=decision, substatus=substatus, sweep=summary, confirm=confirm,
                            seconds=time.monotonic() - started,
                            scope=("Control-side task-carrier sweep only; no candidate run, no A2 effect, "
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
    print(json.dumps({k: v for k, v in outcome.items() if k not in ("sweep", "confirm")}, ensure_ascii=False, default=str))
