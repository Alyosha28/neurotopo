"""EXP0029 v1: 12-seed replication and failure-mode diagnosis of ARCH-0001 candidate learning qualification.

Imports the frozen EXP0028 module (src/neurotopo/arch0001_core.py) without modifying it. See
experiments/EXP-0029/v1.md for the pre-registered decision table.
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

from .arch0001_core import fit, generate, metrics_for, model_tag
from .frozen_binary_probe import save_json
from .order_observation import sha
from .run import git_output, source_fingerprints


def stream_for(cfg: dict[str, Any], seed: int, cell: dict[str, Any]) -> dict[str, np.ndarray]:
    length, mode, amp, leak = cell
    root = [cfg["streams"]["eval_namespace"], seed, length, 1 if mode == "HOLD" else 2]
    return generate(cfg["streams"]["eval_examples"], length, root, mode, amp, leak, 0.)


def loss_checkpoints(losses: list[float]) -> dict[str, float]:
    windows = {"125": (100, 125), "250": (225, 250), "500": (475, 500), "1000": (975, 1000)}
    out = {}
    for name, (start, stop) in windows.items():
        window = losses[start:stop]
        out[name] = float(np.mean(window)) if window else float("nan")
    return out


def classify_failure(cells: dict[str, Any], thresholds: dict[str, Any]) -> str:
    needed = ("IID_H", "IID_U")
    if not all(c in cells for c in needed):
        return "ENGINEERING"
    if not all(np.isfinite([cells[c]["accuracy_all"] for c in needed])):
        return "ENGINEERING"
    hold = cells["IID_H"]["accuracy_all"]
    flip = cells["IID_U"]["flip_accuracy"]
    if flip is None:
        return "ENGINEERING"
    if hold >= thresholds["hold_collapse"]["IID_H_min"] and flip <= thresholds["hold_collapse"]["flip_max"]:
        return "HOLD_COLLAPSE"
    if (thresholds["partial_fit"]["flip_min"] < flip < thresholds["partial_fit"]["flip_max"]
            and hold >= thresholds["partial_fit"]["IID_H_min"]):
        return "PARTIAL_FIT"
    if hold < thresholds["broken"]["IID_H_max"]:
        return "TASK_LEARNING_BROKEN"
    return "UNCLASSIFIED"


def learning_flag(cells: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    flag = thresholds["learning_flag"]
    return bool(cells["IID_H"]["accuracy_all"] >= flag["IID_H"]
                and cells["IID_U"]["accuracy_all"] >= flag["IID_U"]
                and cells["IID_U"]["flip_accuracy"] >= flag["IID_U_flip"])


def run_seed(cfg: dict[str, Any], seed: int, arm: dict[str, Any], out: Path) -> dict[str, Any]:
    kind, width, blocks = arm["kind"], arm["width"], arm["blocks"]
    result = fit(kind, seed, dict(leak=0., sigma=0.), cfg, cfg["training"]["updates"],
                 width=width, blocks=blocks)
    record = result["record"]
    cells = {name: metrics_for(result["model"], stream_for(cfg, seed, cell))
             for name, cell in cfg["cells"].items()}
    record["cells"] = cells
    record["loss_checkpoints"] = loss_checkpoints(record["losses"])
    record.pop("losses", None)
    record["learning_flag"] = learning_flag(cells, cfg["thresholds"])
    record["failure_class"] = None if record["learning_flag"] else classify_failure(cells, cfg["thresholds"])
    save_json(out / "runs" / f"{model_tag(kind, width, blocks)}_seed{seed}.json", record)
    return record



def _mean_defined(values: list[Any]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None

def summarize(cfg: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in cfg["arms"]:
        tag = f"{arm['kind']}-w{arm['width']}-b{arm['blocks']}" if arm["kind"] == "INV_COMP" else arm["kind"]
        rows = [r for r in records if r["tag"] == tag]
        passes = [r for r in rows if r["learning_flag"]]
        failures = [r for r in rows if not r["learning_flag"]]
        arms[tag] = dict(
            tag=tag, kind=arm["kind"], seeds=[r["seed"] for r in rows], attempts=len(rows),
            passed=len(passes), failed=len(failures),
            failure_classes={r["seed"]: r["failure_class"] for r in failures},
            cells_mean={name: {key: _mean_defined([r["cells"][name][key] for r in rows])
                               for key in ("accuracy_all", "flip_accuracy", "response_joint_accuracy",
                                           "current_ok_next_response_bad", "bce")}
                        for name in cfg["cells"]},
            rows=[dict(seed=r["seed"], passed=r["learning_flag"], failure_class=r["failure_class"],
                       loss_checkpoints=r["loss_checkpoints"],
                       final_hash=r["final_hash"], training_data_sha256=r["training_data_sha256"],
                       IID_H=r["cells"]["IID_H"]["accuracy_all"],
                       IID_U=r["cells"]["IID_U"]["accuracy_all"],
                       IID_U_flip=r["cells"]["IID_U"]["flip_accuracy"],
                       response_joint=r["cells"]["IID_U"]["response_joint_accuracy"],
                       seconds=r["seconds"], parameters=r["parameters"]) for r in rows])
    seeds = sorted({r["seed"] for r in records})
    paired = {}
    for seed in seeds:
        fails = sorted(r["tag"] for r in records if r["seed"] == seed and not r["learning_flag"])
        if fails:
            paired[seed] = fails
    return dict(arms=arms, seeds=seeds, paired_failures=paired)


def decide(cfg: dict[str, Any], summary: dict[str, Any]) -> tuple[str, str]:
    t = cfg["thresholds"]
    minimum = t["pass_min_seeds"]
    cand = summary["arms"]["CAND"]
    control = summary["arms"]["INV_COMP-w32-b1"]
    all_rows = [r for arm in summary["arms"].values() for r in arm["rows"]]
    if any(r["failure_class"] == "ENGINEERING" for r in all_rows):
        return "INCONCLUSIVE", "ENGINEERING_FAILURE"
    if control["passed"] < minimum:
        return "INCONCLUSIVE", "CONTROL_LEARNING_UNRELIABLE"
    if cand["passed"] < minimum:
        return "INCONCLUSIVE", "CANDIDATE_LEARNING_UNRELIABLE"
    classes = set(cand["failure_classes"].values())
    if classes and classes != {"HOLD_COLLAPSE"}:
        return "INCONCLUSIVE", "FAILURE_MODE_UNRESOLVED"
    return "PARTIALLY_SUPPORTED", "LEARNING_QUALIFICATION_AT_BUDGET"


def validate_config(c: dict[str, Any]) -> None:
    assert c["experiment_id"] == "EXP-0029" and c["version"] == "v1"
    assert c["arms"] == [dict(kind="CAND", width=32, blocks=1), dict(kind="EXTRA_SUP", width=32, blocks=1),
                         dict(kind="INV_COMP", width=32, blocks=1)]
    assert c["seeds"] == list(range(28201, 28213))
    assert c["training"] == dict(updates=1000, batch_size=64, lr=.003, betas=[.9, .999], eps=1e-8, clip=1.,
                                 common_amplitude=.4, branch_weight=.1,
                                 curriculum=[[125, 1], [250, 2], [500, 4], [1000, 8]])
    assert c["streams"] == dict(train_namespace=282001, eval_namespace=285001, eval_examples=1024)
    assert c["thresholds"] == dict(learning_flag=dict(IID_H=.95, IID_U=.95, IID_U_flip=.95),
                                   hold_collapse=dict(IID_H_min=.95, flip_max=.65),
                                   partial_fit=dict(flip_min=.65, flip_max=.95, IID_H_min=.9),
                                   broken=dict(IID_H_max=.9), pass_min_seeds=11)
    assert c["resource"] == dict(max_updates=36000, wall_seconds_cap=1800, replay_wall_seconds_cap=600)
    assert np.__version__ == "2.3.3" and torch.__version__.split("+")[0] == "2.8.0"


def manifest(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    status = git_output("status", "--porcelain")
    return dict(experiment_id="EXP-0029", version="v1", gate="A1_LEARNING_QUALIFICATION", config=cfg,
                git_commit=git_output("rev-parse", "HEAD"), git_dirty=bool(status.strip()),
                git_status=status, source_sha256=source_fingerprints(),
                frozen_hashes={str(p): sha(p) for p in (config_path, Path(cfg["protocol"]),
                                                        Path(cfg["candidate_contract"]),
                                                        Path(cfg["frozen_module"]), Path("uv.lock"),
                                                        Path("artifacts/validation/suite.json"),
                                                        Path("src/neurotopo/arch0001_learning_qualification.py"),
                                                        Path("tests/test_arch0001_learning_qualification.py"))},
                python=sys.version, torch=torch.__version__, numpy=np.__version__,
                platform=platform.platform(),
                scope="A1 learning-qualification replication only; no A2 effect, no architecture verdict.")


def run(config_path: Path, output: Path, phases: str = "all") -> dict[str, Any]:
    cfg = json.loads(config_path.read_text())
    validate_config(cfg)
    suite = Path("artifacts/validation/suite.json")
    if not suite.exists():
        raise RuntimeError("Run scripts/verify.py before EXP-0029")
    payload = json.loads(suite.read_text())
    if not payload["pass"] or payload["source_sha256"] != source_fingerprints():
        raise RuntimeError("Validation suite does not match current sources; re-run scripts/verify.py")
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "manifest.json").exists():
        save_json(output / "manifest.json", manifest(cfg, config_path))
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    todo = ["run", "analyze"] if phases == "all" else phases.split(",")
    for phase in todo:
        if phase == "run":
            for seed in cfg["seeds"]:
                for arm in cfg["arms"]:
                    if time.monotonic() - started > cfg["resource"]["wall_seconds_cap"]:
                        raise TimeoutError("Shared EXP-0029 wall budget")
                    resumed = output / "runs" / f"{model_tag(arm['kind'], arm['width'], arm['blocks'])}_seed{seed}.json"
                    if resumed.exists():
                        record = json.loads(resumed.read_text())
                        assert (record["seed"], record["kind"]) == (seed, arm["kind"]), resumed
                        records.append(record)
                        print(json.dumps(dict(seed=seed, tag=record["tag"], resumed=True,
                                              flag=record["learning_flag"],
                                              class_=record["failure_class"]), ensure_ascii=False), flush=True)
                        continue
                    records.append(run_seed(cfg, seed, arm, output))
                    print(json.dumps(dict(seed=seed, tag=records[-1]["tag"],
                                          flag=records[-1]["learning_flag"],
                                          class_=records[-1]["failure_class"]), ensure_ascii=False), flush=True)
            summary = summarize(cfg, records)
            save_json(output / "summary.json", summary)
        elif phase == "analyze":
            if not records:
                rows = [json.loads(p.read_text()) for p in sorted((output / "runs").glob("*.json"))
                        if not p.name.startswith("._")]
                records = rows
            summary = summarize(cfg, records)
            decision, substatus = decide(cfg, summary)
            analysis = dict(experiment_id="EXP-0029", version="v1", decision=decision, substatus=substatus,
                            seeds=cfg["seeds"], attempts=len(records), updates=len(records) * cfg["training"]["updates"],
                            summary=summary, seconds=time.monotonic() - started,
                            scope=("Learning-qualification replication only; A1 remains not passed because no "
                                   "non-ceiling qualified task exists, and no A2 effect is claimed."))
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
    print(json.dumps({k: v for k, v in outcome.items() if k != "summary"}, ensure_ascii=False, default=str))
