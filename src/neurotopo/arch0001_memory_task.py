"""EXP0030 v1: DELAY-d command-memory task family and control-side (INV_COMP) qualification.

Imports the frozen EXP0028 module for models/environment without modifying it. See experiments/EXP-0030/v1.md
for the pre-registered task definition, references and decision table. No candidate training here.
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
from torch.nn import functional as F

from .arch0001_core import build_model, generate as generate_env, model_tag, tensors
from .frozen_binary_probe import save_json
from .order_observation import sha
from .run import git_output, source_fingerprints

PI = 3.141592653589793


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def generate_memory(n: int, length: int, root: list[int], d: int, common: float = PI,
                    leak: float = 0., sigma: float = 0.) -> dict[str, np.ndarray]:
    """DELAY-d: readout index r>=1 must equal command u_{r-1-d} (out of range -> 0), i.i.d. commands.

    So d=1 means "remember and output the previous command"; a memoryless pass-through scores ~0.5 for d>=1.
    branch_targets[:, r, v] = y[r+1] (counterfactual next readout, independent of v; last index replicated).
    """
    assert d >= 1 and n % 2 == 0
    env = generate_env(n, length, root, "MIX", common, leak, sigma)
    rng = np.random.default_rng(np.random.SeedSequence([*root, 907]).spawn(1)[0])
    commands = rng.integers(0, 2, size=(n, length), dtype=np.int64)
    targets = np.zeros((n, length + 1), np.float32)
    for t in range(d, length):
        targets[:, t + 1] = commands[:, t - d]
    branch = np.empty((n, length + 1, 2), np.float32)
    branch[:, :-1, :] = targets[:, 1:, None]
    branch[:, -1, :] = targets[:, -1, None]
    env.update(b0=np.zeros(n, np.int64), commands=commands, targets=targets, branch_targets=branch)
    return env


def evaluation_stream(cfg: dict[str, Any], seed: int, d: int, length: int) -> dict[str, np.ndarray]:
    root = [cfg["streams"]["eval_namespace"], seed, length, 0]
    return generate_memory(cfg["streams"]["eval_examples"], length, root, d,
                           cfg["training"]["common_amplitude"], cfg["training"]["leak"], cfg["training"]["sigma"])


def curriculum_length(step: int, curriculum: list[list[int]]) -> int:
    for stop, length in curriculum:
        if step <= stop:
            return length
    raise ValueError(f"step {step} outside curriculum")


# ---------------------------------------------------------------------------
# Loss / metrics / references
# ---------------------------------------------------------------------------

def memory_loss(parts: dict[str, torch.Tensor], data: dict[str, np.ndarray], sup: str,
                weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    target = torch.from_numpy(data["targets"])
    l_actual = F.binary_cross_entropy_with_logits(parts["logits"], target)
    if sup == "PLAIN":
        return l_actual, dict(l_actual=float(l_actual.detach()), l_branch=0.)
    labels = torch.from_numpy(data["branch_targets"])
    l_branch = F.binary_cross_entropy_with_logits(parts["branch"], labels)
    return l_actual + weight * l_branch, dict(l_actual=float(l_actual.detach()), l_branch=float(l_branch.detach()))


def memory_metrics(parts: dict[str, torch.Tensor], data: dict[str, np.ndarray]) -> dict[str, Any]:
    logits = parts["logits"].detach().numpy().astype(np.float64)
    target = data["targets"] >= .5
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite logits")
    correct = (logits >= 0) == target
    branch = parts["branch"].detach().numpy().astype(np.float64)
    branch_ok = (branch >= 0) == (data["branch_targets"] >= .5)
    return dict(accuracy_all=float(correct.mean()),
                final_accuracy=float(correct[:, -1].mean()),
                initial_accuracy=float(correct[:, 0].mean()),
                tail_accuracy=float(correct[:, max(1, correct.shape[1] // 2):].mean()),
                branch_joint_accuracy=float(branch_ok.all(axis=2).mean()),
                logits_sha256=sha_bytes(parts["logits"].detach().numpy().tobytes()),
                branch_sha256=sha_bytes(parts["branch"].detach().numpy().tobytes()),
                examples=int(len(logits)), steps=int(data["commands"].shape[1]))


def sha_bytes(raw: bytes) -> str:
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def reference_memoryless(data: dict[str, np.ndarray]) -> dict[str, float]:
    """Predict the command just applied (no memory)."""
    u = data["commands"]
    n, length = u.shape
    pred = np.zeros((n, length + 1), bool)
    pred[:, 1:] = u == 1
    target = data["targets"] >= .5
    return dict(accuracy_all=float((pred == target).mean()),
                tail_accuracy=float((pred[:, max(1, pred.shape[1] // 2):] == target[:, max(1, pred.shape[1] // 2):]).mean()))


def reference_oracle(data: dict[str, np.ndarray], d: int) -> dict[str, float]:
    """Explicit shift register over the observed command buffer; attainable ceiling."""
    u = data["commands"]
    n, length = u.shape
    out = np.zeros((n, length + 1), bool)
    for t in range(d, length):
        out[:, t + 1] = u[:, t - d] == 1
    target = data["targets"] >= .5
    return dict(accuracy_all=float((out == target).mean()),
                tail_accuracy=float((out[:, max(1, out.shape[1] // 2):] == target[:, max(1, out.shape[1] // 2):]).mean()))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fit_memory(seed: int, d: int, sup: str, cfg: dict[str, Any], updates: int | None = None,
               monitor: dict | None = None) -> dict[str, Any]:
    train = cfg["training"]
    updates = train["updates"] if updates is None else updates
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    control = cfg["control"]
    model = build_model(control["kind"], seed, control["width"], control["blocks"])
    opt = torch.optim.Adam(model.parameters(), lr=train["lr"], betas=tuple(train["betas"]), eps=train["eps"],
                           weight_decay=0)
    started = time.monotonic()
    import hashlib
    data_hash = hashlib.sha256()
    losses, curves = [], []
    for step in range(1, updates + 1):
        length = curriculum_length(step, train["curriculum"])
        data = generate_memory(train["batch_size"], length,
                               [cfg["streams"]["train_namespace"], seed, step, d], d,
                               train["common_amplitude"], train["leak"], train["sigma"])
        for key in ("commands", "common", "targets", "branch_targets"):
            data_hash.update(np.ascontiguousarray(data[key]).tobytes())
        model.train()
        opt.zero_grad(set_to_none=True)
        parts = model(*tensors(data), train=True)
        loss, terms = memory_loss(parts, data, sup, train["branch_weight"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train["clip"], error_if_nonfinite=True)
        opt.step()
        losses.append(float(loss.detach()))
        if monitor is not None and step in {c[0] for c in train["curriculum"]}:
            curves.append(dict(step=step, metrics=memory_metrics(evaluate_memory(model, monitor), monitor),
                               loss=float(np.mean(losses[-25:]))))
    return dict(model=model, record=dict(kind=control["kind"], tag=model_tag(control["kind"], control["width"], control["blocks"]),
                                         seed=seed, d=d, sup=sup, updates=updates,
                                         training_data_sha256=data_hash.hexdigest(),
                                         final_hash=model_hash(model), losses_last=float(np.mean(losses[-25:])),
                                         loss_checkpoints={str(c[0]): _window_mean(losses, c[0])
                                                           for c in train["curriculum"]},
                                         seconds=time.monotonic() - started, parameters=sum(p.numel() for p in model.parameters()),
                                         monitor=curves, executed_flops="NOT_MEASURED", peak_memory="NOT_MEASURED"))



def _window_mean(values: list[float], end: int, window: int = 25) -> float:
    chunk = values[max(0, end - window):end] or values[-window:]
    return float(np.mean(chunk)) if chunk else float("nan")


def model_hash(model: torch.nn.Module) -> str:
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def evaluate_memory(model: torch.nn.Module, data: dict[str, np.ndarray], batch: int = 128) -> dict[str, torch.Tensor]:
    model.eval()
    logits, branch = [], []
    with torch.no_grad():
        for start in range(0, len(data["b0"]), batch):
            chunk = {k: v[start:start + batch] for k, v in data.items()}
            parts = model(*tensors(chunk), train=True)
            logits.append(parts["logits"].numpy())
            branch.append(parts["branch"].numpy())
    return dict(logits=torch.from_numpy(np.concatenate(logits)), branch=torch.from_numpy(np.concatenate(branch)))


# ---------------------------------------------------------------------------
# Run phases
# ---------------------------------------------------------------------------

def run_condition(cfg: dict[str, Any], item: dict[str, Any], seed: int, out: Path) -> dict[str, Any]:
    d, sup = item["d"], item["sup"]
    monitor = evaluation_stream(cfg, seed, d, cfg["evaluation"]["primary_length"])
    result = fit_memory(seed, d, sup, cfg, monitor=monitor)
    record = result["record"]
    record["primary"] = memory_metrics(evaluate_memory(result["model"], monitor), monitor)
    for length in cfg["evaluation"]["secondary_lengths"]:
        data = evaluation_stream(cfg, seed, d, length)
        record[f"secondary_{length}"] = memory_metrics(evaluate_memory(result["model"], data), data)
    save_json(out / "runs" / f"{item['id']}_seed{seed}.json", record)
    return record


def run_scan(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    started = time.monotonic()
    records = []
    for item in cfg["menu"]:
        for seed in cfg["seeds"]["scan"]:
            if time.monotonic() - started > cfg["resource"]["wall_seconds_cap"]:
                raise TimeoutError("Shared EXP-0030 wall budget")
            records.append(run_condition(cfg, item, seed, out))
    summary = summarize_scan(cfg, records)
    save_json(out / "scan_summary.json", summary)
    return summary


def summarize_scan(cfg: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    band = cfg["band"]
    by_d: dict[int, Any] = {}
    for d in sorted({item["d"] for item in cfg["menu"]}):
        variants = {}
        for sup in ("PLAIN", "BRANCH"):
            rows = [r for r in records if r["d"] == d and r["sup"] == sup]
            if not rows:
                continue
            acc = [r["primary"]["accuracy_all"] for r in rows]
            variants[sup] = dict(seeds=[r["seed"] for r in rows], accuracy=acc, mean=float(np.mean(acc)),
                                 min=float(np.min(acc)), max=float(np.max(acc)), rows=rows)
        if not variants:
            continue
        best_sup = max(variants, key=lambda s: variants[s]["mean"])
        best = variants[best_sup]
        seed_streams = [evaluation_stream(cfg, r["seed"], d, cfg["evaluation"]["primary_length"]) for r in best["rows"]]
        memoryless = [reference_memoryless(s) for s in seed_streams]
        oracle = [reference_oracle(s, d) for s in seed_streams]
        learn = best["mean"] >= band["learn_mean_min"] and best["min"] >= band["learn_seed_min"]
        nonceiling = best["mean"] <= band["nonceiling_mean_max"] and best["max"] <= band["nonceiling_seed_max"]
        memory_ok = max(m["accuracy_all"] for m in memoryless) <= band["memoryless_max"]
        oracle_ok = min(o["accuracy_all"] for o in oracle) >= band["oracle_min"]
        by_d[d] = dict(d=d, best_sup=best_sup, variants=variants, learn=bool(learn), nonceiling=bool(nonceiling),
                       memoryless_max=max(m["accuracy_all"] for m in memoryless), oracle_min=min(o["accuracy_all"] for o in oracle),
                       memoryless=memoryless, oracle=oracle,
                       qualified=bool(learn and nonceiling and memory_ok and oracle_ok))
    qualified = None
    for d in sorted(by_d):
        if by_d[d]["qualified"]:
            qualified = dict(d=d, sup=by_d[d]["best_sup"])
            break
    return dict(by_d=by_d, qualified=qualified, band=band, menu=cfg["menu"])


def run_confirm(cfg: dict[str, Any], out: Path, qualified: dict[str, Any]) -> dict[str, Any]:
    item = dict(id=f"D{qualified['d']}_{qualified['sup']}", d=qualified["d"], sup=qualified["sup"])
    records = []
    for seed in cfg["seeds"]["confirm"]:
        records.append(run_condition(cfg, item, seed, out))
    band = cfg["band"]
    acc = [r["primary"]["accuracy_all"] for r in records]
    streams = [evaluation_stream(cfg, r["seed"], qualified["d"], cfg["evaluation"]["primary_length"]) for r in records]
    memoryless = max(reference_memoryless(s)["accuracy_all"] for s in streams)
    oracle = min(reference_oracle(s, qualified["d"])["accuracy_all"] for s in streams)
    confirm = dict(item=item, seeds=cfg["seeds"]["confirm"], accuracy=acc, mean=float(np.mean(acc)),
                   min=float(np.min(acc)), max=float(np.max(acc)),
                   learn=bool(float(np.mean(acc)) >= band["learn_mean_min"] and float(np.min(acc)) >= band["learn_seed_min"]),
                   nonceiling=bool(float(np.mean(acc)) <= band["nonceiling_mean_max"] and float(np.max(acc)) <= band["nonceiling_seed_max"]),
                   memoryless_max=memoryless, oracle_min=oracle,
                   rows=records)
    confirm["reproduced"] = bool(confirm["learn"] and confirm["nonceiling"]
                                 and memoryless <= band["memoryless_max"] and oracle >= band["oracle_min"])
    save_json(out / "confirm_summary.json", confirm)
    return confirm


def validate_config(c: dict[str, Any]) -> None:
    assert c["experiment_id"] == "EXP-0030" and c["version"] == "v1"
    assert c["control"] == dict(kind="INV_COMP", width=32, blocks=1)
    assert [m["id"] for m in c["menu"]] == ["D1_PLAIN", "D1_BRANCH", "D2_PLAIN", "D2_BRANCH", "D3_PLAIN", "D3_BRANCH"]
    assert [(m["d"], m["sup"]) for m in c["menu"]] == [(1, "PLAIN"), (1, "BRANCH"), (2, "PLAIN"), (2, "BRANCH"),
                                                       (3, "PLAIN"), (3, "BRANCH")]
    assert c["seeds"] == dict(scan=[29101, 29102], confirm=[29111, 29112, 29113])
    assert c["training"] == dict(updates=2000, batch_size=64, lr=.003, betas=[.9, .999], eps=1e-8, clip=1.,
                                 branch_weight=.1, curriculum=[[500, 2], [1000, 4], [1500, 8], [2000, 12]],
                                 common_amplitude=PI, leak=0., sigma=0.)
    assert c["streams"] == dict(train_namespace=292001, eval_namespace=293001, eval_examples=1024)
    assert c["evaluation"] == dict(primary_length=12, secondary_lengths=[64, 256])
    assert c["band"] == dict(learn_mean_min=.65, learn_seed_min=.58, nonceiling_mean_max=.95,
                             nonceiling_seed_max=.97, memoryless_max=.55, oracle_min=.99)
    assert c["resource"] == dict(max_updates=30000, wall_seconds_cap=1800, replay_wall_seconds_cap=600)
    assert np.__version__ == "2.3.3" and torch.__version__.split("+")[0] == "2.8.0"


def manifest(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    status = git_output("status", "--porcelain")
    return dict(experiment_id="EXP-0030", version="v1", gate="A2_TASK_QUALIFICATION", config=cfg,
                git_commit=git_output("rev-parse", "HEAD"), git_dirty=bool(status.strip()),
                git_status=status, source_sha256=source_fingerprints(),
                frozen_hashes={str(p): sha(p) for p in (config_path, Path(cfg["protocol"]),
                                                        Path(cfg["candidate_contract"]), Path(cfg["frozen_module"]),
                                                        Path("uv.lock"), Path("artifacts/validation/suite.json"),
                                                        Path("src/neurotopo/arch0001_memory_task.py"),
                                                        Path("tests/test_arch0001_memory_task.py"))},
                python=sys.version, torch=torch.__version__, numpy=np.__version__,
                platform=platform.platform(),
                scope="Control-side task qualification only; no candidate training, no A2 effect claim.")


def run(config_path: Path, output: Path, phases: str = "all") -> dict[str, Any]:
    cfg = json.loads(config_path.read_text())
    validate_config(cfg)
    suite = Path("artifacts/validation/suite.json")
    payload = json.loads(suite.read_text())
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
            raise TimeoutError("Shared EXP-0030 wall budget")
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
            if scan["qualified"] is None:
                decision, substatus = "INCONCLUSIVE", "NO_QUALIFIED_MEMORY_CONDITION"
            elif confirm is None:
                decision, substatus = "INCONCLUSIVE", "CONFIRMATION_MISSING"
            elif confirm["reproduced"]:
                decision, substatus = "SUPPORTED", "A2_TASK_QUALIFIED"
            else:
                decision, substatus = "INCONCLUSIVE", "QUALIFICATION_NOT_REPRODUCED"
            analysis = dict(experiment_id="EXP-0030", version="v1", gate="A2_TASK_QUALIFICATION",
                            decision=decision, substatus=substatus, scan=scan, confirm=confirm,
                            seconds=time.monotonic() - started,
                            scope=("Control-side task qualification only; no candidate run, no A2 effect, "
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
