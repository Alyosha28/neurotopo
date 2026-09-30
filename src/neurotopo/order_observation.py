"""EXP-0017: fixed ORDER-OBS control qualification, no candidate training."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .run import git_output, source_fingerprints, write_json

CELLS = ("CO", "CU", "NO", "NU")
SEEDS = (17001, 17002, 17003)
ARMS = ("D16", "D64")
R = np.array([1, 2, 0, 4, 5, 3])
SC = np.array([3, 4, 5, 0, 1, 2])
SN = np.array([3, 5, 4, 0, 2, 1])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def numeric_code_hash() -> str:
    """Fingerprint precisely the numerical definitions measured in preflight."""
    names = {"oracle", "generate", "ControlGRU", "input_tensor", "kl_scores"}
    parts = [ast.dump(n, include_attributes=False) for n in ast.parse(Path(__file__).read_text()).body
             if getattr(n, "name", None) in names]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def failure_status(failure: dict, manifest: dict) -> str:
    """Preserve a child's original timeout instead of its supervisor exit-code wrapper."""
    if failure.get("stage") == "run" and manifest.get("error", "").startswith("TimeoutError("):
        return "INCOMPLETE_BUDGET"
    return failure["substatus"]


def oracle(tokens: np.ndarray, cell: str) -> np.ndarray:
    """Normalized forward probabilities after every visible token, float64."""
    if cell not in CELLS or tokens.ndim != 2 or np.any((tokens < 0) | (tokens > 4)):
        raise ValueError("Invalid cell/tokens")
    inverse_r = np.argsort(R)
    inverse_s = np.argsort(SC if cell[0] == "C" else SN)
    q = np.zeros((len(tokens), 6), dtype=np.float64)
    q[:, 0] = 1
    outputs = []
    for tok in tokens.T:
        qr, qs = q[:, inverse_r], q[:, inverse_s]
        nxt = q.copy()
        for value, update in ((0, qr), (1, qs), (2, (qr + qs) / 2)):
            selected = tok == value
            nxt[selected] = update[selected]
        for value in (3, 4):
            selected = tok == value
            mask = ((np.arange(6) % 3 == 0) == (value == 4))
            filtered = q[selected] * mask
            normalizer = filtered.sum(1, keepdims=True)
            if np.any(normalizer <= 0):
                raise ValueError("Impossible sensor observation")
            nxt[selected] = filtered / normalizer
        q = nxt
        outputs.append(q.copy())
    return np.stack(outputs, axis=1)


def generate(n: int, operations: int, cell: str, root: list[int]) -> dict[str, np.ndarray]:
    """Independent local operation/masking streams; no hidden global RNG."""
    if operations % 4 or cell not in CELLS or n < 1:
        raise ValueError("Invalid fixed task dimensions")
    op_seed, mask_seed = np.random.SeedSequence(root).spawn(2)
    actions = np.random.default_rng(op_seed).integers(0, 2, (n, operations), dtype=np.int8)
    masked = np.random.default_rng(mask_seed).random((n, operations)) < .25
    if cell[1] == "O":
        masked[:] = False
    states = np.zeros(n, dtype=np.int8)
    tokens = np.empty((n, operations + operations // 4), dtype=np.int8)
    s = SC if cell[0] == "C" else SN
    cursor = 0
    for t in range(operations):
        states = np.where(actions[:, t] == 0, R[states], s[states]).astype(np.int8)
        tokens[:, cursor] = np.where(masked[:, t], 2, actions[:, t])
        cursor += 1
        if (t + 1) % 4 == 0:
            tokens[:, cursor] = 3 + (states % 3 == 0)
            cursor += 1
    return {"tokens": tokens, "target": oracle(tokens, cell), "latent_final": states}


class ControlGRU(nn.Module):
    def __init__(self, arm: str, seed: int):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Only registered controls may be constructed")
        self.hidden = int(arm[1:])
        self.gru = nn.GRU(5, self.hidden, batch_first=True)
        self.readout = nn.Linear(self.hidden, 6)
        torch.manual_seed(seed)
        with torch.no_grad():
            for p in (self.gru.weight_ih_l0, self.gru.weight_hh_l0, self.readout.weight):
                p.uniform_(-1 / math.sqrt(self.hidden), 1 / math.sqrt(self.hidden))
            self.gru.bias_ih_l0.zero_()
            self.gru.bias_hh_l0.zero_()
            self.readout.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.gru(x)
        return self.readout(states)


def input_tensor(tokens: np.ndarray) -> torch.Tensor:
    return F.one_hot(torch.from_numpy(tokens.astype(np.int64)), 5).float()


def kl_scores(q: np.ndarray, logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    logq = np.zeros_like(q)
    np.log(q, out=logq, where=q > 0)
    scores = (q * (logq - logp)).sum(-1)
    if not np.isfinite(scores).all() or scores.min() < -1e-10:
        raise FloatingPointError("Invalid KL")
    return scores


def seed_statistics(values: list[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=np.float64)
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    half = 4.302652729911275 * sd / math.sqrt(3)
    return {"values": values, "mean": mean, "sample_sd": sd, "t95_df2": [mean - half, mean + half]}


def classify(records: list[dict]) -> tuple[str, str, dict]:
    expected = {(c, s, a) for c in CELLS for s in SEEDS for a in ARMS}
    keys = [(r["cell"], r["seed"], r["arm"]) for r in records]
    if len(keys) != 24 or set(keys) != expected or any(r.get("status") != "COMPLETED" or r.get("updates") != 750 for r in records):
        return "INCONCLUSIVE", "ENGINEERING_ABORT", {}
    if any(not np.isfinite(r["metrics"][str(n)]["kl_mean"]) for r in records for n in (32, 128)):
        return "INCONCLUSIVE", "ENGINEERING_ABORT", {}
    cells = {}
    for cell in CELLS:
        rows = sorted((r for r in records if r["cell"] == cell and r["arm"] == "D64"), key=lambda r: r["seed"])
        iid = [r["metrics"]["32"]["kl_mean"] for r in rows]
        ood = [r["metrics"]["128"]["kl_mean"] for r in rows]
        cells[cell] = {"learning_pass": max(iid) < .1, "headroom_pass": float(np.mean(ood)) >= .02,
                       "iid": seed_statistics(iid), "ood": seed_statistics(ood),
                       "observed_max_possible_mean_improvement": float(np.mean(ood))}
    if not all(c["learning_pass"] for c in cells.values()):
        return "INCONCLUSIVE", "LEARNING_FAILURE", cells
    if not all(c["headroom_pass"] for c in cells.values()):
        return "INCONCLUSIVE", "CONTROL_CEILING", cells
    return "SUPPORTED", "CONTROL_PILOT_WINDOW", cells


def validate_config(c: dict) -> None:
    if c["experiment_id"] != "EXP-0017" or c["version"] != "v2" or c["workflow"] != "order_observation" or c["gate1_claim"] is not False:
        raise ValueError("Frozen identity differs")
    expected_task = dict(cells=list(CELLS), train_operations=32, eval_operations=[32, 128], sensor_every=4, mask_probability=.25, train_namespace=171001, eval_namespace=172001, reserved_namespace=173001, engineering_namespace=179001, eval_examples=4096)
    expected_training = dict(seeds=list(SEEDS), arms=list(ARMS), batch_size=64, updates=750, lr=.001, betas=[.9, .999], eps=1e-8, clip=1., threads=1, monitor_every=150)
    if c["task"] != expected_task or c["training"] != expected_training:
        raise ValueError("Frozen task/training differs")
    if c["qualification"] != dict(d64_iid_each_max=.1, d64_ood_mean_min=.02, t_critical_df2=4.302652729911275):
        raise ValueError("Frozen qualification differs")
    if c["resource"] != dict(worker_wall_seconds_cap=900, reproduction_wall_seconds_cap=300, research_update_cap=18000):
        raise ValueError("Frozen budget differs")
    if c["matching_criterion_version"] != "G1-MATCH-v3" or np.__version__ != "2.3.3" or torch.__version__.split("+")[0] != "2.8.0":
        raise ValueError("Pinned criterion/runtime required")


def evaluate(model: ControlGRU, cell: str, seed: int, deadline: float) -> tuple[dict, dict]:
    arrays, metrics = {}, {}
    model.eval()
    for n in (32, 128):
        data = generate(4096, n, cell, [172001, seed, CELLS.index(cell), n])
        q, tokens = data["target"][:, -1], data["tokens"]
        logits = []
        started = time.monotonic()
        with torch.no_grad():
            for start in range(0, 4096, 128):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Shared run budget reached")
                logits.append(model(input_tensor(tokens[start:start + 128]))[:, -1].numpy())
        z = np.concatenate(logits)
        latency = time.monotonic() - started
        kl = kl_scores(q, z)
        logq = np.zeros_like(q)
        np.log(q, out=logq, where=q > 0)
        metrics[str(n)] = {"kl_mean": float(kl.mean()), "data_sha256": array_hash(tokens), "target_sha256": array_hash(q),
                           "prediction_seconds": latency, "seconds_per_example": latency / 4096,
                           "posterior_entropy_mean": float(-(q * logq).sum(1).mean()),
                           "sensor1_fraction": float((tokens[:, 4::5] == 4).mean()),
                           "oracle_bayes_accuracy": float(q.max(1).mean()),
                           "oracle_realized_accuracy": float((q.argmax(1) == data["latent_final"]).mean()),
                           "model_accuracy": float((z.argmax(1) == data["latent_final"]).mean()),
                           "support_size_counts": np.bincount((q > 0).sum(1), minlength=7).tolist()}
        metrics[str(n)]["accuracy_minus_oracle"] = metrics[str(n)]["model_accuracy"] - metrics[str(n)]["oracle_realized_accuracy"]
        for name, value in (("tokens", tokens), ("target", q), ("logits", z), ("kl", kl), ("latent", data["latent_final"])):
            arrays[f"{name}_{n}"] = value
    return metrics, arrays


def train_control(cell: str, seed: int, arm: str, output: Path, manifest: dict, deadline: float) -> dict:
    path = output / f"seed-{seed}" / cell / arm
    path.mkdir(parents=True, exist_ok=False)
    model = ControlGRU(arm, seed)
    opt = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    losses, checkpoints = [], []
    data_hash = hashlib.sha256()
    started, cpu = time.monotonic(), time.process_time()
    attempted = 0
    try:
        for step in range(1, 751):
            if time.monotonic() >= deadline:
                raise TimeoutError("Shared run budget reached")
            data = generate(64, 32, cell, [171001, seed, CELLS.index(cell), step])
            data_hash.update(data["tokens"].tobytes())
            data_hash.update(data["target"].tobytes())
            x, q = input_tensor(data["tokens"]), torch.from_numpy(data["target"]).float()
            opt.zero_grad(set_to_none=True)
            loss = -(q * F.log_softmax(model(x), -1)).sum(-1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            attempted += 1
            manifest["research_updates"] += 1
            opt.step()
            losses.append(float(loss.detach()))
            if step % 150 == 0:
                monitor = dict(cell=cell, seed=seed, arm=arm, updates=step, loss=losses[-1], seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu, cumulative_examples=64*step, cumulative_tokens=64*40*step)
                checkpoints.append(monitor)
                print(json.dumps(monitor), flush=True)
                write_json(output / "manifest.json", manifest)
        training_wall, training_cpu = time.monotonic() - started, time.process_time() - cpu
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), cell=cell, seed=seed, arm=arm), path / "final.pt")
        metrics, arrays = evaluate(model, cell, seed, deadline)
        with (path / "predictions.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        result = dict(cell=cell, seed=seed, arm=arm, status="COMPLETED", updates=attempted, metrics=metrics,
                      training_wall_seconds=training_wall, training_cpu_seconds=training_cpu, wall_seconds=time.monotonic() - started,
                      losses=losses, monitors=checkpoints, training_data_sha256=data_hash.hexdigest(),
                      checkpoint_sha256=sha(path / "final.pt"), predictions_sha256=sha(path / "predictions.npz"),
                      resources=dict(allocated_parameters=sum(p.numel() for p in model.parameters()), effective_parameters=sum(p.numel() for p in model.parameters()),
                                     parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()), dynamic_state_values=model.hidden,
                                     examples=64*attempted, input_tokens=64*40*attempted, updates=attempted, internal_steps_per_example=40,
                                     executed_flops="NOT_MEASURED: control qualification, no matched effect claim", peak_memory="NOT_MEASURED"))
        write_json(path / "result.json", result)
        return {k: v for k, v in result.items() if k not in ("losses", "monitors")}
    except BaseException as error:
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), path / "failed.pt")
        write_json(path / "failure.json", dict(cell=cell, seed=seed, arm=arm, attempted_updates=attempted, losses=losses, error=repr(error)))
        raise


def run(config_path: Path, output_override: Path | None = None) -> Path:
    started = time.monotonic()
    c = json.loads(config_path.read_text())
    validate_config(c)
    if git_output("status", "--porcelain"):
        raise RuntimeError("Clean frozen checkout required")
    suite_path = Path("artifacts/validation/suite.json")
    suite = json.loads(suite_path.read_text())
    if not suite.get("pass") or suite["source_sha256"] != source_fingerprints():
        raise RuntimeError("Source validation mismatch")
    preflight_path = Path("artifacts/validation/exp0017_preflight.json")
    preflight = json.loads(preflight_path.read_text())
    if not preflight["pass"] or preflight["numeric_code_sha256"] != numeric_code_hash() or preflight["config_sha256"] != sha(config_path):
        raise RuntimeError("Preflight mismatch")
    output = output_override or Path(c["output"])
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = dict(experiment_id=c["experiment_id"], version="v2", config=c, git_commit=git_output("rev-parse", "HEAD"), git_status="", source_repo=str(Path.cwd()),
                    source_sha256=source_fingerprints(), frozen_hashes={str(p): sha(p) for p in (config_path, Path(c["protocol"]), Path(c["entry_contract"]), Path("uv.lock"), suite_path, preflight_path)},
                    python=sys.version, numpy=np.__version__, torch=torch.__version__, platform=platform.platform(), machine=platform.machine(),
                    executable_sha256=sha(Path(sys.executable).resolve()), validation=suite, preflight=preflight,
                    status="RUNNING", records=[], research_updates=0, candidate_updates=0, gate1_claim=False)
    write_json(output / "manifest.json", manifest)
    try:
        for seed in SEEDS:
            for cell in CELLS:
                for arm in ARMS:
                    row = train_control(cell, seed, arm, output, manifest, started + 900)
                    manifest["records"].append(row)
                    write_json(output / "manifest.json", manifest)
        manifest.update(status="COMPLETED", seconds=time.monotonic() - started)
    except BaseException as error:
        manifest.update(status="FAILED", seconds=time.monotonic() - started, error=repr(error))
        write_json(output / "manifest.json", manifest)
        raise
    write_json(output / "manifest.json", manifest)
    return output


def analyze(output: Path) -> dict:
    if (output / "analysis.json").exists():
        raise FileExistsError("Immutable analysis exists")
    if (output / "worker_failure.json").exists():
        failure = json.loads((output / "worker_failure.json").read_text())
        manifest_path = output / "manifest.json"
        partial = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        result = dict(experiment_id="EXP-0017", decision="INCONCLUSIVE", substatus=failure_status(failure, partial), failure=failure,
                      research_updates=partial.get("research_updates", 0), records=partial.get("records", []), gate1_claim=False)
    else:
        manifest = json.loads((output / "manifest.json").read_text())
        validate_config(manifest["config"])
        if manifest["status"] != "COMPLETED":
            raise ValueError("Incomplete run")
        rows = []
        for row in manifest["records"]:
            path = output / f"seed-{row['seed']}" / row["cell"] / row["arm"]
            stored = json.loads((path / "result.json").read_text())
            for filename, field in (("final.pt", "checkpoint_sha256"), ("predictions.npz", "predictions_sha256")):
                if sha(path / filename) != stored[field]:
                    raise ValueError("Changed raw artifact")
            rows.append(row)
        decision, substatus, cells = classify(rows)
        result = dict(experiment_id="EXP-0017", decision=decision, substatus=substatus, cells=cells, records=rows,
                      research_updates=manifest["research_updates"], candidate_updates=0, gate1_claim=False,
                      interpretation="Control pilot only; all roots exposed. No organization effect or Gate1 claim.",
                      next_action="Prepare complete candidate protocol only" if substatus == "CONTROL_PILOT_WINDOW" else "Close this fixed 2x2 effect plan; retain all control outcomes")
    write_json(output / "analysis.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    def terminated(signum: int, frame: Any) -> None:
        raise TimeoutError("Supervisor terminated run")
    signal.signal(signal.SIGTERM, terminated)
    run(args.config, args.output)


if __name__ == "__main__":
    main()
