"""SC-REL-v4: task-general counterfactual branch supervision (training-side only).

Implements the ARCH-0001 v4 contract (research/design/arch0001_successor_compatible_core_v4.md) on top of the
frozen v3 module: same SuccessorCore inference path, but the branch loss uses task-provided counterfactual
targets with a mask (excluding the terminal readout index). Nothing in arch0001_core.py is modified.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .arch0001_core import SuccessorCore, op_estimate, tensors

CANDIDATE_INFERENCE_OPS = op_estimate("CAND")["inference"]


def build_v4(seed: int, correct: bool = True) -> SuccessorCore:
    return SuccessorCore(seed, correct=correct)


def with_mask(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(data)
    mask = np.ones((len(data["b0"]), data["targets"].shape[1]), bool)
    mask[:, -1] = False
    out["branch_mask"] = mask
    return out


def v4_loss(parts: dict[str, torch.Tensor], data: dict[str, np.ndarray],
            weight: float = .1) -> tuple[torch.Tensor, dict[str, float]]:
    target = torch.from_numpy(data["targets"])
    l_actual = F.binary_cross_entropy_with_logits(parts["logits"], target)
    labels = torch.from_numpy(data["branch_targets"])
    per = F.binary_cross_entropy_with_logits(parts["branch"], labels, reduction="none").mean(dim=2)
    mask = torch.from_numpy(data["branch_mask"])
    l_branch = per[mask].mean()
    return l_actual + weight * l_branch, dict(l_actual=float(l_actual.detach()), l_branch=float(l_branch.detach()))


def curriculum_length(step: int, curriculum: list[list[int]]) -> int:
    for stop, length in curriculum:
        if step <= stop:
            return length
    raise ValueError(step)


def fit_v4(seed: int, d: int, cfg: dict[str, Any], updates: int | None = None,
           generate=None, correct: bool = True) -> dict[str, Any]:
    """Train SC-REL-v4 on a DELAY-d style stream. `generate` must accept (n, length, root, d, common, leak, sigma)."""
    if generate is None:
        raise ValueError("generate callable required")
    train = cfg["training"]
    updates = train["updates"] if updates is None else updates
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = build_v4(seed, correct=correct)
    opt = torch.optim.Adam(model.parameters(), lr=train["lr"], betas=tuple(train["betas"]), eps=train["eps"],
                           weight_decay=0)
    started = time.monotonic()
    data_hash = hashlib.sha256()
    losses = []
    for step in range(1, updates + 1):
        length = curriculum_length(step, train["curriculum"])
        data = with_mask(generate(train["batch_size"], length,
                                  [cfg["streams"]["train_namespace"], seed, step, d], d,
                                  train["common_amplitude"], train["leak"], train["sigma"]))
        for key in ("commands", "common", "targets", "branch_targets"):
            data_hash.update(np.ascontiguousarray(data[key]).tobytes())
        model.train()
        opt.zero_grad(set_to_none=True)
        parts = model(*tensors(data), train=True)
        loss, terms = v4_loss(parts, data, train["branch_weight"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train["clip"], error_if_nonfinite=True)
        opt.step()
        losses.append(float(loss.detach()))
    import hashlib as _h
    h = _h.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().numpy().tobytes())
    return dict(model=model, record=dict(kind="CAND_V4", seed=seed, d=d, updates=updates,
                                         training_data_sha256=data_hash.hexdigest(),
                                         final_hash=h.hexdigest(), losses_last=float(np.mean(losses[-25:])),
                                         loss_checkpoints={str(c[0]): float(np.mean(losses[max(0, c[0] - 25):c[0]] or losses[-25:]))
                                                           for c in train["curriculum"]},
                                         seconds=time.monotonic() - started,
                                         parameters=sum(p.numel() for p in model.parameters()),
                                         inference_ops=CANDIDATE_INFERENCE_OPS,
                                         executed_flops="NOT_MEASURED", peak_memory="NOT_MEASURED"))


def evaluate_v4(model: torch.nn.Module, data: dict[str, np.ndarray], batch: int = 128) -> dict[str, torch.Tensor]:
    model.eval()
    logits, branch = [], []
    with torch.no_grad():
        for start in range(0, len(data["b0"]), batch):
            chunk = {k: v[start:start + batch] for k, v in data.items()}
            parts = model(*tensors(chunk), train=True)
            logits.append(parts["logits"].numpy())
            branch.append(parts["branch"].numpy())
    return dict(logits=torch.from_numpy(np.concatenate(logits)), branch=torch.from_numpy(np.concatenate(branch)))


def candidate_metrics(parts: dict[str, torch.Tensor], data: dict[str, np.ndarray], d: int,
                      meaningful_slice) -> dict[str, Any]:
    logits = parts["logits"].detach().numpy().astype(np.float64)
    target = data["targets"] >= .5
    sel = meaningful_slice(d, data["commands"].shape[1])
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite logits")
    correct = (logits >= 0) == target
    branch = parts["branch"].detach().numpy().astype(np.float64)
    branch_ok = (branch >= 0) == (data["branch_targets"] >= .5)
    return dict(meaningful_accuracy=float(correct[:, sel].mean()),
                accuracy_all=float(correct.mean()),
                per_position=[float(x) for x in correct[:, sel].mean(axis=0)],
                branch_joint_accuracy=float(branch_ok.all(axis=2).mean()),
                logits_sha256=hashlib.sha256(parts["logits"].detach().numpy().tobytes()).hexdigest(),
                branch_sha256=hashlib.sha256(parts["branch"].detach().numpy().tobytes()).hexdigest(),
                examples=int(len(logits)), steps=int(data["commands"].shape[1]))
