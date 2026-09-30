"""EXP0028 / ARCH-0001 A1: SC-REL-v3 candidate, INV-COMP control, engineering checks and qualification runs.

Protocol: experiments/EXP-0028/v1.md (frozen before the run). Candidate contract:
research/design/arch0001_successor_compatible_core_v3.md (A0, decision0102).
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
from torch import nn
from torch.nn import functional as F

from .frozen_binary_probe import save_json
from .learned_protected_pilot import generate as exp0020_generate
from .learned_protected_pilot import gram, length_at, radial, rotations
from .order_observation import sha
from .run import git_output, source_fingerprints

PI = 3.141592653589793
KINDS = ("CAND", "EXTRA_SUP", "INV_COMP")
DATA_KEYS = ("b0", "commands", "common", "leak", "noise")


# ---------------------------------------------------------------------------
# Data generation (EXP0020-compatible; sigma=0 must be bitwise identical)
# ---------------------------------------------------------------------------

def generate(n: int, length: int, root: list[int], mode: str, common: float = .4,
             leak: float = 0., sigma: float = 0.) -> dict[str, np.ndarray]:
    assert n % 2 == 0 and mode in ("MIX", "HOLD", "UPDATE")
    seeds = np.random.SeedSequence(root).spawn(5)
    initial, cmd, rot, extra, noisy = [np.random.default_rng(s) for s in seeds]
    b0 = initial.permutation(np.repeat([0, 1], n // 2)).astype(np.int64)
    update = initial.permutation(np.repeat([False, True], n // 2)) if mode == "MIX" else np.full(n, mode == "UPDATE")
    u = cmd.integers(0, 2, size=(n, length), dtype=np.int64)
    u[~update] = 0
    u[update, 0] = 1
    target = np.bitwise_xor.accumulate(np.column_stack((b0, u)), axis=1).astype(np.float32)
    return dict(b0=b0, commands=u, common=rotations(rot, (n, length + 1), common),
                leak=rotations(extra, (n, length), leak),
                noise=(noisy.normal(size=(n, length, 3, 2)) * sigma).astype(np.float32),
                targets=target)


def tensors(data: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    return (torch.from_numpy(data["b0"]), torch.from_numpy(data["commands"]),
            torch.from_numpy(data["common"]), torch.from_numpy(data["leak"]),
            torch.from_numpy(data["noise"]))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SuccessorCore(nn.Module):
    """SC-REL-v3 (correct=True) and the EXTRA_SUP base-REL control (correct=False)."""

    def __init__(self, seed: int, correct: bool):
        super().__init__()
        self.kind = "CAND" if correct else "EXTRA_SUP"
        self.correct = correct
        self.encoder = nn.Embedding(2, 6)
        self.readout = nn.Linear(3, 1)
        self.controller = nn.Sequential(nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, 4))
        self.corrector = (nn.Sequential(nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, 4))
                          if correct else None)
        torch.manual_seed(seed)
        with torch.no_grad():
            self.encoder.weight.normal_(0, .5)
            self.readout.weight.normal_(0, .1)
            self.readout.bias.zero_()
            nn.init.xavier_uniform_(self.controller[0].weight, gain=1.)
            nn.init.xavier_uniform_(self.controller[2].weight, gain=.5)
            self.controller[0].bias.zero_()
            self.controller[2].bias.zero_()
            if self.corrector is not None:
                nn.init.xavier_uniform_(self.corrector[0].weight, gain=1.)
                self.corrector[0].bias.zero_()
                self.corrector[2].weight.zero_()
                self.corrector[2].bias.zero_()
        self.calls = 0

    def transition(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        code = F.one_hot(u, 2).to(h.dtype)
        a = self.controller(torch.cat((gram(h), code), dim=1)).reshape(-1, 2, 2)
        return radial(h @ (torch.eye(2, dtype=h.dtype) + a))

    def forward(self, b0: torch.Tensor, commands: torch.Tensor, common: torch.Tensor,
                leak: torch.Tensor, noise: torch.Tensor, init_frame: torch.Tensor | None = None,
                train: bool = True) -> dict[str, torch.Tensor]:
        self.calls = 0
        h = radial(self.encoder(b0).reshape(-1, 3, 2))
        if init_frame is not None:
            h = init_frame @ h
        logits = [self.readout(gram(h)).squeeze(-1)]
        hypothesis = []
        branch = []
        for t in range(commands.shape[1]):
            if train:
                branch.append(self.branch_response(h))
            distorted = common[:, t + 1] @ h
            second = (leak[:, t] @ distorted[:, :, 1:2]).squeeze(-1)
            distorted = torch.stack((distorted[:, :, 0], second), dim=-1)
            distorted = distorted + noise[:, t]
            v = self.transition(distorted, commands[:, t])
            zeros = torch.zeros_like(commands[:, t])
            ones = torch.ones_like(commands[:, t])
            s0 = self.readout(gram(self.transition(v, zeros))).squeeze(-1)
            s1 = self.readout(gram(self.transition(v, ones))).squeeze(-1)
            hypothesis.append(torch.stack((s0, s1), dim=1))
            if self.correct:
                b = (0.1 * torch.tanh(self.corrector(torch.cat((gram(v), s0[:, None], s1[:, None]), dim=1)))
                     .reshape(-1, 2, 2))
                h = radial(v @ (torch.eye(2, dtype=v.dtype) + b))
            else:
                h = v
            logits.append(self.readout(gram(h)).squeeze(-1))
        if train:
            branch.append(self.branch_response(h))
        parts = dict(logits=torch.stack(logits, dim=1),
                     hypothesis=(torch.stack(hypothesis, dim=1) if hypothesis else None),
                     branch=(torch.stack(branch, dim=1) if branch else None))
        return parts

    def branch_response(self, h: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(h.shape[0], dtype=torch.int64)
        ones = torch.ones(h.shape[0], dtype=torch.int64)
        return torch.stack((self.readout(gram(self.transition(h, zeros))).squeeze(-1),
                            self.readout(gram(self.transition(h, ones))).squeeze(-1)), dim=1)


class InvComp(nn.Module):
    """INV-COMP strong control: direct regeneration of the six-value state, menu width x blocks."""

    def __init__(self, width: int, blocks: int, seed: int):
        super().__init__()
        self.kind = "INV_COMP"
        self.width = width
        self.blocks = blocks
        self.encoder = nn.Embedding(2, 6)
        self.readout = nn.Linear(3, 1)
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(5, width), nn.Tanh(), nn.Linear(width, 6))
                                   for _ in range(blocks)])
        torch.manual_seed(seed)
        with torch.no_grad():
            self.encoder.weight.normal_(0, .5)
            self.readout.weight.normal_(0, .1)
            self.readout.bias.zero_()
            for m in self.mlps:
                nn.init.xavier_uniform_(m[0].weight, gain=1.)
                nn.init.xavier_uniform_(m[2].weight, gain=.5)
                m[0].bias.zero_()
                m[2].bias.zero_()
        self.calls = 0

    def transition(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        code = F.one_hot(u, 2).to(h.dtype)
        j = radial(self.mlps[0](torch.cat((gram(h), code), dim=1)).reshape(-1, 3, 2))
        for m in self.mlps[1:]:
            j = radial(j + m(torch.cat((gram(j), code), dim=1)).reshape(-1, 3, 2))
        return j

    def forward(self, b0: torch.Tensor, commands: torch.Tensor, common: torch.Tensor,
                leak: torch.Tensor, noise: torch.Tensor, init_frame: torch.Tensor | None = None,
                train: bool = True) -> dict[str, torch.Tensor]:
        self.calls = 0
        h = radial(self.encoder(b0).reshape(-1, 3, 2))
        if init_frame is not None:
            h = init_frame @ h
        logits = [self.readout(gram(h)).squeeze(-1)]
        branch = []
        for t in range(commands.shape[1]):
            if train:
                branch.append(self.branch_response(h))
            distorted = common[:, t + 1] @ h
            second = (leak[:, t] @ distorted[:, :, 1:2]).squeeze(-1)
            distorted = torch.stack((distorted[:, :, 0], second), dim=-1)
            distorted = distorted + noise[:, t]
            h = self.transition(distorted, commands[:, t])
            logits.append(self.readout(gram(h)).squeeze(-1))
        if train:
            branch.append(self.branch_response(h))
        return dict(logits=torch.stack(logits, dim=1), hypothesis=None,
                    branch=(torch.stack(branch, dim=1) if branch else None))

    def branch_response(self, h: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(h.shape[0], dtype=torch.int64)
        ones = torch.ones(h.shape[0], dtype=torch.int64)
        return torch.stack((self.readout(gram(self.transition(h, zeros))).squeeze(-1),
                            self.readout(gram(self.transition(h, ones))).squeeze(-1)), dim=1)


def build_model(kind: str, seed: int, width: int = 32, blocks: int = 1) -> nn.Module:
    if kind == "CAND":
        return SuccessorCore(seed, correct=True)
    if kind == "EXTRA_SUP":
        return SuccessorCore(seed, correct=False)
    if kind == "INV_COMP":
        return InvComp(width, blocks, seed)
    raise ValueError(kind)


def model_tag(kind: str, width: int = 32, blocks: int = 1) -> str:
    return f"{kind}-w{width}-b{blocks}" if kind == "INV_COMP" else kind


def op_estimate(kind: str, width: int = 32, blocks: int = 1) -> dict[str, int]:
    """Analytic dense-MAC estimates; labelled estimates in every artifact."""
    if kind in ("CAND", "EXTRA_SUP"):
        base = 5 * 32 + 32 * 4
        corrector = (5 * 32 + 32 * 4) if kind == "CAND" else 0
        inference = 3 * base + corrector + 3 * 3
        training = 5 * base + corrector + 5 * 3
    else:
        per_block = 5 * width + width * 6
        inference = blocks * per_block + 3
        training = 3 * blocks * per_block + 3 * 3
    return dict(inference=inference, training=training)


def param_count(kind: str, width: int = 32, blocks: int = 1) -> int:
    if kind in ("CAND", "EXTRA_SUP"):
        total = 2 * 6 + (3 + 1) + (5 * 32 + 32) + (32 * 4 + 4)
        if kind == "CAND":
            total += (5 * 32 + 32) + (32 * 4 + 4)
        return total
    return 2 * 6 + (3 + 1) + blocks * ((5 * width + width) + (width * 6 + 6))


# ---------------------------------------------------------------------------
# Loss, training, evaluation
# ---------------------------------------------------------------------------

def batch_loss(parts: dict[str, torch.Tensor], data: dict[str, np.ndarray]) -> tuple[torch.Tensor, dict[str, float]]:
    target = torch.from_numpy(data["targets"])
    l_actual = F.binary_cross_entropy_with_logits(parts["logits"], target)
    labels = torch.stack((target, 1. - target), dim=2)
    l_branch = F.binary_cross_entropy_with_logits(parts["branch"], labels)
    loss = l_actual + 0.1 * l_branch
    return loss, dict(l_actual=float(l_actual.detach()), l_branch=float(l_branch.detach()))


def condition_metrics(parts: dict[str, torch.Tensor], data: dict[str, np.ndarray]) -> dict[str, Any]:
    logits = parts["logits"].detach().numpy().astype(np.float64)
    target = data["targets"].astype(np.float64)
    u = data["commands"]
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite predictions")
    correct = (logits >= 0) == (target >= .5)
    hold = u == 0
    flip = u == 1
    out = dict(bce=float((np.logaddexp(0, logits) - target * logits).mean()),
               accuracy_all=float(correct.mean()),
               initial_accuracy=float(correct[:, 0].mean()),
               final_accuracy=float(correct[:, -1].mean()),
               hold_accuracy=float(correct[:, 1:][hold].mean()) if hold.any() else None,
               flip_accuracy=float(correct[:, 1:][flip].mean()) if flip.any() else None,
               examples=int(len(logits)), steps=int(u.shape[1]),
               hold_steps=int(hold.sum()), flip_steps=int(flip.sum()),
               logits_sha256=hashlib.sha256(parts["logits"].detach().numpy().tobytes()).hexdigest())
    branch = parts["branch"] if parts.get("branch") is not None else None
    if branch is not None:
        b = branch.detach().numpy().astype(np.float64)
        ok_hold = (b[:, :, 0] >= 0) == (target >= .5)
        ok_flip = (b[:, :, 1] >= 0) == (target < .5)
        joint = ok_hold & ok_flip
        out["response_joint_accuracy"] = float(joint.mean())
        out["response_error"] = float(1. - joint.mean())
        current_ok = correct
        bad = (~joint) & current_ok
        out["current_ok_next_response_bad"] = float(bad.sum() / max(int(current_ok.sum()), 1))
        out["branch_sha256"] = hashlib.sha256(branch.detach().numpy().tobytes()).hexdigest()
    return out


def evaluate(model: nn.Module, data: dict[str, np.ndarray], batch: int = 128,
             init_frame: torch.Tensor | None = None) -> dict[str, Any]:
    model.eval()
    logits, branch, hyp = [], [], []
    with torch.no_grad():
        for start in range(0, len(data["b0"]), batch):
            chunk = {k: v[start:start + batch] for k, v in data.items()}
            parts = model(*tensors(chunk), init_frame=init_frame, train=True)
            logits.append(parts["logits"].numpy())
            if parts.get("branch") is not None:
                branch.append(parts["branch"].numpy())
            if parts.get("hypothesis") is not None:
                hyp.append(parts["hypothesis"].numpy())
    parts = dict(logits=torch.from_numpy(np.concatenate(logits)),
                 branch=(torch.from_numpy(np.concatenate(branch)) if branch else None),
                 hypothesis=(torch.from_numpy(np.concatenate(hyp)) if hyp else None))
    return parts


def metrics_for(model: nn.Module, data: dict[str, np.ndarray]) -> dict[str, Any]:
    return condition_metrics(evaluate(model, data), data)


def fit(kind: str, seed: int, condition: dict[str, Any], cfg: dict[str, Any], updates: int,
        width: int = 32, blocks: int = 1, lr: float = None, monitor: dict | None = None,
        train_data_override: dict | None = None) -> dict[str, Any]:
    """Fixed-budget deterministic training; returns the trained model plus a record."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train = cfg["training"]
    model = build_model(kind, seed, width, blocks)
    opt = torch.optim.Adam(model.parameters(), lr=train["lr"] if lr is None else lr,
                           betas=tuple(train["betas"]), eps=train["eps"], weight_decay=0)
    started = time.monotonic()
    cpu_started = time.process_time()
    data_hash = hashlib.sha256()
    losses = []
    curves = []
    initial = dict(encoder=sha_of_tensor(model.encoder.weight), readout=sha_of_tensor(model.readout.weight),
                   all=model_hash(model))
    for step in range(1, updates + 1):
        if train_data_override is not None:
            data = train_data_override
        else:
            cond = condition
            data = generate(train["batch_size"], length_at(step),
                            [cfg["streams"]["train_namespace"], seed, step, 0], "MIX",
                            train["common_amplitude"], cond["leak"], cond["sigma"])
        data_hash.update(np.ascontiguousarray(data["b0"]).tobytes())
        data_hash.update(np.ascontiguousarray(data["commands"]).tobytes())
        data_hash.update(np.ascontiguousarray(data["common"]).tobytes())
        data_hash.update(np.ascontiguousarray(data["leak"]).tobytes())
        data_hash.update(np.ascontiguousarray(data["noise"]).tobytes())
        model.train()
        opt.zero_grad(set_to_none=True)
        parts = model(*tensors(data), train=True)
        loss, terms = batch_loss(parts, data)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train["clip"], error_if_nonfinite=True)
        opt.step()
        losses.append(float(loss.detach()))
        if monitor is not None and step in (125, 250, 500, 1000):
            curves.append(dict(step=step, metrics=metrics_for(model, monitor),
                               loss=float(np.mean(losses[-25:]))))
    record = dict(kind=kind, tag=model_tag(kind, width, blocks), seed=seed, width=width, blocks=blocks,
                  updates=updates, losses=losses, monitor_curves=curves,
                  training_data_sha256=data_hash.hexdigest(), initial_hashes=initial,
                  final_hash=model_hash(model), seconds=time.monotonic() - started,
                  cpu_seconds=time.process_time() - cpu_started,
                  parameters=param_count(kind, width, blocks), state_values=6,
                  op_estimate=op_estimate(kind, width, blocks), executed_flops="NOT_MEASURED",
                  peak_memory="NOT_MEASURED")
    return dict(model=model, record=record)


def sha_of_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().numpy().tobytes()).hexdigest()


def model_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Engineering checks
# ---------------------------------------------------------------------------

def check_shapes_finite() -> dict[str, Any]:
    torch.manual_seed(11)
    data = generate(8, 4, [1, 1, 1, 1], "MIX", .4, .4, .3)
    out = {}
    expected_calls = {"CAND": 5 * 4 + 2, "EXTRA_SUP": 5 * 4 + 2, "INV_COMP": 3 * 4 + 2}
    for kind in KINDS:
        model = build_model(kind, 7)
        parts = model(*tensors(data), train=True)
        ok = bool(torch.isfinite(parts["logits"]).all() and torch.isfinite(parts["branch"]).all())
        out[kind] = dict(pass_=ok and tuple(parts["logits"].shape) == (8, 5)
                         and tuple(parts["branch"].shape) == (8, 5, 2) and model.calls == expected_calls[kind],
                         logits_shape=list(parts["logits"].shape), branch_shape=list(parts["branch"].shape),
                         calls=model.calls, expected_calls=expected_calls[kind])
    return dict(check="shapes_finite_call_counts", pass_=all(v["pass_"] for v in out.values()), detail=out)


def check_gradient_flow() -> dict[str, Any]:
    data = generate(16, 2, [2, 2, 2, 2], "MIX", 0., 0., 0.)
    report = {}
    for kind in KINDS:
        model = build_model(kind, 5)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        seen = {name: [] for name, _ in model.named_parameters()}
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss, _ = batch_loss(model(*tensors(data), train=True), data)
            loss.backward()
            for name, p in model.named_parameters():
                g = p.grad
                seen[name].append(bool(g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0))
            opt.step()
        report[kind] = dict(all_params_nonzero_by_step2=all(v[1] for v in seen.values()),
                            first_step_zero=sum(1 for v in seen.values() if not v[0]),
                            detail={k: v for k, v in seen.items() if not v[1]})
    ok = all(v["all_params_nonzero_by_step2"] for v in report.values())
    return dict(check="gradient_flow_two_steps", pass_=bool(ok), detail=report)


def check_gauge_invariance() -> dict[str, Any]:
    """Full-path gauge equivariance for the covariant arms; canonical-update invariance for INV-COMP.

    The control deliberately regenerates a canonical state from Gram inputs, so a global frame rotation of
    its input is not a symmetry of its update; the registered check for it is single-step input invariance.
    """
    torch.manual_seed(3)
    data = generate(8, 6, [3, 3, 3, 3], "MIX", .9, .4, .1)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if float(torch.linalg.det(q)) < 0:
        q[:, 0] = -q[:, 0]
    r = q.to(torch.float32).numpy()
    transformed = dict(data)
    transformed["common"] = np.einsum("ij,nljk,mk->nlim", r, data["common"], r).astype(np.float32)
    transformed["leak"] = np.einsum("ij,nljk,mk->nlim", r, data["leak"], r).astype(np.float32)
    transformed["noise"] = np.einsum("ij,nljm->nlim", r, data["noise"]).astype(np.float32)
    frame = torch.from_numpy(r)
    report = {}
    for kind in ("CAND", "EXTRA_SUP"):
        model = build_model(kind, 9)
        with torch.no_grad():
            base = model(*tensors(data), train=True)["logits"]
            rotated = model(*tensors(transformed), init_frame=frame, train=True)["logits"]
        error = float((base - rotated).abs().max())
        report[kind + "_full_path_gauge"] = dict(max_abs_error=error, pass_=error <= 1e-5)
    control = build_model("INV_COMP", 9)
    h = radial(torch.from_numpy(data["common"][:, 1]) @ torch.ones(8, 3, 2))
    u = torch.from_numpy(data["commands"][:, 0])
    with torch.no_grad():
        j0 = control.transition(h, u)
        j1 = control.transition(frame @ h, u)
        error = float((j0 - j1).abs().max())
    report["INV_COMP_canonical_update"] = dict(max_abs_error=error, pass_=error <= 1e-5)
    return dict(check="common_rotation_gauge", pass_=all(v["pass_"] for v in report.values()), detail=report)


def check_prefix_consistency() -> dict[str, Any]:
    data = generate(8, 6, [4, 4, 4, 4], "MIX", .4, 0., 0.)
    changed = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    changed["commands"][:, 3:] = 1 - changed["commands"][:, 3:]
    changed["targets"] = np.bitwise_xor.accumulate(
        np.column_stack((data["b0"], changed["commands"])), axis=1).astype(np.float32)
    report = {}
    for kind in KINDS:
        model = build_model(kind, 13)
        a = model(*tensors(data), train=True)["logits"]
        b = model(*tensors(changed), train=True)["logits"]
        error = float((a[:, :4] - b[:, :4]).abs().max())
        report[kind] = dict(max_abs_error=error, pass_=error == 0.)
    return dict(check="prefix_consistency", pass_=all(v["pass_"] for v in report.values()), detail=report)


def check_determinism() -> dict[str, Any]:
    data = generate(8, 5, [5, 5, 5, 5], "MIX", .4, .2, .2)
    report = {}
    for kind in KINDS:
        hashes = []
        for _ in range(2):
            model = build_model(kind, 17)
            hashes.append(hashlib.sha256(model(*tensors(data), train=True)["logits"].detach().numpy().tobytes()).hexdigest())
        report[kind] = dict(pass_=hashes[0] == hashes[1], logits_sha256=hashes[0])
    return dict(check="determinism", pass_=all(v["pass_"] for v in report.values()), detail=report)


def check_single_state() -> dict[str, Any]:
    expected = {"encoder.weight", "readout.weight", "readout.bias",
                "controller.0.weight", "controller.0.bias", "controller.2.weight", "controller.2.bias"}
    detail = {}
    for kind in KINDS:
        model = build_model(kind, 19)
        names = set(dict(model.named_parameters()))
        extra = names - (expected | {f"corrector.{i}.{n}" for i in (0, 2) for n in ("weight", "bias")}
                         | {f"mlps.{b}.{i}.{n}" for b in range(3) for i in (0, 2) for n in ("weight", "bias")})
        buffers = list(model.named_buffers())
        detail[kind] = dict(unknown_parameters=sorted(extra), buffers=sorted(n for n, _ in buffers),
                            pass_=not extra and not buffers)
    return dict(check="single_recursive_state", pass_=all(v["pass_"] for v in detail.values()), detail=detail)


def check_tiny_fit() -> dict[str, Any]:
    data = generate(4, 1, [6, 6, 6, 6], "MIX", 0., 0., 0.)
    report = {}
    for kind in KINDS:
        out = fit(kind, 23, dict(leak=0., sigma=0.), cfg=_tiny_cfg(), updates=50,
                  train_data_override=data, lr=1e-2)
        losses = out["record"]["losses"]
        drop = (losses[0] - float(np.mean(losses[-5:]))) / losses[0]
        report[kind] = dict(first_loss=losses[0], last5_mean=float(np.mean(losses[-5:])),
                            drop=drop, pass_=drop >= .10)
    return dict(check="tiny_fit", pass_=all(v["pass_"] for v in report.values()), detail=report)


def _tiny_cfg() -> dict[str, Any]:
    return dict(training=dict(batch_size=4, lr=1e-2, betas=[.9, .999], eps=1e-8, clip=1.,
                              common_amplitude=0., branch_weight=.1),
                streams=dict(train_namespace=1))


def engineering_checks(cfg: dict[str, Any]) -> dict[str, Any]:
    checks = [check_shapes_finite(), check_gradient_flow(), check_gauge_invariance(),
              check_prefix_consistency(), check_determinism(), check_single_state(), check_tiny_fit()]
    parity = generate(32, 6, [7, 7, 7, 7], "MIX", .4, .3, 0.)
    reference = exp0020_generate(32, 6, [7, 7, 7, 7], "MIX", .4, .3)
    keys = ("b0", "commands", "common", "leak", "targets")
    identical = all(np.array_equal(parity[k], reference[k]) for k in keys)
    checks.append(dict(check="exp0020_generator_parity_sigma0", pass_=bool(identical),
                       detail=dict(keys=list(keys), identical=identical)))
    return dict(checks=checks, pass_=all(c["pass_"] for c in checks), config=cfg)


# ---------------------------------------------------------------------------
# Stimuli
# ---------------------------------------------------------------------------

def condition_streams(cfg: dict[str, Any], condition: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    ns = cfg["streams"]["eval_namespace"]
    length, leak, sigma = condition["length"], condition["leak"], condition["sigma"]
    n = cfg["streams"]["eval_examples"]
    return dict(primary=generate(n, length, [ns, length, 1, 0], "MIX", PI, leak, sigma),
                secondary=generate(n, length, [ns, length, 1, 0], "MIX", .4, leak, sigma))


def parse_menu(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in config["menu"]]


# ---------------------------------------------------------------------------
# Run phases
# ---------------------------------------------------------------------------

def run_engineering(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    report = engineering_checks(cfg)
    save_json(out / "engineering" / "checks.json", report)
    return report


def run_scan(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    band = cfg["band"]
    results = []
    qualified = None
    for condition in parse_menu(cfg):
        streams = condition_streams(cfg, condition)
        seeds = []
        for seed in cfg["seeds"]["scan"]:
            out_fit = fit("INV_COMP", seed, condition, cfg, cfg["training"]["updates"], width=32, blocks=1)
            record = out_fit["record"]
            record["primary"] = metrics_for(out_fit["model"], streams["primary"])
            record["secondary"] = metrics_for(out_fit["model"], streams["secondary"])
            seeds.append(record)
            save_json(out / "scan" / f"{condition['id']}_seed{seed}.json", record)
        flip = [r["primary"]["flip_accuracy"] for r in seeds]
        hold = [r["primary"]["hold_accuracy"] for r in seeds]
        passed = (band["flip_mean"][0] <= float(np.mean(flip)) <= band["flip_mean"][1]
                  and all(band["flip_seed"][0] <= x <= band["flip_seed"][1] for x in flip)
                  and float(np.mean(hold)) >= band["hold_mean_min"])
        entry = dict(condition=condition, seeds=seeds, flip=flip, hold=hold,
                     flip_mean=float(np.mean(flip)), hold_mean=float(np.mean(hold)), qualified=bool(passed))
        results.append(entry)
        if passed and qualified is None:
            qualified = condition
    summary = dict(results=results, qualified_condition=qualified, config=cfg)
    save_json(out / "scan" / "summary.json", summary)
    return summary


def run_match(cfg: dict[str, Any], out: Path, qualified: dict[str, Any]) -> dict[str, Any]:
    band = cfg["band"]
    streams = condition_streams(cfg, qualified)
    cand_ops = op_estimate("CAND")["inference"]
    results = []
    for width, blocks in cfg["control_menu"]:
        seeds = []
        for seed in cfg["seeds"]["scan"]:
            out_fit = fit("INV_COMP", seed, qualified, cfg, cfg["training"]["updates"], width=width, blocks=blocks)
            record = out_fit["record"]
            record["primary"] = metrics_for(out_fit["model"], streams["primary"])
            seeds.append(record)
            save_json(out / "match" / f"w{width}_b{blocks}_seed{seed}.json", record)
        flip = [r["primary"]["flip_accuracy"] for r in seeds]
        hold = [r["primary"]["hold_accuracy"] for r in seeds]
        learned = (float(np.mean(flip)) >= band["learn_flip_mean_min"]
                   and all(x >= band["learn_flip_seed_min"] for x in flip)
                   and float(np.mean(hold)) >= band["learn_hold_mean_min"])
        ops = op_estimate("INV_COMP", width, blocks)["inference"]
        results.append(dict(width=width, blocks=blocks, seeds=seeds, flip=flip, hold=hold,
                            flip_mean=float(np.mean(flip)), hold_mean=float(np.mean(hold)),
                            learned=bool(learned), inference_ops=ops, parameters=param_count("INV_COMP", width, blocks),
                            log_ratio=float(abs(np.log(ops / cand_ops)))))
    eligible = [r for r in results if r["learned"]]
    selected = None
    if eligible:
        selected = min(eligible, key=lambda r: (r["log_ratio"], r["parameters"]))
    summary = dict(condition=qualified, candidate_inference_ops=cand_ops, results=results,
                   selected=(dict(width=selected["width"], blocks=selected["blocks"],
                                  inference_ops=selected["inference_ops"], parameters=selected["parameters"],
                                  log_ratio=selected["log_ratio"]) if selected else None),
                   control_ops_ratio=(selected["inference_ops"] / cand_ops if selected else None))
    save_json(out / "match" / "summary.json", summary)
    return summary


def run_learn(cfg: dict[str, Any], out: Path, qualified: dict[str, Any],
              selected: dict[str, Any] | None) -> dict[str, Any]:
    band = cfg["band"]
    streams = condition_streams(cfg, qualified)
    arms = [("CAND", 32, 1), ("EXTRA_SUP", 32, 1)]
    if selected is not None:
        arms.append(("INV_COMP", selected["width"], selected["blocks"]))
    results = []
    for kind, width, blocks in arms:
        seeds = []
        for seed in cfg["seeds"]["learn"]:
            out_fit = fit(kind, seed, qualified, cfg, cfg["training"]["updates"], width=width, blocks=blocks)
            record = out_fit["record"]
            record["primary"] = metrics_for(out_fit["model"], streams["primary"])
            record["secondary"] = metrics_for(out_fit["model"], streams["secondary"])
            predictions = evaluate(out_fit["model"], streams["primary"])
            np.savez_compressed(out / "learn" / f"{record['tag']}_seed{seed}_primary.npz",
                                logits=predictions["logits"].numpy(), branch=predictions["branch"].numpy())
            record["predictions_sha256"] = sha(out / "learn" / f"{record['tag']}_seed{seed}_primary.npz")
            seeds.append(record)
            save_json(out / "learn" / f"{record['tag']}_seed{seed}.json", record)
        flip = [r["primary"]["flip_accuracy"] for r in seeds]
        hold = [r["primary"]["hold_accuracy"] for r in seeds]
        learned = (float(np.mean(flip)) >= band["learn_flip_mean_min"]
                   and all(x >= band["learn_flip_seed_min"] for x in flip)
                   and float(np.mean(hold)) >= band["learn_hold_mean_min"])
        results.append(dict(kind=kind, tag=model_tag(kind, width, blocks), seeds=seeds, flip=flip, hold=hold,
                            flip_mean=float(np.mean(flip)), hold_mean=float(np.mean(hold)), learned=bool(learned)))
    summary = dict(condition=qualified, results=results, config_seeds=cfg["seeds"]["learn"],
                   all_learned=all(r["learned"] for r in results))
    save_json(out / "learn" / "summary.json", summary)
    return summary


def replay_learning_seed(cfg: dict[str, Any], qualified: dict[str, Any], seed: int,
                         kind: str = "CAND", width: int = 32, blocks: int = 1) -> dict[str, Any]:
    streams = condition_streams(cfg, qualified)
    out_fit = fit(kind, seed, qualified, cfg, cfg["training"]["updates"], width=width, blocks=blocks)
    metrics = metrics_for(out_fit["model"], streams["primary"])
    return dict(kind=kind, seed=seed, final_hash=out_fit["record"]["final_hash"],
                training_data_sha256=out_fit["record"]["training_data_sha256"],
                training_seconds=out_fit["record"]["seconds"], metrics=metrics)


def run_replay(cfg: dict[str, Any], out: Path, qualified: dict[str, Any], seed: int = 28111) -> dict[str, Any]:
    replay = replay_learning_seed(cfg, qualified, seed)
    record = json.loads((out / "learn" / f"CAND_seed{seed}.json").read_text())
    gaps = {k: abs(record["primary"][k] - replay["metrics"][k]) for k in
            ("flip_accuracy", "hold_accuracy", "accuracy_all", "bce")}
    detail = dict(final_hash_equal=record["final_hash"] == replay["final_hash"],
                  data_hash_equal=record["training_data_sha256"] == replay["training_data_sha256"],
                  logits_hash_equal=record["primary"]["logits_sha256"] == replay["metrics"]["logits_sha256"],
                  metric_gaps=gaps)
    detail["pass_"] = (detail["final_hash_equal"] and detail["data_hash_equal"]
                       and detail["logits_hash_equal"] and all(v <= 1e-8 for v in gaps.values()))
    summary = dict(replay_of_seed=seed, pass_=bool(detail["pass_"]), detail=detail,
                   replay_metrics=replay["metrics"], replay_seconds=replay["training_seconds"])
    save_json(out / "replay.json", summary)
    return summary


def run_pretest(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    ns = cfg["streams"]["pretest_namespace"]
    cells = {k: generate(cfg["streams"]["eval_examples"], v[0], [ns, v[0], (1 if v[1] == "HOLD" else 2), 0],
                         v[1], v[2], v[3]) for k, v in cfg["pretest_cells"].items()}
    results = []
    for kind, width, blocks in (("CAND", 32, 1), ("INV_COMP", 32, 1)):
        seeds = []
        for seed in cfg["seeds"]["pretest"]:
            out_fit = fit(kind, seed, dict(leak=0., sigma=0.), cfg, cfg["training"]["updates"], width=width, blocks=blocks)
            record = out_fit["record"]
            record["cells"] = {k: metrics_for(out_fit["model"], v) for k, v in cells.items()}
            seeds.append(record)
            save_json(out / "pretest" / f"{record['tag']}_seed{seed}.json", record)
        flags = [bool(r["cells"]["IID_H"]["accuracy_all"] >= .95 and r["cells"]["IID_U"]["accuracy_all"] >= .95
                      and r["cells"]["IID_U"]["flip_accuracy"] >= .95) for r in seeds]
        results.append(dict(kind=kind, tag=model_tag(kind, width, blocks), seeds=seeds,
                            iid8_learned_flags=flags, learned=all(flags)))
    summary = dict(results=results, note="IID8 pretest only; cannot grant A1 under ARCH-GATES-v1")
    save_json(out / "pretest" / "summary.json", summary)
    return summary


def analyze(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    eng = json.loads((out / "engineering" / "checks.json").read_text()) if (out / "engineering" / "checks.json").exists() else None
    pretest = json.loads((out / "pretest" / "summary.json").read_text()) if (out / "pretest" / "summary.json").exists() else None
    scan = json.loads((out / "scan" / "summary.json").read_text()) if (out / "scan" / "summary.json").exists() else None
    match = json.loads((out / "match" / "summary.json").read_text()) if (out / "match" / "summary.json").exists() else None
    learn = json.loads((out / "learn" / "summary.json").read_text()) if (out / "learn" / "summary.json").exists() else None
    replay = json.loads((out / "replay.json").read_text()) if (out / "replay.json").exists() else None
    if eng is None or not eng["pass_"]:
        decision, substatus = "INCONCLUSIVE", "ENGINEERING_FAILURE"
    elif scan is None or scan["qualified_condition"] is None:
        decision, substatus = "INCONCLUSIVE", "NO_QUALIFIED_CONDITION"
    elif match is None or match.get("selected") is None:
        decision, substatus = "INCONCLUSIVE", "CONTROL_QUALIFICATION_FAILED"
    elif learn is None or not learn["all_learned"]:
        decision, substatus = "INCONCLUSIVE", "LEARNING_QUALIFICATION_FAILED"
    elif replay is None or not replay["pass_"]:
        decision, substatus = "INCONCLUSIVE", "REPLAY_FAILED"
    else:
        decision, substatus = "SUPPORTED", "A1_RUNNABLE"
    analysis = dict(experiment_id="EXP-0028", version="v1", gate="A1", decision=decision, substatus=substatus,
                    engineering_pass=bool(eng and eng["pass_"]), pretest=pretest, scan=scan, match=match,
                    learn=learn, replay=replay, updates=_count_updates(out),
                    scope="A1 runnable computation only; no A2 effect, no old Gate1 credit.")
    save_json(out / "analysis.json", analysis)
    return analysis


def _count_updates(out: Path) -> int:
    total = 0
    for path in out.rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and "updates" in data and isinstance(data["updates"], int):
            total += data["updates"]
    return total


def manifest(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    status = git_output("status", "--porcelain")
    return dict(experiment_id="EXP-0028", version="v1", gate="A1", config=cfg,
                git_commit=git_output("rev-parse", "HEAD"), git_dirty=bool(status.strip()),
                git_status=status, source_sha256=source_fingerprints(),
                frozen_hashes={str(p): sha(p) for p in (config_path, Path(cfg["protocol"]),
                                                        Path(cfg["candidate_contract"]), Path("uv.lock"),
                                                        Path("artifacts/validation/suite.json"),
                                                        Path("src/neurotopo/arch0001_core.py"),
                                                        Path("tests/test_arch0001_core.py"))},
                python=sys.version, torch=torch.__version__, numpy=np.__version__,
                platform=platform.platform(), scope="A1 runnable computation only")


def run(config_path: Path, output: Path, phases: str, replay_seed: int = 28111) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text())
    validate_config(cfg)
    suite = Path("artifacts/validation/suite.json")
    if not suite.exists():
        raise RuntimeError("Run scripts/verify.py before the registered EXP-0028 phases")
    payload = json.loads(suite.read_text())
    if not payload["pass"] or payload["source_sha256"] != source_fingerprints():
        raise RuntimeError("Validation suite does not match current sources; re-run scripts/verify.py")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        manifest_data = json.loads((output / "manifest.json").read_text())
    else:
        manifest_data = manifest(cfg, config_path)
        save_json(output / "manifest.json", manifest_data)
    started = time.monotonic()
    todo = ["eng", "pretest", "scan", "match", "learn", "replay"] if phases == "all" else phases.split(",")
    scan = json.loads((output / "scan" / "summary.json").read_text()) if (output / "scan" / "summary.json").exists() else None
    match = json.loads((output / "match" / "summary.json").read_text()) if (output / "match" / "summary.json").exists() else None
    for phase in todo:
        if time.monotonic() - started > cfg["resource"]["wall_seconds_cap"]:
            raise TimeoutError("Shared EXP-0028 wall budget")
        if phase == "eng":
            run_engineering(cfg, output)
        elif phase == "pretest":
            run_pretest(cfg, output)
        elif phase == "scan":
            scan = run_scan(cfg, output)
        elif phase == "match":
            if scan is None:
                scan = json.loads((output / "scan" / "summary.json").read_text())
            if scan["qualified_condition"] is not None:
                match = run_match(cfg, output, scan["qualified_condition"])
        elif phase == "learn":
            if scan is None:
                scan = json.loads((output / "scan" / "summary.json").read_text())
            if match is None and (output / "match" / "summary.json").exists():
                match = json.loads((output / "match" / "summary.json").read_text())
            if scan["qualified_condition"] is not None and match is not None and match.get("selected") is not None:
                run_learn(cfg, output, scan["qualified_condition"], match["selected"])
        elif phase == "replay":
            if scan is None:
                scan = json.loads((output / "scan" / "summary.json").read_text())
            if scan["qualified_condition"] is not None:
                run_replay(cfg, output, scan["qualified_condition"], replay_seed)
        else:
            raise ValueError(phase)
    if phases == "all":
        return analyze(cfg, output)
    return dict(phases=todo)


def validate_config(c: dict[str, Any]) -> None:
    assert c["experiment_id"] == "EXP-0028" and c["version"] == "v1" and c["gate"] == "A1"
    assert c["candidate"] == "SC-REL-v3"
    assert c["control_menu"] == [[32, 1], [32, 2], [32, 3], [48, 1], [48, 2], [48, 3],
                                 [64, 1], [64, 2], [64, 3], [96, 1], [96, 2], [96, 3]]
    assert c["training"] == dict(updates=1000, batch_size=64, lr=.003, betas=[.9, .999], eps=1e-8,
                                 clip=1., common_amplitude=.4, branch_weight=.1,
                                 curriculum=[[125, 1], [250, 2], [500, 4], [1000, 8]])
    assert c["seeds"] == dict(pretest=[28101, 28102, 28103], scan=[28101, 28102], learn=[28111, 28112, 28113])
    assert c["streams"] == dict(train_namespace=282001, eval_namespace=283001, pretest_namespace=285001,
                                eval_examples=1024, monitor_examples=128)
    assert [m["id"] for m in c["menu"]] == [f"N{i}_{name}" for i, name in
                                           enumerate(["T64_leak0_s0", "T256_leak4_s0", "T64_leak0_s05",
                                                      "T64_leak0_s15", "T64_leak0_s40", "T256_leak4_s05",
                                                      "T256_leak4_s15", "T256_leak4_s40"], start=1)]
    assert [(m["length"], m["leak"], m["sigma"]) for m in c["menu"]] == [
        (64, 0., 0.), (256, .4, 0.), (64, 0., .05), (64, 0., .15), (64, 0., .4),
        (256, .4, .05), (256, .4, .15), (256, .4, .4)]
    assert c["band"] == dict(flip_mean=[.6, .95], flip_seed=[.55, .97], hold_mean_min=.6,
                             learn_flip_mean_min=.6, learn_flip_seed_min=.55, learn_hold_mean_min=.6)
    assert c["resource"] == dict(max_updates=60000, wall_seconds_cap=1800, replay_wall_seconds_cap=600)
    assert np.__version__ == "2.3.3" and torch.__version__.split("+")[0] == "2.8.0"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phases", default="all")
    parser.add_argument("--replay-seed", type=int, default=28111)
    args = parser.parse_args()
    result = run(args.config, args.output, args.phases, args.replay_seed)
    print(json.dumps({k: v for k, v in result.items() if k not in ("pretest", "scan", "match", "learn")},
                     ensure_ascii=False, default=str))
