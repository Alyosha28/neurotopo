"""Engineering tests for EXP0028 / ARCH-0001 A1 (see experiments/EXP-0028/v1.md)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neurotopo import arch0001_core as core
from neurotopo.learned_protected_pilot import generate as exp0020_generate

CONFIG = Path("configs/exp0028_v1.json")


def test_config_and_protocol_exist() -> None:
    cfg = json.loads(CONFIG.read_text())
    core.validate_config(cfg)
    assert Path(cfg["protocol"]).exists() and Path(cfg["candidate_contract"]).exists()


def test_exp0020_generator_parity_when_sigma_zero() -> None:
    for length, mode, common, leak in ((8, "MIX", .4, 0.), (64, "HOLD", 3.141592653589793, 0.),
                                       (64, "UPDATE", 3.141592653589793, .4)):
        new = core.generate(8, length, [11, 12, 13, mode == "UPDATE"], mode, common, leak, 0.)
        old = exp0020_generate(8, length, [11, 12, 13, mode == "UPDATE"], mode, common, leak)
        for key in ("b0", "commands", "common", "leak", "targets"):
            assert np.array_equal(new[key], old[key]), (length, mode, key)
        assert np.all(new["noise"] == 0.)


def test_noise_axis_is_deterministic_and_scales() -> None:
    a = core.generate(8, 16, [21, 22, 23, 1], "MIX", .4, 0., .3)
    b = core.generate(8, 16, [21, 22, 23, 1], "MIX", .4, 0., .3)
    assert np.array_equal(a["noise"], b["noise"])
    assert np.allclose(a["noise"] / .3, core.generate(8, 16, [21, 22, 23, 1], "MIX", .4, 0., 1.)["noise"])
    assert np.array_equal(a["commands"], core.generate(8, 16, [21, 22, 23, 1], "MIX", .4, 0., 0.)["commands"])


def test_shapes_calls_and_finite() -> None:
    data = core.generate(8, 4, [31, 32, 33, 1], "MIX", .4, .4, .2)
    for kind in core.KINDS:
        model = core.build_model(kind, 7)
        parts = model(*core.tensors(data), train=True)
        assert tuple(parts["logits"].shape) == (8, 5)
        assert tuple(parts["branch"].shape) == (8, 5, 2)
        assert torch.isfinite(parts["logits"]).all() and torch.isfinite(parts["branch"]).all()
        assert model.calls == (5 * 4 + 2 if kind != "INV_COMP" else 3 * 4 + 2)


def test_gradient_flow_two_steps_and_zero_init_corrector() -> None:
    data = core.generate(16, 2, [41, 42, 43, 1], "MIX", 0., 0., 0.)
    for kind in core.KINDS:
        model = core.build_model(kind, 5)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        seen = {name: [] for name, _ in model.named_parameters()}
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss, _ = core.batch_loss(model(*core.tensors(data), train=True), data)
            loss.backward()
            for name, p in model.named_parameters():
                g = p.grad
                seen[name].append(bool(g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0))
            opt.step()
        assert all(v[1] for v in seen.values()), (kind, {k: v for k, v in seen.items() if not v[1]})


def test_common_rotation_gauge_invariance() -> None:
    torch.manual_seed(3)
    data = core.generate(8, 6, [51, 52, 53, 1], "MIX", .9, .4, .1)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if float(torch.linalg.det(q)) < 0:
        q[:, 0] = -q[:, 0]
    r = q.to(torch.float32).numpy()
    transformed = dict(data)
    transformed["common"] = np.einsum("ij,nljk,mk->nlim", r, data["common"], r).astype(np.float32)
    transformed["leak"] = np.einsum("ij,nljk,mk->nlim", r, data["leak"], r).astype(np.float32)
    transformed["noise"] = np.einsum("ij,nljm->nlim", r, data["noise"]).astype(np.float32)
    frame = torch.from_numpy(r)
    for kind in ("CAND", "EXTRA_SUP"):
        model = core.build_model(kind, 9)
        with torch.no_grad():
            base = model(*core.tensors(data), train=True)["logits"]
            rotated = model(*core.tensors(transformed), init_frame=frame, train=True)["logits"]
        assert float((base - rotated).abs().max()) <= 1e-5, kind
    control = core.build_model("INV_COMP", 9)
    h = core.radial(torch.from_numpy(data["common"][:, 1]) @ torch.ones(8, 3, 2))
    u = torch.from_numpy(data["commands"][:, 0])
    with torch.no_grad():
        assert float((control.transition(h, u) - control.transition(frame @ h, u)).abs().max()) <= 1e-5


def test_prefix_consistency_and_single_state() -> None:
    data = core.generate(8, 6, [61, 62, 63, 1], "MIX", .4, 0., 0.)
    changed = {k: v.copy() for k, v in data.items()}
    changed["commands"][:, 3:] = 1 - changed["commands"][:, 3:]
    changed["targets"] = np.bitwise_xor.accumulate(
        np.column_stack((changed["b0"], changed["commands"])), axis=1).astype(np.float32)
    for kind in core.KINDS:
        model = core.build_model(kind, 13)
        a = model(*core.tensors(data), train=True)["logits"]
        b = model(*core.tensors(changed), train=True)["logits"]
        assert float((a[:, :4] - b[:, :4]).abs().max()) == 0.
        assert not list(model.named_buffers())


def test_parameter_and_operator_counts_match_exp0020_references() -> None:
    assert core.param_count("CAND") == 664
    assert core.param_count("EXTRA_SUP") == 340
    assert core.param_count("INV_COMP", 32, 1) == 406
    assert core.op_estimate("CAND")["inference"] == 1161
    assert core.op_estimate("INV_COMP", 32, 1)["inference"] == 355


def test_metrics_and_loss_finite() -> None:
    data = core.generate(16, 4, [71, 72, 73, 2], "MIX", .4, .2, .1)
    model = core.build_model("CAND", 11)
    parts = model(*core.tensors(data), train=True)
    loss, terms = core.batch_loss(parts, data)
    assert torch.isfinite(loss) and terms["l_actual"] > 0 and terms["l_branch"] > 0
    metrics = core.condition_metrics(parts, data)
    for key in ("flip_accuracy", "hold_accuracy", "response_error", "current_ok_next_response_bad"):
        assert metrics[key] is not None
    assert 0. <= metrics["flip_accuracy"] <= 1. and 0. <= metrics["response_error"] <= 1.


def test_engineering_check_suite_passes() -> None:
    cfg = json.loads(CONFIG.read_text())
    report = core.engineering_checks(cfg)
    failed = [c["check"] for c in report["checks"] if not c["pass_"]]
    assert report["pass_"], failed
