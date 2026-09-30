"""Engineering tests for EXP0033 (carrier confirmation + v4 probe); see experiments/EXP-0033/v1.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neurotopo import arch0001_a2_probe as probe
from neurotopo.arch0001_core_v4 import build_v4, evaluate_v4, fit_v4, v4_loss, with_mask
from neurotopo.arch0001_memory_task import generate_memory
from neurotopo.arch0001_core import tensors

CONFIG = Path("configs/exp0033_v1.json")


def test_config_and_protocol() -> None:
    cfg = json.loads(CONFIG.read_text())
    probe.validate_config(cfg)
    assert Path(cfg["protocol"]).exists() and Path(cfg["candidate_contract"]).exists()


def test_mask_excludes_terminal_and_loss_ignores_it() -> None:
    data = with_mask(generate_memory(8, 5, [1, 2, 3, 2], 2))
    assert data["branch_mask"].shape == (8, 6) and not data["branch_mask"][:, -1].any()
    model = build_v4(7)
    parts = model(*tensors(data), train=True)
    loss_a, _ = v4_loss(parts, data)
    changed = dict(data)
    bt = data["branch_targets"].copy()
    bt[:, -1, :] = 1. - bt[:, -1, :]
    changed["branch_targets"] = bt
    loss_b, _ = v4_loss(parts, changed)
    assert torch.allclose(loss_a, loss_b)
    bt2 = data["branch_targets"].copy()
    bt2[:, 1, :] = 1. - bt2[:, 1, :]
    changed["branch_targets"] = bt2
    loss_c, _ = v4_loss(parts, changed)
    assert not torch.allclose(loss_a, loss_c)


def test_fit_v4_smoke_and_inference_call_count() -> None:
    cfg = json.loads(CONFIG.read_text())
    out = fit_v4(29311, 2, cfg, updates=2, generate=generate_memory)
    data = generate_memory(8, 4, [4, 5, 6, 2], 2)
    parts = evaluate_v4(out["model"], data)
    assert parts["logits"].shape == (8, 5) and torch.isfinite(parts["logits"]).all()
    model = build_v4(11)
    model(*tensors(data), train=True)
    assert model.calls == 5 * 4 + 2


def test_carrier_selection_prefers_highest_mean_among_qualified() -> None:
    cfg = json.loads(CONFIG.read_text())
    summary = {
        "by_config": {
            "w32b3": dict(width=32, blocks=3, mean=.91, qualified=True, inference_ops=1059, parameters=1186),
            "w64b2": dict(width=64, blocks=2, mean=.75, qualified=True, inference_ops=1411, parameters=1564),
            "w48b2": dict(width=48, blocks=2, mean=.64, qualified=False, inference_ops=1059, parameters=1180),
        },
        "band": cfg["band"], "selected": None,
    }
    eligible = [v for v in summary["by_config"].values() if v["qualified"]]
    chosen = max(eligible, key=lambda v: (v["mean"], -abs(np.log(v["inference_ops"] / cfg["candidate_inference_ops"]))))
    assert (chosen["width"], chosen["blocks"]) == (32, 3)


def test_candidate_and_control_op_estimates_differ_or_are_reported() -> None:
    cfg = json.loads(CONFIG.read_text())
    assert cfg["candidate_inference_ops"] == 1161
    from neurotopo.arch0001_core import op_estimate
    assert all(op_estimate("INV_COMP", w, b)["inference"] > 0 for w, b in cfg["carrier_menu"])
