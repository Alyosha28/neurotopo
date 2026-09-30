"""Engineering tests for EXP0032 (control-menu and budget sweep); see experiments/EXP-0032/v1.md."""
from __future__ import annotations

import json
from pathlib import Path

from neurotopo import arch0001_control_sweep as sweep

CONFIG = Path("configs/exp0032_v1.json")


def test_config_and_protocol() -> None:
    cfg = json.loads(CONFIG.read_text())
    sweep.validate_config(cfg)
    assert Path(cfg["protocol"]).exists()


def _record(d: int, width: int, blocks: int, acc: float) -> dict:
    return dict(d=d, width=width, blocks=blocks, parameters=sweep.param_count("INV_COMP", width, blocks),
                inference_ops=sweep.op_estimate("INV_COMP", width, blocks)["inference"],
                primary=dict(meaningful_accuracy=acc))


def test_band_and_selection_rule() -> None:
    cfg = json.loads(CONFIG.read_text())
    assert sweep.in_band(cfg, _record(2, 32, 1, .65)) is True
    assert sweep.in_band(cfg, _record(2, 32, 1, .64)) is False
    assert sweep.in_band(cfg, _record(2, 32, 1, .96)) is False
    # d=2 has no in-band config, d=3 has two: pick the compute-closest to the candidate (1161 ops)
    records = [_record(2, 32, 1, .60), _record(2, 32, 2, .64),
               _record(3, 96, 1, .80), _record(3, 64, 2, .90)]
    chosen = sweep.select(cfg, records)
    assert chosen is not None
    d, record = chosen
    assert d == 3 and (record["width"], record["blocks"]) == (96, 1)
    assert sweep.select(cfg, [_record(2, 32, 1, .60)]) is None


def test_first_d_preference() -> None:
    cfg = json.loads(CONFIG.read_text())
    records = [_record(2, 96, 1, .80), _record(3, 32, 1, .70)]
    d, _ = sweep.select(cfg, records)
    assert d == 2


def test_curriculum_switch_for_budget() -> None:
    cfg = json.loads(CONFIG.read_text())
    c2000 = sweep.training_cfg(cfg, 2000, cfg["stage_a"]["curriculum"])
    c6000 = sweep.training_cfg(cfg, 6000, cfg["stage_b"]["curriculum"])
    assert c2000["training"]["updates"] == 2000 and c6000["training"]["updates"] == 6000
    assert c2000["training"]["curriculum"][-1] == [2000, 12] and c6000["training"]["curriculum"][-1] == [6000, 12]
    assert c2000["training"]["common_amplitude"] == cfg["training"]["common_amplitude"]


def test_control_config_reaches_the_model() -> None:
    """Regression for the EXP0032 v1 defect: the swept width/blocks must reach fit_memory's model."""
    from neurotopo.arch0001_memory_task import fit_memory
    cfg = json.loads(CONFIG.read_text())
    c = sweep.training_cfg(cfg, 1, [[1, 2]], width=96, blocks=3)
    assert c["control"] == dict(kind="INV_COMP", width=96, blocks=3)
    out = fit_memory(29101, 2, "PLAIN", c, updates=1)
    assert sum(p.numel() for p in out["model"].parameters()) == sweep.param_count("INV_COMP", 96, 3)
    small = sweep.training_cfg(cfg, 1, [[1, 2]], width=32, blocks=1)
    out_small = fit_memory(29101, 2, "PLAIN", small, updates=1)
    assert sum(p.numel() for p in out_small["model"].parameters()) == sweep.param_count("INV_COMP", 32, 1)
