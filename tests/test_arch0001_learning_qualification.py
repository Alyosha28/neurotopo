"""Engineering tests for EXP0029 (learning-qualification replication); see experiments/EXP-0029/v1.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurotopo import arch0001_learning_qualification as lq

CONFIG = Path("configs/exp0029_v1.json")


def test_config_and_protocol_exist() -> None:
    cfg = json.loads(CONFIG.read_text())
    lq.validate_config(cfg)
    assert Path(cfg["protocol"]).exists() and Path(cfg["candidate_contract"]).exists()


def test_frozen_candidate_module_matches_exp0028_manifest() -> None:
    manifest = json.loads(Path("artifacts/EXP-0028/v1/manifest.json").read_text())
    frozen = manifest["frozen_hashes"]["src/neurotopo/arch0001_core.py"]
    from neurotopo.order_observation import sha
    assert sha(Path("src/neurotopo/arch0001_core.py")) == frozen


def test_stream_for_is_deterministic_and_seed_dependent() -> None:
    cfg = json.loads(CONFIG.read_text())
    cell = cfg["cells"]["IID_U"]
    a = lq.stream_for(cfg, 28201, cell)
    b = lq.stream_for(cfg, 28201, cell)
    c = lq.stream_for(cfg, 28202, cell)
    for key in ("b0", "commands", "common", "targets"):
        assert np.array_equal(a[key], b[key])
        assert not np.array_equal(a[key], c[key])


def test_loss_checkpoints_windows() -> None:
    losses = [float(i) for i in range(1000)]
    points = lq.loss_checkpoints(losses)
    assert points["125"] == pytest.approx(np.mean(range(100, 125)))
    assert points["1000"] == pytest.approx(np.mean(range(975, 1000)))


def test_failure_classification_table() -> None:
    thresholds = json.loads(CONFIG.read_text())["thresholds"]

    def cells(hold: float, flip: float) -> dict:
        return {"IID_H": {"accuracy_all": hold}, "IID_U": {"accuracy_all": .5, "flip_accuracy": flip}}

    assert lq.classify_failure(cells(1.0, .50), thresholds) == "HOLD_COLLAPSE"
    assert lq.classify_failure(cells(1.0, .80), thresholds) == "PARTIAL_FIT"
    assert lq.classify_failure(cells(.60, .50), thresholds) == "TASK_LEARNING_BROKEN"
    assert lq.classify_failure(cells(1.0, .999), thresholds) == "UNCLASSIFIED"


def test_learning_flag_matches_registered_thresholds() -> None:
    thresholds = json.loads(CONFIG.read_text())["thresholds"]
    good = {"IID_H": {"accuracy_all": .96}, "IID_U": {"accuracy_all": .96, "flip_accuracy": .96}}
    bad = {"IID_H": {"accuracy_all": 1.0}, "IID_U": {"accuracy_all": .96, "flip_accuracy": .50}}
    assert lq.learning_flag(good, thresholds)
    assert not lq.learning_flag(bad, thresholds)


def _summary(cand_passed: int, control_passed: int, classes: dict, engineering: bool = False) -> dict:
    rows = [{"failure_class": "HOLD_COLLAPSE" if engineering is False else "ENGINEERING"}] if engineering else []
    return dict(arms={
        "CAND": dict(passed=cand_passed, failure_classes=classes, rows=rows),
        "EXTRA_SUP": dict(passed=12, failure_classes={}, rows=[]),
        "INV_COMP-w32-b1": dict(passed=control_passed, failure_classes={}, rows=[]),
    })


def test_decision_table_rows() -> None:
    cfg = json.loads(CONFIG.read_text())
    assert lq.decide(cfg, _summary(12, 12, {})) == ("PARTIALLY_SUPPORTED", "LEARNING_QUALIFICATION_AT_BUDGET")
    assert lq.decide(cfg, _summary(11, 12, {28203: "HOLD_COLLAPSE"})) == (
        "PARTIALLY_SUPPORTED", "LEARNING_QUALIFICATION_AT_BUDGET")
    assert lq.decide(cfg, _summary(11, 12, {28203: "PARTIAL_FIT"})) == (
        "INCONCLUSIVE", "FAILURE_MODE_UNRESOLVED")
    assert lq.decide(cfg, _summary(9, 12, {s: "HOLD_COLLAPSE" for s in range(28201, 28204)})) == (
        "INCONCLUSIVE", "CANDIDATE_LEARNING_UNRELIABLE")
    assert lq.decide(cfg, _summary(12, 9, {})) == ("INCONCLUSIVE", "CONTROL_LEARNING_UNRELIABLE")
    assert lq.decide(cfg, _summary(12, 12, {}, engineering=True)) == ("INCONCLUSIVE", "ENGINEERING_FAILURE")
