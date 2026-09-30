"""Engineering tests for EXP0031 (corrected-endpoint re-qualification); see experiments/EXP-0031/v1.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neurotopo import arch0001_memory_requalification as rq

CONFIG = Path("configs/exp0031_v1.json")


def test_config_protocol_and_parent() -> None:
    cfg = json.loads(CONFIG.read_text())
    rq.validate_config(cfg)
    assert Path(cfg["protocol"]).exists()
    assert Path(cfg["parent_config"]).exists() and Path(cfg["parent_artifacts"]).exists()


def test_meaningful_slice_excludes_forced_prefix() -> None:
    assert rq.meaningful_slice(1, 12) == slice(2, 13)
    assert rq.meaningful_slice(3, 12) == slice(4, 13)


def test_references_on_meaningful_positions() -> None:
    cfg = json.loads(CONFIG.read_text())
    for d in (1, 2, 3):
        data = rq.generate_memory(256, 12, [9, d, 12, 0], d)
        scores = rq.reference_scores(data, d, rq.calibration_stream(cfg, d))
        assert set(scores) == {"constant_zero", "constant_one", "pass_through", "position_majority"}
        assert all(0. <= v <= 1. for v in scores.values())
        assert max(scores.values()) <= .60, scores
        assert rq.oracle_score(data, d) == 1.0


def test_constant_zero_is_not_above_band_on_meaningful_positions() -> None:
    cfg = json.loads(CONFIG.read_text())
    band = cfg["band"]
    for d in (1, 2, 3):
        data = rq.generate_memory(1024, 12, [11, d, 12, 0], d)
        scores = rq.reference_scores(data, d, rq.calibration_stream(cfg, d))
        assert max(scores.values()) + band["memoryless_margin_mean"] < band["learn_mean_min"] + .10
        assert scores["constant_zero"] <= .58


def test_tie_break_prefers_plain() -> None:
    cfg = json.loads(CONFIG.read_text())
    records = [dict(d=2, sup="PLAIN", seed=29101, primary=dict(meaningful_accuracy=.70), references={"constant_zero": .5, "constant_one": .5, "pass_through": .5, "position_majority": .5}, oracle=1.0, parent_reproduced=True),
               dict(d=2, sup="PLAIN", seed=29102, primary=dict(meaningful_accuracy=.72), references={"constant_zero": .5, "constant_one": .5, "pass_through": .5, "position_majority": .5}, oracle=1.0, parent_reproduced=True),
               dict(d=2, sup="BRANCH", seed=29101, primary=dict(meaningful_accuracy=.70), references={"constant_zero": .5, "constant_one": .5, "pass_through": .5, "position_majority": .5}, oracle=1.0, parent_reproduced=True),
               dict(d=2, sup="BRANCH", seed=29102, primary=dict(meaningful_accuracy=.72), references={"constant_zero": .5, "constant_one": .5, "pass_through": .5, "position_majority": .5}, oracle=1.0, parent_reproduced=True)]
    summary = rq.summarize_scan(cfg, records)
    assert summary["by_d"][2]["best_sup"] == "PLAIN"
    assert summary["qualified"] == dict(d=2, sup="PLAIN")


def test_band_rejects_reproduction_failure() -> None:
    cfg = json.loads(CONFIG.read_text())
    records = []
    for sup in ("PLAIN", "BRANCH"):
        for seed, acc in ((29101, .80), (29102, .82)):
            records.append(dict(d=2, sup=sup, seed=seed, primary=dict(meaningful_accuracy=acc),
                                references={"constant_zero": .5, "constant_one": .5, "pass_through": .5, "position_majority": .5},
                                oracle=1.0, parent_reproduced=sup == "PLAIN"))
    summary = rq.summarize_scan(cfg, records)
    assert summary["by_d"][2]["reproduced"] is True
    records[-1]["parent_reproduced"] = False
    records[1]["parent_reproduced"] = False
    summary = rq.summarize_scan(cfg, records)
    assert summary["by_d"][2]["qualified"] is False
