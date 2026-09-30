"""Engineering tests for EXP0030 (DELAY-d memory task family); see experiments/EXP-0030/v1.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neurotopo import arch0001_memory_task as mt

CONFIG = Path("configs/exp0030_v1.json")


def test_config_and_protocol_exist() -> None:
    cfg = json.loads(CONFIG.read_text())
    mt.validate_config(cfg)
    assert Path(cfg["protocol"]).exists() and Path(cfg["candidate_contract"]).exists()


def test_frozen_candidate_module_untouched() -> None:
    manifest = json.loads(Path("artifacts/EXP-0028/v1/manifest.json").read_text())
    frozen = manifest["frozen_hashes"]["src/neurotopo/arch0001_core.py"]
    from neurotopo.order_observation import sha
    assert sha(Path("src/neurotopo/arch0001_core.py")) == frozen


def test_generator_targets_and_branch_labels() -> None:
    for d in (1, 2, 3):
        data = mt.generate_memory(64, 16, [5, 6, 7, d], d)
        u = data["commands"]
        n, length = u.shape
        assert data["targets"].shape == (n, length + 1) and data["branch_targets"].shape == (n, length + 1, 2)
        assert np.all(data["targets"][:, 0] == 0)
        for t in range(d, length):
            assert np.array_equal(data["targets"][:, t + 1], u[:, t - d])
        assert np.all(data["targets"][:, 1:d + 1] == 0)
        assert np.array_equal(data["branch_targets"][:, :-1, 0], data["targets"][:, 1:])
        assert np.array_equal(data["branch_targets"][:, :, 0], data["branch_targets"][:, :, 1])
        assert np.array_equal(data["b0"], np.zeros(n, np.int64))
        assert .35 < u.mean() < .65


def test_generator_deterministic_and_seed_dependent() -> None:
    a = mt.generate_memory(32, 12, [8, 9, 10, 1], 2)
    b = mt.generate_memory(32, 12, [8, 9, 10, 1], 2)
    c = mt.generate_memory(32, 12, [8, 9, 11, 1], 2)
    for key in ("commands", "targets", "branch_targets", "common"):
        assert np.array_equal(a[key], b[key])
        assert not np.array_equal(a[key], c[key])


def test_references_are_chance_and_ceiling() -> None:
    cfg = json.loads(CONFIG.read_text())
    for d in (1, 2, 3):
        stream = mt.evaluation_stream(cfg, 29101, d, 12)
        memoryless = mt.reference_memoryless(stream)["accuracy_all"]
        oracle = mt.reference_oracle(stream, d)["accuracy_all"]
        assert memoryless <= cfg["band"]["memoryless_max"], (d, memoryless)
        assert oracle >= cfg["band"]["oracle_min"] and oracle == 1.0


def test_curriculum_and_loss_variants() -> None:
    cfg = json.loads(CONFIG.read_text())
    assert [mt.curriculum_length(s, cfg["training"]["curriculum"]) for s in (1, 500, 501, 1000, 1500, 2000)] == [2, 2, 4, 4, 8, 12]
    data = mt.generate_memory(8, 4, [11, 12, 13, 1], 1)
    parts = mt.fit_memory(29101, 1, "PLAIN", cfg, updates=3)["model"]
    parts = mt.evaluate_memory(parts, data)
    loss_plain, terms_plain = mt.memory_loss(parts, data, "PLAIN", .1)
    loss_branch, terms_branch = mt.memory_loss(parts, data, "BRANCH", .1)
    assert terms_plain["l_branch"] == 0. and terms_branch["l_branch"] > 0.
    assert torch.isfinite(loss_plain) and torch.isfinite(loss_branch) and loss_branch > loss_plain
    metrics = mt.memory_metrics(parts, data)
    assert 0. <= metrics["accuracy_all"] <= 1. and 0. <= metrics["branch_joint_accuracy"] <= 1.


def _record(d: int, sup: str, seed: int, accuracy: float) -> dict:
    return dict(d=d, sup=sup, seed=seed, primary=dict(accuracy_all=accuracy))


def test_scan_band_flags() -> None:
    cfg = json.loads(CONFIG.read_text())
    records = [_record(1, "PLAIN", 29101, .70), _record(1, "PLAIN", 29102, .72),
               _record(2, "PLAIN", 29101, .99), _record(2, "PLAIN", 29102, .995),
               _record(3, "PLAIN", 29101, .40), _record(3, "PLAIN", 29102, .45)]
    summary = mt.summarize_scan(cfg, records)
    assert summary["by_d"][1]["qualified"] is True
    assert summary["by_d"][2]["nonceiling"] is False
    assert summary["by_d"][3]["learn"] is False
    assert summary["qualified"] == dict(d=1, sup="PLAIN")
