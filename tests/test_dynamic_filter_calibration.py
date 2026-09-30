from pathlib import Path
from runpy import run_path
import time

import numpy as np
import pytest
import torch

from neurotopo.dynamic_filter_task import make_dynamic_filter_batch
from neurotopo.resources import resource_account


_script = run_path(str(Path(__file__).resolve().parents[1] / "scripts/calibrate_dynamic_filter.py"))
build_arm = _script["build_arm"]
oracle_metrics = _script["oracle_metrics"]
prefix_digest = _script["prefix_digest"]
train_seed = _script["train_seed"]
finish_report = _script["finish_report"]
verify_publication = _script["verify_publication"]
model_metrics = _script["model_metrics"]


def test_registered_streams_are_disjoint_and_prefix_hash_matches():
    train = {train_seed(seed, step) for seed in (601, 602, 603) for step in range(1, 1201)}
    validation = {6_300_000 + seed for seed in (601, 602, 603)}
    assert len(train) == 3600
    assert train.isdisjoint(validation | {6_400_001, 9_000_001})
    assert validation.isdisjoint({6_400_001, 9_000_001})
    short = make_dynamic_filter_batch(64, 8, 6_400_001)
    long = make_dynamic_filter_batch(64, 12, 6_400_001)
    assert prefix_digest(short) == prefix_digest(long)
    assert torch.equal(short.truth, long.truth)
    assert torch.equal(short.final_joint, long.posterior_history[:, 7].to(torch.float32))


def test_single_state_screen_resource_boundary_and_oracle_metrics():
    dense = build_arm("dense", 601)
    random = build_arm("random", 601)
    a, b = resource_account(dense, 10, 128), resource_account(random, 10, 128)
    assert a["hidden_state_floats"] == b["hidden_state_floats"] == 64
    assert a["pre_readout_cache_floats_per_example"] == b["pre_readout_cache_floats_per_example"] == 4
    assert a["dense_forward_linear_flops_per_batch"] == b["dense_forward_linear_flops_per_batch"]
    assert a["allocated_parameters"] == b["allocated_parameters"]
    assert a["effective_parameters"] > b["effective_parameters"]
    assert random.graph_info["strongly_connected"]
    batch = make_dynamic_filter_batch(100, 12, 661)
    metric = oracle_metrics(batch)
    p = batch.final_joint.numpy().astype(np.float64)
    expected = float((p.argmax(1) == batch.truth.numpy()).mean())
    assert metric["final_true_map_accuracy"] == expected
    assert np.isclose(metric["uniform_kl_to_oracle_nats"],
                      np.log(4) + float((p * np.log(np.clip(p, 1e-12, 1))).sum(1).mean()), atol=1e-7)


def test_publication_is_exclusive_and_deadline_has_priority(tmp_path):
    output = tmp_path / "CAL-0006_v3.json"
    report = {"status": "SCREEN_PASSED_FOR_DESIGN_REVIEW", "qualified_for_design_review": True}
    done = finish_report(output, report, time.perf_counter() - 901)
    assert done["status"] == "INCOMPLETE_BUDGET"
    assert not done["qualified_for_design_review"]
    assert verify_publication(output)["status"] == "INCOMPLETE_BUDGET"
    with pytest.raises(FileExistsError):
        finish_report(output, {"status": "ENGINEERING_ABORT"}, time.perf_counter())


def test_abort_report_can_record_nonfinite_value(tmp_path):
    output = tmp_path / "CAL-0006_v3.json"
    finish_report(output, {"status": "ENGINEERING_ABORT", "bad_metric": float("nan"),
                           "qualified_for_design_review": False}, time.perf_counter())
    assert verify_publication(output)["bad_metric"] == "NONFINITE:nan"


def test_nonfinite_model_metric_fails_closed():
    model = build_arm("dense", 601)
    with torch.no_grad():
        model.readout.weight.fill_(float("nan"))
    batch = make_dynamic_filter_batch(12, 8, 75)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        model_metrics(model, batch)
