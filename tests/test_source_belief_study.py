import json

import numpy as np
import pytest
import torch

from neurotopo.resources import resource_account
from neurotopo.run import configure
from neurotopo.source_belief_study import (build_arm, evaluate, loss_on,
                                           paired_evaluation_batches, parameter_sha256,
                                           schedule)
from neurotopo.source_belief_task import make_source_belief_batch


def _config():
    with open("configs/exp0009_v1.json", encoding="utf-8") as stream:
        return json.load(stream)


def test_registered_schedule_and_matched_primary_initialization():
    config = _config()
    values = schedule(config["schedule_seed"], config["updates"])
    assert values == schedule(config["schedule_seed"], config["updates"])
    assert len(values) == 1500
    for first in range(0, 1500, 300):
        assert {d: values[first:first + 300].count(d) for d in (0, 1, 2)} == {0: 100, 1: 100, 2: 100}
    models = {name: build_arm(name, config, 9401) for name in config["models"]}
    assert parameter_sha256(models["multi_gru"]) == parameter_sha256(models["random_gru"])
    a, b = (resource_account(models[name], 42, 128) for name in ("multi_gru", "random_gru"))
    assert a["effective_parameters"] == b["effective_parameters"] == 5704
    assert a["allocated_parameters"] == b["allocated_parameters"] == 14920
    assert a["hidden_state_floats"] == b["hidden_state_floats"] == 64
    assert a["pre_readout_cache_floats_per_example"] == b["pre_readout_cache_floats_per_example"] == 8
    assert a["dense_forward_linear_flops_per_batch"] == b["dense_forward_linear_flops_per_batch"]
    assert models["random_gru"].graph_info["strongly_connected"]


def test_final_stream_conditions_share_informative_evidence_and_targets():
    config = {**_config(), "test_examples": 32}
    batches = paired_evaluation_batches(config)
    assert list(batches) == [1, 4, 8]
    reference = batches[1]
    for batch in batches.values():
        for field in ("truth_h", "truth_r", "evidence_cues", "evidence_sources", "pre_joint", "final_joint"):
            assert torch.equal(getattr(reference, field), getattr(batch, field))
    assert [batch.x.shape[1] for batch in batches.values()] == [18, 42, 74]


@pytest.mark.parametrize("name", ["multi_gru", "random_gru", "gru_wide"])
def test_each_registered_model_can_overfit_a_fixed_posterior_batch(name):
    config = _config()
    configure(51, 1)
    model = build_arm(name, config, 51)
    batch = make_source_belief_batch(16, np.random.default_rng(5))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss_on(model, batch).backward()
        optimizer.step()
    metrics, _ = evaluate(model, batch)
    assert metrics["pre_joint_kl_to_oracle"] < 0.01
    assert metrics["final_joint_kl_to_oracle"] < 0.01
