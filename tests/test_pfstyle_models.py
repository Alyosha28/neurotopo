"""EXP-0011 engineering checks on isolated seeds and inputs only."""

import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from neurotopo.models import recurrent_mask, strongly_connected
from neurotopo.pfstyle_models import ARM_NAMES, build_model
from neurotopo.pfstyle_task import make_batch


MODEL_SEED = 9_900_071
GRAPH_SEED = 9_900_072
DATA_ROOT = 9_900_073


@pytest.mark.parametrize("name", ARM_NAMES)
def test_five_arms_shapes_gradients_and_prefix(name: str) -> None:
    torch.set_num_threads(1)
    model = build_model(name, MODEL_SEED, GRAPH_SEED)
    batch = make_batch(5, 16, DATA_ROOT)
    pre, final = model(batch.x)
    assert pre.shape == final.shape == (5, 4)
    assert model.dynamic_state_floats == 68
    assert torch.isfinite(pre).all() and torch.isfinite(final).all()
    if name.endswith("score"):
        torch.testing.assert_close(torch.logsumexp(pre, -1), torch.zeros(5), atol=1e-6, rtol=0)
    altered = batch.x.clone()
    altered[:, 8:] += 5
    assert torch.equal(model(altered)[0], pre)
    target_pre = batch.pre_target.to(torch.float32)
    target_final = batch.final_target.to(torch.float32)
    loss = -0.5 * (target_pre * F.log_softmax(pre, -1)).sum(-1).mean()
    loss += -0.5 * (target_final * F.log_softmax(final, -1)).sum(-1).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    longer = make_batch(5, 32, DATA_ROOT)
    early, late = model(longer.x)
    assert early.shape == late.shape == (5, 4)
    assert torch.equal(pre, early)


def test_gru_step_matches_independent_gate_equation() -> None:
    model = build_model("S68_sparse", MODEL_SEED, GRAPH_SEED)
    x = torch.randn(3, 2, generator=torch.Generator().manual_seed(23))
    h = torch.randn(3, 68, generator=torch.Generator().manual_seed(24))
    with torch.no_grad():
        input_affine = F.linear(x, model.weight_ih, model.bias_ih)
        recurrent_affine = F.linear(h, model.weight_hh * model.mask, model.bias_hh)
        ir, iz, inn = input_affine.chunk(3, -1)
        hr, hz, hn = recurrent_affine.chunk(3, -1)
        reset = (ir + hr).sigmoid()
        update = (iz + hz).sigmoid()
        candidate = (inn + reset * hn).tanh()
        expected = (1 - update) * candidate + update * h
        torch.testing.assert_close(model.step(x, h), expected, atol=0, rtol=0)


def test_scoring_uses_old_state_and_current_control_before_gru() -> None:
    model = build_model("K4_score", MODEL_SEED, GRAPH_SEED)
    x = torch.tensor([[0.4, -1.3], [-0.7, 0.2]], dtype=torch.float32)
    h = torch.randn(2, 64, generator=torch.Generator().manual_seed(25))
    with torch.no_grad():
        model.head_weight.copy_(torch.arange(136, dtype=torch.float32).reshape(4, 2, 17) / 137)
        model.head_bias.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2) / 13)
    initial = torch.full((2, 4), -math.log(4))
    emission = []
    for k in range(4):
        emission.append(F.linear(torch.cat((h[:, 16*k:16*(k+1)], x[:, :1]), -1),
                                 model.head_weight[k], model.head_bias[k]))
    emission = torch.stack(emission, dim=1)
    mu = emission[:, :, 0]
    sigma = 0.1 + 2.9 * torch.sigmoid(emission[:, :, 1])
    vector_mu, vector_sigma = model._gaussian_parameters(x, h)
    torch.testing.assert_close(vector_mu, mu, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(vector_sigma, sigma, atol=1e-6, rtol=1e-6)
    log_likelihood = -0.5 * ((x[:, 1:2] - mu) / sigma).square()
    log_likelihood -= torch.log(sigma) + 0.5 * math.log(2 * math.pi)
    expected = torch.log_softmax(initial + log_likelihood, -1)
    torch.testing.assert_close(model._score_update(x, h, initial), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("name", ("K4_score", "S64_score"))
def test_diagnostic_trace_is_read_only_and_matches_forward(name: str) -> None:
    model = build_model(name, MODEL_SEED, GRAPH_SEED)
    batch = make_batch(3, 16, DATA_ROOT)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    trace = model.diagnostic_trace(batch.x, include_hidden=True)
    assert set(trace) == {"mu", "sigma", "logw", "hidden"}
    assert trace["mu"].shape == trace["sigma"].shape == trace["logw"].shape == (3, 16, 4)
    assert trace["hidden"].shape == (3, 16, 64)
    assert torch.all(trace["sigma"] >= 0.1) and torch.all(trace["sigma"] <= 3.0)
    assert all(not item.requires_grad for item in trace.values())
    pre, final = model(batch.x)
    torch.testing.assert_close(trace["logw"][:, 7], pre, atol=1e-6, rtol=0)
    torch.testing.assert_close(trace["logw"][:, -1], final, atol=1e-6, rtol=0)
    # Registered interpretation diagnostics can be computed without feeding
    # the generator's mode means or hidden labels back into the model.
    entropy = -(trace["logw"].exp() * trace["logw"]).sum(-1)
    calibration_error = trace["mu"].double() - batch.mode_means
    standardized_residual = ((batch.observations[:, :, None] - trace["mu"].double())
                             / trace["sigma"].double())
    assert entropy.shape == (3, 16)
    assert torch.isfinite(entropy).all() and torch.all(entropy >= 0)
    assert torch.all(entropy <= math.log(4) + 1e-6)
    assert calibration_error.shape == standardized_residual.shape == (3, 16, 4)
    assert torch.isfinite(calibration_error).all()
    assert torch.isfinite(standardized_residual).all()
    plain = model.diagnostic_trace(batch.x)
    assert set(plain) == {"mu", "sigma", "logw"}
    for key, tensor in before.items():
        assert torch.equal(tensor, model.state_dict()[key])


def test_registered_graph_invariants_hash_and_parameter_accounts() -> None:
    expected = {
        "K4_score": (13_200, 3_984, 16, 0),
        "S68_sparse": (14_964, 4_152, 15, 47_600),
        "S68_dense": (14_964, 14_964, 68, 0),
        "S64_score": (13_200, 3_984, 16, 48_000),
        "K4_unscored": (14_760, 4_356, 17, 0),
    }
    for name, (allocated, effective, degree, swaps) in expected.items():
        model = build_model(name, MODEL_SEED, GRAPH_SEED)
        base = model.mask[:model.hidden].detach().numpy().astype(bool)
        assert sum(p.numel() for p in model.parameters()) == allocated
        assert allocated - model.weight_hh.numel() + int(model.mask.sum()) == effective
        np.testing.assert_array_equal(base.sum(0), degree)
        np.testing.assert_array_equal(base.sum(1), degree)
        assert np.all(np.diag(base))
        assert len(model.graph_info["mask_sha256"]) == 64
        assert model.graph_info["successful_swaps"] == swaps
        if name in ("S68_sparse", "S64_score", "S68_dense"):
            assert strongly_connected(base)
        if name == "S68_sparse":
            assert model.graph_info["initial_nonself_edges"] == 952
            assert model.graph_info["required_swaps"] == 47_600
            assert model.graph_info["attempts"] <= 4_760_000
        if name == "S64_score":
            registered, _ = recurrent_mask(64, 4, "random", GRAPH_SEED)
            assert torch.equal(model.mask[:64], registered)
        assert model.graph_info["mask_sha256"] == build_model(name, MODEL_SEED, GRAPH_SEED).graph_info["mask_sha256"]


def test_score_arms_have_identical_named_initial_trainable_tensors() -> None:
    block = build_model("K4_score", MODEL_SEED, GRAPH_SEED)
    random = build_model("S64_score", MODEL_SEED, GRAPH_SEED)
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
            block.named_parameters(), random.named_parameters(), strict=True):
        assert name_a == name_b
        assert torch.equal(parameter_a, parameter_b), name_a


def test_fixed_engineering_minibatch_is_learnable_with_40_updates() -> None:
    """Optimization smoke check: one arm, 40 engineering updates, no model selection."""
    torch.set_num_threads(1)
    batch = make_batch(24, 16, DATA_ROOT + 1)
    model = build_model("K4_score", MODEL_SEED + 1, GRAPH_SEED + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    pre_target = batch.pre_target.float()
    final_target = batch.final_target.float()

    def loss_value() -> torch.Tensor:
        pre, final = model(batch.x)
        return -0.5 * (pre_target * F.log_softmax(pre, -1)).sum(-1).mean() \
               -0.5 * (final_target * F.log_softmax(final, -1)).sum(-1).mean()

    initial = float(loss_value().detach())
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final = float(loss_value().detach())
    assert math.isfinite(final)
    assert final < initial - 0.01
