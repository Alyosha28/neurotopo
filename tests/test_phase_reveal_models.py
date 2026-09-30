"""Engineering-only checks for the registered G1-RS model arms.

Inputs here are arbitrary torch.randn tensors, never phase-reveal samples.
"""

from hashlib import sha256

import numpy as np
import pytest
import torch

from neurotopo.models import recurrent_mask, strongly_connected
from neurotopo.phase_reveal_models import ARM_NAMES, RoutedModel, build_model
from neurotopo.phase_reveal_resources import resource_account


MODEL_SEED = 14_201
GRAPH_SEED = 14_000_000 + MODEL_SEED


@pytest.fixture(scope="module")
def arms() -> dict[str, RoutedModel]:
    return {name: build_model(name, MODEL_SEED, GRAPH_SEED) for name in ARM_NAMES}


def test_identical_sampled_parameters_and_registered_counts(
    arms: dict[str, RoutedModel],
) -> None:
    rs, sc = arms["RS4"], arms["SC64"]
    for (rs_name, rs_value), (sc_name, sc_value) in zip(
        rs.named_parameters(), sc.named_parameters(), strict=True
    ):
        assert rs_name == sc_name
        assert rs_value.detach().numpy().tobytes() == sc_value.detach().numpy().tobytes()
    assert torch.count_nonzero(rs.bias_ih) == 0
    assert torch.count_nonzero(rs.bias_hh) == 0
    assert torch.count_nonzero(rs.readout_bias) == 0
    assert torch.count_nonzero(rs.weight_ih[rs.input_mask == 0]) > 0
    assert torch.count_nonzero(rs.weight_hh[rs.mask == 0]) > 0

    for name, (allocated, effective) in {
        "RS4": (16_004, 4_484), "SC64": (16_004, 4_484),
        "D64_full": (16_004, 16_004),
    }.items():
        account = resource_account(arms[name], 32, 128)
        assert (account["allocated_parameters"], account["effective_parameters"]) == (
            allocated, effective)
        assert account["hidden_state_floats"] == 64
        assert account["pre_readout_cache_floats_per_example"] == 4
        assert account["input_linear_macs_per_token"] == 3_072
        assert account["dense_recurrent_linear_macs_per_token"] == 12_288
        assert account["dense_forward_linear_macs_per_example"] == 32 * 15_360 + 512
        assert account["operator_shapes"]["input_linear_per_token"]["weight"] == [192, 16]
        assert account["operator_shapes"]["recurrent_linear_per_token"]["weight"] == [192, 64]
        assert account["operator_shapes"]["query_readout"]["weight"] == [4, 64]
    assert resource_account(rs, 48, 128)["operator_shapes"] == resource_account(
        sc, 48, 128)["operator_shapes"]


def test_routed_input_and_graph_invariants(arms: dict[str, RoutedModel]) -> None:
    for name, model in arms.items():
        base = model.mask[:64].numpy().astype(bool)
        input_base = model.input_mask[:64].numpy().astype(bool)
        assert len(model.graph_info["mask_sha256"]) == 64
        assert model.graph_info["mask_sha256"] == sha256(
            base.astype(np.uint8).tobytes(order="C")).hexdigest()
        assert np.all(np.diag(base))
        assert model.graph_info["diagonal_connections"] == 64
        assert np.array_equal(base.sum(axis=0), np.array(model.graph_info["out_degree"]))
        assert np.array_equal(base.sum(axis=1), np.array(model.graph_info["in_degree"]))
        assert base.shape == (64, 64)
        assert input_base.shape == (64, 16)
        assert np.array_equal(model.mask.numpy(), np.tile(base, (3, 1)))
        assert np.array_equal(model.input_mask.numpy(), np.tile(input_base, (3, 1)))
        if name == "D64_full":
            assert base.all() and input_base.all()
            assert model.graph_info["successful_swaps"] == 0
        else:
            assert np.all(base.sum(axis=0) == 16)
            assert np.all(base.sum(axis=1) == 16)
            assert np.all(input_base.sum(axis=1) == 4)
            for group in range(4):
                expected = np.zeros(16, dtype=bool)
                expected[group * 4:(group + 1) * 4] = True
                assert np.all(input_base[group * 16:(group + 1) * 16] == expected)
        if name == "RS4":
            assert not strongly_connected(base)
            assert model.graph_info["successful_swaps"] == 0
            assert np.array_equal(base, np.kron(np.eye(4, dtype=bool),
                                                 np.ones((16, 16), dtype=bool)))
        if name == "SC64":
            assert strongly_connected(base)
            assert model.graph_info["required_swaps"] == 48_000
            assert model.graph_info["successful_swaps"] == 48_000
            registered, _ = recurrent_mask(64, 4, "random", GRAPH_SEED)
            assert torch.equal(model.mask[:64], registered)


def test_shape_gradients_and_prefix_suffix_replay(arms: dict[str, RoutedModel]) -> None:
    torch.manual_seed(9_900_014)
    x = torch.randn(3, 48, 16)
    for model in arms.values():
        early, final = model(x, pre_index=8)
        assert early.shape == final.shape == (3, 4)
        trace = model.run_trace(x)
        assert trace["hidden_by_time"][16].shape == (3, 64)
        assert trace["block_hidden_by_time"][48].shape == (3, 4, 16)
        assert torch.equal(final, trace["logits_by_time"][48])
        h16 = model.hidden_after_prefix(x[:, :16])
        assert torch.equal(h16, trace["hidden_by_time"][16])
        replay_logits, h48 = model.replay_suffix(x[:, 16:], h16)
        assert torch.equal(replay_logits, final)
        assert torch.equal(h48, trace["hidden_by_time"][48])
        assert torch.isfinite(early).all() and torch.isfinite(final).all()
        model.zero_grad(set_to_none=True)
        (early.square().mean() + final.square().mean()).backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters())
        assert torch.count_nonzero(model.weight_ih.grad[model.input_mask == 0]) == 0
        assert torch.count_nonzero(model.weight_hh.grad[model.mask == 0]) == 0


def test_rs_blocks_have_exactly_zero_cross_block_effect(
    arms: dict[str, RoutedModel],
) -> None:
    rs = arms["RS4"]
    torch.manual_seed(101)
    x = torch.randn(2, 16, 16)
    changed = x.clone()
    changed[:, :, 0:4] += 10.0
    original_h = rs.hidden_after_prefix(x).reshape(2, 4, 16)
    changed_h = rs.hidden_after_prefix(changed).reshape(2, 4, 16)
    assert not torch.equal(original_h[:, 0], changed_h[:, 0])
    assert torch.equal(original_h[:, 1:], changed_h[:, 1:])

    suffix = torch.randn(2, 16, 16)
    transplanted = original_h.clone()
    transplanted[:, 0] = changed_h[:, 0]
    _, factual_final_h = rs.replay_suffix(suffix, original_h.reshape(2, 64))
    _, transplanted_final_h = rs.replay_suffix(suffix, transplanted.reshape(2, 64))
    factual_blocks = factual_final_h.reshape(2, 4, 16)
    transplanted_blocks = transplanted_final_h.reshape(2, 4, 16)
    assert not torch.equal(factual_blocks[:, 0], transplanted_blocks[:, 0])
    assert torch.equal(factual_blocks[:, 1:], transplanted_blocks[:, 1:])


def test_invalid_contracts() -> None:
    with pytest.raises(ValueError):
        build_model("other", MODEL_SEED, GRAPH_SEED)
    model = build_model("RS4", MODEL_SEED, GRAPH_SEED)
    with pytest.raises(ValueError):
        model(torch.randn(1, 7, 16))
    with pytest.raises(ValueError):
        model.run_trace(torch.randn(1, 16, 16), (16, 16))
    with pytest.raises(ValueError):
        model.replay_suffix(torch.randn(1, 16, 16), torch.zeros(1, 4, 16))
