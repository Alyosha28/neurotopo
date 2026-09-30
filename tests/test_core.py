import numpy as np
import pytest
import torch

from neurotopo.models import RecurrentModel, recurrent_mask, strongly_connected
from neurotopo.resources import resource_account
from neurotopo.task import CANDIDATES, EVIDENCE, INPUT_SIZE, make_batch


def test_counterfactual_targets_and_determinism():
    b = make_batch(256, 8, np.random.default_rng(1))
    c = make_batch(256, 8, np.random.default_rng(1))
    assert torch.equal(b.x, c.x)
    assert torch.equal(b.x[::2, :b.pre_index + 1], b.x[1::2, :b.pre_index + 1])
    assert torch.equal(b.x[::2, b.pre_index + 2:], b.x[1::2, b.pre_index + 2:])
    assert torch.equal(b.pre_target[::2], b.final_target[::2])
    assert torch.all(b.pre_target[1::2] != b.final_target[1::2])
    assert torch.all(b.x[:, b.pre_index + 1, 4:8] == 0)  # no label in selector
    assert torch.all(b.x[:, b.pre_index + 1, 8 + EVIDENCE] == 1)
    for j in range(len(b.x)):
        observed = {}
        for event in b.x[j, :CANDIDATES]:
            observed[event[:4].argmax().item()] = event[4:8].argmax().item()
        assert observed[b.prior[j].item()] == b.pre_target[j].item()
        assert observed[b.selected[j].item()] == b.final_target[j].item()


def test_same_selector_different_prefix_changes_target():
    b = make_batch(512, 4, np.random.default_rng(3))
    for selected in range(CANDIDATES):
        targets = b.final_target[b.selected == selected].unique()
        assert len(targets) == 4  # selector alone never determines class


def test_no_future_leakage():
    torch.manual_seed(1)
    m = RecurrentModel(INPUT_SIZE, 16, 4, groups=4, topology="block")
    b = make_batch(8, 4, np.random.default_rng(2))
    x = b.x.clone()
    x[:, b.pre_index + 1:] = torch.randn_like(x[:, b.pre_index + 1:])
    assert torch.equal(m(x, b.pre_index)[0], m(b.x, b.pre_index)[0])


def test_dense_gru_matches_pytorch():
    torch.manual_seed(2)
    m = RecurrentModel(5, 12, 4)
    reference = torch.nn.GRUCell(5, 12)
    with torch.no_grad():
        for key in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
            getattr(reference, key).copy_(getattr(m, key))
    x, h = torch.randn(7, 5), torch.randn(7, 12)
    torch.testing.assert_close(m.step(x, h), reference(x, h), rtol=1e-6, atol=1e-7)


def test_block_gru_equals_independent_cells():
    torch.manual_seed(3)
    h, width, groups = 16, 4, 4
    m = RecurrentModel(5, h, 4, groups=groups, topology="block")
    x, state = torch.randn(3, 5), torch.randn(3, h)
    outputs = []
    for k in range(groups):
        sl = slice(k * width, (k + 1) * width)
        gate_idx = torch.cat([torch.arange(g * h + k * width, g * h + (k + 1) * width) for g in range(3)])
        cell = torch.nn.GRUCell(5, width)
        with torch.no_grad():
            cell.weight_ih.copy_(m.weight_ih[gate_idx])
            cell.weight_hh.copy_(m.weight_hh[gate_idx, sl])
            cell.bias_ih.copy_(m.bias_ih[gate_idx])
            cell.bias_hh.copy_(m.bias_hh[gate_idx])
        outputs.append(cell(x, state[:, sl]))
    torch.testing.assert_close(m.step(x, state), torch.cat(outputs, dim=-1), rtol=1e-6, atol=1e-7)


def test_masks_preserve_degree_and_count():
    block, _ = recurrent_mask(16, 4, "block", 0)
    random, metadata = recurrent_mask(16, 4, "random", 1)
    repeated, _ = recurrent_mask(16, 4, "random", 1)
    assert torch.equal(random, repeated)
    assert torch.equal(block.sum(0), random.sum(0))
    assert torch.equal(block.sum(1), random.sum(1))
    assert strongly_connected(random.numpy())
    assert not strongly_connected(block.numpy())
    assert metadata["successful_swaps"] == 50 * 16 * 3
    assert torch.equal(block.diag(), random.diag())
    assert metadata["diagonal_connections"] == 16


@pytest.mark.parametrize("topology,cell", [("dense", "gru"), ("block", "gru"), ("random", "gru"), ("random", "tanh")])
def test_shapes_gradients_and_masked_parameters(topology, cell):
    m = RecurrentModel(INPUT_SIZE, 16, 4, groups=4, topology=topology, cell=cell)
    b = make_batch(4, 2, np.random.default_rng(3))
    pre, final = m(b.x, b.pre_index)
    assert pre.shape == final.shape == (8, 4)
    (pre.square().mean() + final.square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    assert torch.all(m.weight_hh.grad[m.mask == 0] == 0)
    before = final.detach()
    with torch.no_grad():
        m.weight_hh[m.mask == 0] = 1e6
    assert torch.equal(m(b.x, b.pre_index)[1], before)


def test_matched_accounting():
    block = RecurrentModel(INPUT_SIZE, 64, 4, groups=4, topology="block")
    random = RecurrentModel(INPUT_SIZE, 64, 4, groups=4, topology="random")
    a, b = resource_account(block, 18), resource_account(random, 18)
    for key in ["allocated_parameters", "effective_parameters", "hidden_state_floats",
                "pre_readout_cache_floats_per_example", "pre_readout_cache_bytes_per_example",
                "dense_forward_linear_macs_per_example", "active_forward_linear_macs_per_example"]:
        assert a[key] == b[key]
    assert a["hidden_state_floats"] == 64
    assert a["pre_readout_cache_floats_per_example"] == 4
    assert a["pre_readout_cache_bytes_per_example"] == 16
    expected = 3 * 64 * INPUT_SIZE + 3 * 64 * 16 + 6 * 64 + 64 * 4 + 4
    assert a["effective_parameters"] == expected


def test_training_is_deterministic():
    from neurotopo.run import configure, loss_on
    batch = make_batch(4, 2, np.random.default_rng(99))
    final = []
    for _ in range(2):
        configure(32)
        model = RecurrentModel(INPUT_SIZE, 16, 4, groups=4, topology="block")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _step in range(5):
            optimizer.zero_grad()
            loss_on(model, batch).backward()
            optimizer.step()
        final.append(model(batch.x, batch.pre_index)[1].detach())
    assert torch.equal(*final)
