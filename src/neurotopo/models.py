"""Matched dense execution for block and random masked recurrent controls."""
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def strongly_connected(mask: np.ndarray) -> bool:
    """Directed adjacency convention: row is receiving node."""
    for adjacency in (mask, mask.T):
        visited = {0}
        pending = [0]
        while pending:
            current = pending.pop()
            for target in np.flatnonzero(adjacency[:, current]):
                if int(target) not in visited:
                    visited.add(int(target))
                    pending.append(int(target))
        if len(visited) != len(mask):
            return False
    return True


def recurrent_mask(hidden: int, groups: int, topology: str, seed: int) -> tuple[Tensor, dict[str, Any]]:
    if hidden <= 0 or groups <= 0 or hidden % groups:
        raise ValueError("hidden must be a positive multiple of groups")
    width = hidden // groups
    idx = np.arange(hidden) // width
    mask = idx[:, None] == idx[None, :]
    swaps = 0
    if topology == "dense":
        mask[:] = True
    elif topology == "random":
        if groups == 1:
            raise ValueError("random topology requires groups > 1")
        rng = np.random.default_rng(seed)
        # Preserve self-feedback exactly: randomize only non-diagonal edges.
        off_diagonal = mask & ~np.eye(hidden, dtype=bool)
        edges = np.argwhere(off_diagonal)
        if not len(edges):
            raise ValueError("random topology needs non-diagonal recurrent edges")
        required = 50 * len(edges)
        attempts = 0
        while swaps < required:
            attempts += 1
            if attempts > required * 100:
                raise RuntimeError("degree-preserving randomization exhausted attempts")
            i, j = rng.integers(len(edges), size=2)
            a, b = edges[i]
            c, d = edges[j]
            if a == c or b == d or a == d or c == b or mask[a, d] or mask[c, b]:
                continue
            mask[a, b] = mask[c, d] = False
            mask[a, d] = mask[c, b] = True
            edges[i] = (a, d)
            edges[j] = (c, b)
            swaps += 1
        if not strongly_connected(mask):
            raise RuntimeError("random control must be strongly connected; no silent resampling")
    elif topology != "block":
        raise ValueError(f"unknown topology: {topology}")
    return torch.tensor(mask, dtype=torch.float32), {
        "topology": topology, "seed": seed, "successful_swaps": swaps,
        "in_degree": mask.sum(axis=1).tolist(), "out_degree": mask.sum(axis=0).tolist(),
        "strongly_connected": strongly_connected(mask),
        "diagonal_connections": int(np.trace(mask)),
        "uniform_graph_sample_claimed": False,
    }


class RecurrentModel(nn.Module):
    """Ablatable GRU/tanh with fixed recurrent masks, no candidate assignment."""

    def __init__(self, input_size: int, hidden: int, classes: int, *, groups: int = 1,
                 topology: str = "dense", cell: str = "gru", graph_seed: int = 0) -> None:
        super().__init__()
        if cell not in {"gru", "tanh"}:
            raise ValueError(cell)
        self.hidden, self.input_size, self.classes, self.cell = hidden, input_size, classes, cell
        self.groups, self.topology = groups, topology
        self.gates = 3 if cell == "gru" else 1
        base_mask, self.graph_info = recurrent_mask(hidden, groups, topology, graph_seed)
        self.register_buffer("mask", base_mask.repeat(self.gates, 1))
        self.weight_ih = nn.Parameter(torch.empty(self.gates * hidden, input_size))
        self.weight_hh = nn.Parameter(torch.empty(self.gates * hidden, hidden))
        self.bias_ih = nn.Parameter(torch.zeros(self.gates * hidden))
        self.bias_hh = nn.Parameter(torch.zeros(self.gates * hidden))
        self.readout = nn.Linear(hidden, classes)
        nn.init.uniform_(self.weight_ih, -1 / math.sqrt(input_size), 1 / math.sqrt(input_size))
        degree = int(base_mask[0].sum().item())
        nn.init.uniform_(self.weight_hh, -1 / math.sqrt(degree), 1 / math.sqrt(degree))

    def step(self, x: Tensor, h: Tensor, recurrent_weight: Tensor | None = None) -> Tensor:
        weights = self.weight_hh * self.mask if recurrent_weight is None else recurrent_weight
        gi = F.linear(x, self.weight_ih, self.bias_ih)
        gh = F.linear(h, weights, self.bias_hh)
        if self.cell == "tanh":
            return torch.tanh(gi + gh)
        ir, iz, inn = gi.chunk(3, dim=-1)
        hr, hz, hn = gh.chunk(3, dim=-1)
        r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        n = torch.tanh(inn + r * hn)
        return (1 - z) * n + z * h

    def forward(self, x: Tensor, pre_index: int) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.input_size or not 0 <= pre_index < x.shape[1]:
            raise ValueError("expected [batch, time, features] and valid pre_index")
        h = x.new_zeros((x.shape[0], self.hidden))
        weight = self.weight_hh * self.mask
        pre = None
        for t in range(x.shape[1]):
            h = self.step(x[:, t], h, weight)
            if t == pre_index:
                pre = self.readout(h)
        assert pre is not None
        return pre, self.readout(h)


def build_model(name: str, config: dict[str, Any], seed: int) -> RecurrentModel:
    from .task import CLASSES, INPUT_SIZE
    hidden = config["small_hidden_size"] if name == "gru_small" else config["hidden_size"]
    options = {
        "gru_small": ("dense", "gru"), "gru_wide": ("dense", "gru"),
        "sparse_rnn": ("random", "tanh"), "multi_gru": ("block", "gru"),
        "random_gru": ("random", "gru"),
    }
    topology, cell = options[name]
    return RecurrentModel(INPUT_SIZE, hidden, CLASSES, groups=config["trajectories"],
                          topology=topology, cell=cell, graph_seed=config["graph_seed_offset"] + seed)
