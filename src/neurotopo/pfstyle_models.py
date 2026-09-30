"""Independent implementation of the five registered EXP-0011 arms."""

from __future__ import annotations

from hashlib import sha256
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import recurrent_mask, strongly_connected


ARM_NAMES = ("K4_score", "S68_sparse", "S68_dense", "S64_score", "K4_unscored")


def sparse_ring_mask(hidden: int, degree: int, seed: int) -> tuple[Tensor, dict[str, Any]]:
    """Degree-preserving switches from a bidirectional ring, row = receiver.

    This uses the same eligible-edge and attempted-switch rule as
    ``recurrent_mask(..., topology='random')``. It does not retry another graph.
    """
    if hidden != 68 or degree != 15:
        raise ValueError("EXP-0011 sparse control requires hidden=68 and degree=15")
    mask = np.zeros((hidden, hidden), dtype=bool)
    for receiver in range(hidden):
        mask[receiver, receiver] = True
        for distance in range(1, 8):
            mask[receiver, (receiver - distance) % hidden] = True
            mask[receiver, (receiver + distance) % hidden] = True
    edges = np.argwhere(mask & ~np.eye(hidden, dtype=bool))
    initial_edges = len(edges)
    required = 50 * initial_edges
    rng = np.random.default_rng(seed)
    swaps = attempts = 0
    while swaps < required:
        attempts += 1
        if attempts > 100 * required:
            raise RuntimeError("EXP-0011 sparse graph exhausted registered attempt bound")
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
    indegree = mask.sum(axis=1)
    outdegree = mask.sum(axis=0)
    connected = strongly_connected(mask)
    if not (np.all(indegree == degree) and np.all(outdegree == degree)
            and np.all(np.diag(mask)) and connected and swaps == required):
        raise RuntimeError("EXP-0011 sparse graph invariant failed")
    return torch.tensor(mask, dtype=torch.float32), {
        "topology": "sparse_ring_randomized", "seed": seed,
        "initial_nonself_edges": initial_edges, "required_swaps": required,
        "successful_swaps": swaps, "attempts": attempts,
        "in_degree": indegree.tolist(), "out_degree": outdegree.tolist(),
        "diagonal_connections": int(np.trace(mask)), "strongly_connected": connected,
        "mask_sha256": sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest(),
        "uniform_graph_sample_claimed": False,
    }


class PfstyleModel(nn.Module):
    """One masked dense-execution GRU with a registered output mechanism."""

    def __init__(self, name: str, graph_seed: int) -> None:
        super().__init__()
        if name not in ARM_NAMES:
            raise ValueError(f"unknown EXP-0011 arm: {name}")
        self.name = name
        self.input_size = 2
        self.classes = 4
        self.gates = 3
        self.hidden = 64 if name in ("K4_score", "S64_score") else 68
        self.score_state_floats = 4 if name in ("K4_score", "S64_score") else 0
        self.dynamic_state_floats = self.hidden + self.score_state_floats
        if name == "K4_score":
            base_mask, info = recurrent_mask(64, 4, "block", graph_seed)
            self.groups, self.topology = 4, "block"
        elif name == "S64_score":
            # Registered exact existing construction: block start + 50x swaps.
            base_mask, info = recurrent_mask(64, 4, "random", graph_seed)
            self.groups, self.topology = 4, "random"
        elif name == "S68_sparse":
            base_mask, info = sparse_ring_mask(68, 15, graph_seed)
            self.groups, self.topology = 1, "sparse_ring_randomized"
        elif name == "S68_dense":
            base_mask, info = recurrent_mask(68, 1, "dense", graph_seed)
            self.groups, self.topology = 1, "dense"
        else:
            base_mask, info = recurrent_mask(68, 4, "block", graph_seed)
            self.groups, self.topology = 4, "block"
        raw_mask = base_mask.detach().cpu().numpy().astype(np.uint8)
        info = dict(info)
        info["mask_sha256"] = sha256(raw_mask.tobytes(order="C")).hexdigest()
        info["required_swaps"] = (50 * (int(base_mask.sum().item()) - self.hidden)
                                  if name == "S64_score" else info.get("required_swaps", 0))
        self.graph_info = info
        self.register_buffer("mask", base_mask.repeat(3, 1))
        self.weight_ih = nn.Parameter(torch.empty(3 * self.hidden, 2))
        self.weight_hh = nn.Parameter(torch.empty(3 * self.hidden, self.hidden))
        self.bias_ih = nn.Parameter(torch.zeros(3 * self.hidden))
        self.bias_hh = nn.Parameter(torch.zeros(3 * self.hidden))
        nn.init.uniform_(self.weight_ih, -1 / math.sqrt(2), 1 / math.sqrt(2))
        degree = int(base_mask[0].sum().item())
        nn.init.uniform_(self.weight_hh, -1 / math.sqrt(degree), 1 / math.sqrt(degree))
        if self.score_state_floats:
            # Four independent Linear(17, 2) heads in a single batched affine.
            self.head_weight = nn.Parameter(torch.empty(4, 2, 17))
            self.head_bias = nn.Parameter(torch.empty(4, 2))
            nn.init.uniform_(self.head_weight, -1 / math.sqrt(17), 1 / math.sqrt(17))
            nn.init.uniform_(self.head_bias, -1 / math.sqrt(17), 1 / math.sqrt(17))
        elif name == "K4_unscored":
            self.heads = nn.ModuleList(nn.Linear(17, 1) for _ in range(4))
        else:
            self.readout = nn.Linear(68, 4)

    def step(self, x: Tensor, h: Tensor, recurrent_weight: Tensor | None = None) -> Tensor:
        """Exactly the project's RecurrentModel GRU gate equation."""
        weights = self.weight_hh * self.mask if recurrent_weight is None else recurrent_weight
        gi = F.linear(x, self.weight_ih, self.bias_ih)
        gh = F.linear(h, weights, self.bias_hh)
        ir, iz, inn = gi.chunk(3, dim=-1)
        hr, hz, hn = gh.chunk(3, dim=-1)
        r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        n = torch.tanh(inn + r * hn)
        return (1 - z) * n + z * h

    def _gaussian_parameters(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        if not self.score_state_floats:
            raise ValueError("Gaussian heads exist only on scored arms")
        # Heads see current u and the OLD block state; y enters only likelihood.
        features = torch.cat((h.reshape(h.shape[0], 4, 16),
                              x[:, None, :1].expand(-1, 4, -1)), dim=-1)
        emission = torch.einsum("bki,koi->bko", features, self.head_weight) + self.head_bias
        mu = emission[:, :, 0]
        sigma = 0.1 + 2.9 * torch.sigmoid(emission[:, :, 1])
        return mu, sigma

    @staticmethod
    def _logw_update(x: Tensor, logw: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        residual = (x[:, 1:2] - mu) / sigma
        likelihood = -0.5 * residual.square() - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
        return F.log_softmax(logw + likelihood, dim=-1)

    def _score_update(self, x: Tensor, h: Tensor, logw: Tensor) -> Tensor:
        mu, sigma = self._gaussian_parameters(x, h)
        return self._logw_update(x, logw, mu, sigma)

    @torch.no_grad()
    def diagnostic_trace(self, x: Tensor, *, include_hidden: bool = False) -> dict[str, Tensor]:
        """Read-only post-token Gaussian and score traces for scored arms.

        ``mu`` and ``sigma`` are predicted from the pre-token hidden state;
        ``logw`` and optional ``hidden`` are states after each token.
        """
        if not self.score_state_floats:
            raise ValueError("diagnostic_trace applies only to scored arms")
        if x.ndim != 3 or x.shape[-1] != 2 or x.shape[1] not in (16, 32):
            raise ValueError("expected [batch, 16 or 32, 2]")
        h = x.new_zeros((x.shape[0], self.hidden))
        logw = x.new_full((x.shape[0], 4), -math.log(4))
        recurrent_weight = self.weight_hh * self.mask
        means: list[Tensor] = []
        sigmas: list[Tensor] = []
        weights: list[Tensor] = []
        hidden: list[Tensor] = []
        for t in range(x.shape[1]):
            mu, sigma = self._gaussian_parameters(x[:, t], h)
            logw = self._logw_update(x[:, t], logw, mu, sigma)
            h = self.step(x[:, t], h, recurrent_weight)
            means.append(mu)
            sigmas.append(sigma)
            weights.append(logw)
            if include_hidden:
                hidden.append(h)
        trace = {"mu": torch.stack(means, dim=1),
                 "sigma": torch.stack(sigmas, dim=1),
                 "logw": torch.stack(weights, dim=1)}
        if include_hidden:
            trace["hidden"] = torch.stack(hidden, dim=1)
        return trace

    def _unscored_logits(self, h: Tensor) -> Tensor:
        return torch.cat([head(h[:, k * 17:(k + 1) * 17])
                          for k, head in enumerate(self.heads)], dim=-1)

    def forward(self, x: Tensor, pre_index: int = 8) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != 2 or x.shape[1] not in (16, 32) or pre_index != 8:
            raise ValueError("expected [batch, 16 or 32, 2] and pre_index=8")
        h = x.new_zeros((x.shape[0], self.hidden))
        logw = x.new_full((x.shape[0], 4), -math.log(4)) if self.score_state_floats else None
        recurrent_weight = self.weight_hh * self.mask
        pre: Tensor | None = None
        for t in range(x.shape[1]):
            if logw is not None:
                logw = self._score_update(x[:, t], h, logw)
            h = self.step(x[:, t], h, recurrent_weight)
            if t + 1 == pre_index:
                pre = (logw if logw is not None else
                       self._unscored_logits(h) if self.name == "K4_unscored" else
                       self.readout(h))
        assert pre is not None
        final = (logw if logw is not None else
                 self._unscored_logits(h) if self.name == "K4_unscored" else
                 self.readout(h))
        return pre, final


def build_model(name: str, seed: int, graph_seed: int) -> PfstyleModel:
    """Initialize one arm from the registered PyTorch seed and fixed graph seed."""
    if seed < 0 or graph_seed < 0:
        raise ValueError("nonnegative model and graph seeds required")
    torch.manual_seed(seed)
    return PfstyleModel(name, graph_seed)
