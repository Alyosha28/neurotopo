"""The three registered G1-RS phase-reveal GRU arms.

All arms execute the same dense affine shapes. Masks are applied to sampled
parameters in the forward computation; they never alter allocation or RNG
consumption. ``pre_index`` denotes a one-based observation count (t=8).
"""

from __future__ import annotations

from hashlib import sha256
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import recurrent_mask


ARM_NAMES = ("RS4", "SC64", "D64_full")


class RoutedModel(nn.Module):
    """H64 GRU with registered routed or full input and recurrent connectivity."""

    input_size = 16
    hidden = 64
    classes = 4
    gates = 3

    def __init__(self, name: str, model_seed: int, graph_seed: int) -> None:
        super().__init__()
        if name not in ARM_NAMES:
            raise ValueError(f"unknown G1-RS arm: {name}")
        if model_seed < 0 or graph_seed < 0:
            raise ValueError("model and graph seeds must be nonnegative")
        self.name = name
        self.model_seed = model_seed
        self.graph_seed = graph_seed
        self.groups = 4 if name != "D64_full" else 1
        self.topology = {"RS4": "block", "SC64": "random",
                         "D64_full": "dense"}[name]

        # NumPy PCG64 graph construction does not consume the PyTorch stream.
        # The caller supplies the registered graph seed (14_000_000+model_seed).
        torch.manual_seed(model_seed)
        base_mask, graph_info = recurrent_mask(
            self.hidden, self.groups, self.topology, graph_seed)
        base_array = base_mask.numpy().astype(np.uint8, copy=False)
        self.graph_info: dict[str, Any] = {
            **graph_info,
            "mask_sha256": sha256(base_array.tobytes(order="C")).hexdigest(),
            "initial_nonself_edges": (int(base_mask.sum().item()) - self.hidden
                                      if name == "SC64" else 0),
            "required_swaps": (50 * (int(base_mask.sum().item()) - self.hidden)
                               if name == "SC64" else 0),
        }
        self.register_buffer("mask", base_mask.repeat(self.gates, 1))
        input_base = torch.zeros(self.hidden, self.input_size, dtype=torch.float32)
        if name == "D64_full":
            input_base.fill_(1)
        else:
            for group in range(4):
                input_base[group * 16:(group + 1) * 16,
                           group * 4:(group + 1) * 4] = 1
        self.register_buffer("input_mask", input_base.repeat(self.gates, 1))

        # Keep this exact Parameter creation/initialization order across arms.
        # In particular, masked entries are sampled rather than omitted.
        self.weight_ih = nn.Parameter(torch.empty(192, 16))
        self.weight_hh = nn.Parameter(torch.empty(192, 64))
        self.bias_ih = nn.Parameter(torch.empty(192))
        self.bias_hh = nn.Parameter(torch.empty(192))
        self.readout_weight = nn.Parameter(torch.empty(4, 64))
        self.readout_bias = nn.Parameter(torch.empty(4))
        input_degree = 16 if name == "D64_full" else 4
        recurrent_degree = 64 if name == "D64_full" else 16
        nn.init.uniform_(self.weight_ih, -1 / math.sqrt(input_degree),
                         1 / math.sqrt(input_degree))
        nn.init.uniform_(self.weight_hh, -1 / math.sqrt(recurrent_degree),
                         1 / math.sqrt(recurrent_degree))
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)
        nn.init.uniform_(self.readout_weight, -1 / math.sqrt(64),
                         1 / math.sqrt(64))
        nn.init.zeros_(self.readout_bias)

    def readout(self, h: Tensor) -> Tensor:
        """Full H64 to four-logit readout for every arm."""
        return F.linear(h, self.readout_weight, self.readout_bias)

    def _step(self, x_t: Tensor, h: Tensor, input_weight: Tensor,
              recurrent_weight: Tensor) -> Tensor:
        gi = F.linear(x_t, input_weight, self.bias_ih)
        gh = F.linear(h, recurrent_weight, self.bias_hh)
        ir, iz, inn = gi.chunk(3, dim=-1)
        hr, hz, hn = gh.chunk(3, dim=-1)
        r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        n = torch.tanh(inn + r * hn)
        return (1 - z) * n + z * h

    def step(self, x_t: Tensor, h: Tensor,
             recurrent_weight: Tensor | None = None) -> Tensor:
        """One registered GRU step; ``h`` can be a transplanted hidden state."""
        input_weight = self.weight_ih * self.input_mask
        weight = (self.weight_hh * self.mask if recurrent_weight is None
                  else recurrent_weight)
        return self._step(x_t, h, input_weight, weight)

    def _validate_sequence(self, x: Tensor, *, allow_empty: bool = False) -> None:
        if x.ndim != 3 or x.shape[-1] != self.input_size or x.shape[1] < int(not allow_empty):
            raise ValueError("expected [batch, time, 16] with a nonempty time axis")

    def hidden_after_prefix(self, x_prefix: Tensor) -> Tensor:
        """Return the hidden state after a prefix, e.g. 16 observations."""
        self._validate_sequence(x_prefix)
        h = x_prefix.new_zeros((x_prefix.shape[0], self.hidden))
        input_weight = self.weight_ih * self.input_mask
        recurrent_weight = self.weight_hh * self.mask
        for x_t in x_prefix.unbind(dim=1):
            h = self._step(x_t, h, input_weight, recurrent_weight)
        return h

    def replay_suffix(self, x_suffix: Tensor,
                      h_initial: Tensor) -> tuple[Tensor, Tensor]:
        """Run a fixed suffix from a supplied state; return logits and final h."""
        self._validate_sequence(x_suffix, allow_empty=True)
        if h_initial.shape != (x_suffix.shape[0], self.hidden):
            raise ValueError("expected h_initial [batch, 64]")
        h = h_initial
        input_weight = self.weight_ih * self.input_mask
        recurrent_weight = self.weight_hh * self.mask
        for x_t in x_suffix.unbind(dim=1):
            h = self._step(x_t, h, input_weight, recurrent_weight)
        return self.readout(h), h

    def run_trace(self, x: Tensor,
                  capture_times: tuple[int, ...] = (16, 48)) -> dict[str, Any]:
        """Capture post-token states and logits at one-based observation times.

        Captured block states are views of each H64 state, shaped [B, 4, 16].
        This diagnostic path performs no training or task-label operations.
        """
        self._validate_sequence(x)
        if len(set(capture_times)) != len(capture_times) or any(
            not 1 <= time <= x.shape[1] for time in capture_times
        ):
            raise ValueError("capture_times must be distinct one-based times within T")
        h = x.new_zeros((x.shape[0], self.hidden))
        input_weight = self.weight_ih * self.input_mask
        recurrent_weight = self.weight_hh * self.mask
        hidden: dict[int, Tensor] = {}
        logits: dict[int, Tensor] = {}
        for t, x_t in enumerate(x.unbind(dim=1), start=1):
            h = self._step(x_t, h, input_weight, recurrent_weight)
            if t in capture_times:
                hidden[t] = h
                logits[t] = self.readout(h)
        return {
            "hidden_by_time": hidden,
            "block_hidden_by_time": {
                t: state.reshape(state.shape[0], 4, 16)
                for t, state in hidden.items()
            },
            "logits_by_time": logits,
        }

    def forward(self, x: Tensor, pre_index: int = 8) -> tuple[Tensor, Tensor]:
        """Return logits after observation ``pre_index`` and after all T."""
        self._validate_sequence(x)
        if not 1 <= pre_index <= x.shape[1]:
            raise ValueError("pre_index must be a one-based time within the sequence")
        h = x.new_zeros((x.shape[0], self.hidden))
        input_weight = self.weight_ih * self.input_mask
        recurrent_weight = self.weight_hh * self.mask
        pre: Tensor | None = None
        for t, x_t in enumerate(x.unbind(dim=1), start=1):
            h = self._step(x_t, h, input_weight, recurrent_weight)
            if t == pre_index:
                pre = self.readout(h)
        assert pre is not None
        return pre, self.readout(h)


def build_model(name: str, model_seed: int, graph_seed: int) -> RoutedModel:
    """Build one registered arm; graph generation leaves model RNG untouched."""
    return RoutedModel(name, model_seed, graph_seed)
