"""Synthetic counterfactual delayed-evidence retrieval and answer revision."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

CANDIDATES = 4
CLASSES = 4
INPUT_SIZE = 13  # candidate 4, value 4, event kind 5
MEMORY, PRIOR, DISTRACTOR, QUERY, EVIDENCE = range(5)


@dataclass(frozen=True)
class Batch:
    x: Tensor
    pre_target: Tensor
    final_target: Tensor
    pre_index: int
    prior: Tensor
    selected: Tensor
    values: Tensor


def make_batch(pairs: int, delay: int, rng: np.random.Generator) -> Batch:
    """Adjacent confirm/revise examples share all input except late selector."""
    if pairs < 1 or delay < 0:
        raise ValueError("pairs >= 1 and delay >= 0 required")
    values = np.stack([rng.permutation(CLASSES) for _ in range(pairs)])
    order = np.stack([rng.permutation(CANDIDATES) for _ in range(pairs)])
    prior = rng.integers(CANDIDATES, size=pairs)
    alternative = (prior + rng.integers(1, CANDIDATES, size=pairs)) % CANDIDATES
    pre_index = CANDIDATES + 1 + delay
    length = pre_index + 5  # pre-query, evidence, 2 nuisance events, final query
    x = np.zeros((pairs, length, INPUT_SIZE), dtype=np.float32)
    row = np.arange(pairs)
    for t in range(CANDIDATES):
        candidate = order[:, t]
        x[row, t, candidate] = 1
        x[row, t, CANDIDATES + values[row, candidate]] = 1
        x[:, t, 8 + MEMORY] = 1
    x[row, CANDIDATES, prior] = 1
    x[:, CANDIDATES, 8 + PRIOR] = 1
    nuisance_times = list(range(CANDIDATES + 1, pre_index)) + [pre_index + 2, pre_index + 3]
    for t in nuisance_times:
        x[row, t, rng.integers(CANDIDATES, size=pairs)] = 1
        x[row, t, CANDIDATES + rng.integers(CLASSES, size=pairs)] = 1
        x[:, t, 8 + DISTRACTOR] = 1
    x[:, [pre_index, length - 1], 8 + QUERY] = 1
    x[:, pre_index + 1, 8 + EVIDENCE] = 1
    x = np.repeat(x, 2, axis=0)
    selected = np.stack((prior, alternative), axis=1).reshape(-1)
    x[np.arange(2 * pairs), pre_index + 1, selected] = 1
    repeated_values = np.repeat(values, 2, axis=0)
    repeated_prior = np.repeat(prior, 2)
    return Batch(
        torch.from_numpy(x),
        torch.from_numpy(repeated_values[np.arange(2 * pairs), repeated_prior]),
        torch.from_numpy(repeated_values[np.arange(2 * pairs), selected]),
        pre_index,
        torch.from_numpy(repeated_prior),
        torch.from_numpy(selected),
        torch.from_numpy(repeated_values),
    )
