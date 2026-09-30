"""Known-oracle inference over content and an unobserved source regime."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


HYPOTHESES = 4
REGIMES = 2
GOOD_RELIABILITY = 0.80
BAD_RELIABILITY = 0.40
EVIDENCE, DISTRACTOR, QUERY = range(3)
INPUT_SIZE = HYPOTHESES + REGIMES + 3


@dataclass(frozen=True)
class SourceBeliefBatch:
    x: Tensor
    pre_joint: Tensor
    final_joint: Tensor
    truth_h: Tensor
    truth_r: Tensor
    pre_index: int
    evidence_cues: Tensor
    evidence_sources: Tensor


def _joint_posterior(cues: np.ndarray, sources: np.ndarray) -> np.ndarray:
    count, steps = cues.shape
    log_weights = np.zeros((count, HYPOTHESES, REGIMES), dtype=np.float64)
    rows = np.arange(count)
    for t in range(steps):
        for regime in range(REGIMES):
            reliability = np.where(sources[:, t] == regime, GOOD_RELIABILITY, BAD_RELIABILITY)
            wrong = (1 - reliability) / (HYPOTHESES - 1)
            log_weights[:, :, regime] += np.log(wrong)[:, None]
            log_weights[rows, cues[:, t], regime] += np.log(reliability / wrong)
    log_weights -= log_weights.max(axis=(1, 2), keepdims=True)
    weights = np.exp(log_weights)
    weights /= weights.sum(axis=(1, 2), keepdims=True)
    return weights.reshape(count, HYPOTHESES * REGIMES)


def _balanced_sources(count: int, rng: np.random.Generator) -> np.ndarray:
    pattern = np.array((0, 0, 1, 1), dtype=np.int64)
    order = np.argsort(rng.random((count, 4)), axis=1)
    return pattern[order]


def make_source_belief_batch(examples: int, rng: np.random.Generator,
                             *, distractors_per_evidence: int = 1) -> SourceBeliefBatch:
    if examples < 1 or distractors_per_evidence < 0:
        raise ValueError("positive examples and nonnegative distractors required")
    total, early = 8, 4
    truth_h = rng.integers(HYPOTHESES, size=examples)
    truth_r = rng.integers(REGIMES, size=examples)
    sources = np.concatenate((_balanced_sources(examples, rng),
                              _balanced_sources(examples, rng)), axis=1)
    reliability = np.where(sources == truth_r[:, None], GOOD_RELIABILITY, BAD_RELIABILITY)
    other = rng.integers(HYPOTHESES - 1, size=(examples, total))
    other += other >= truth_h[:, None]
    cues = np.where(rng.random((examples, total)) < reliability, truth_h[:, None], other)
    distractor_cues = rng.integers(HYPOTHESES, size=(examples, total, distractors_per_evidence))
    distractor_sources = rng.integers(REGIMES, size=(examples, total, distractors_per_evidence))

    pre_index = early * (1 + distractors_per_evidence)
    length = total * (1 + distractors_per_evidence) + 2
    x = np.zeros((examples, length, INPUT_SIZE), dtype=np.float32)
    rows = np.arange(examples)
    position = 0
    for t in range(total):
        if t == early:
            x[:, position, HYPOTHESES + REGIMES + QUERY] = 1
            position += 1
        x[rows, position, cues[:, t]] = 1
        x[rows, position, HYPOTHESES + sources[:, t]] = 1
        x[:, position, HYPOTHESES + REGIMES + EVIDENCE] = 1
        position += 1
        for d in range(distractors_per_evidence):
            x[rows, position, distractor_cues[:, t, d]] = 1
            x[rows, position, HYPOTHESES + distractor_sources[:, t, d]] = 1
            x[:, position, HYPOTHESES + REGIMES + DISTRACTOR] = 1
            position += 1
    x[:, position, HYPOTHESES + REGIMES + QUERY] = 1
    assert position == length - 1
    return SourceBeliefBatch(
        x=torch.from_numpy(x),
        pre_joint=torch.from_numpy(_joint_posterior(cues[:, :early], sources[:, :early]).astype(np.float32)),
        final_joint=torch.from_numpy(_joint_posterior(cues, sources).astype(np.float32)),
        truth_h=torch.from_numpy(truth_h.astype(np.int64)),
        truth_r=torch.from_numpy(truth_r.astype(np.int64)),
        pre_index=pre_index,
        evidence_cues=torch.from_numpy(cues.astype(np.int64)),
        evidence_sources=torch.from_numpy(sources.astype(np.int64)),
    )
