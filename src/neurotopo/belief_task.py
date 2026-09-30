"""Small known-posterior task for calibrating evidence-update benchmarks."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


HYPOTHESES = 4
RELIABILITIES = np.array((0.45, 0.60, 0.75), dtype=np.float64)
EVIDENCE, DISTRACTOR, QUERY = range(3)
INPUT_SIZE = HYPOTHESES + len(RELIABILITIES) + 3


@dataclass(frozen=True)
class BeliefBatch:
    x: Tensor
    pre_posterior: Tensor
    final_posterior: Tensor
    truth: Tensor
    pre_index: int
    evidence_cues: Tensor
    evidence_reliabilities: Tensor


def _posterior(cues: np.ndarray, reliability_index: np.ndarray) -> np.ndarray:
    """Uniform-prior posterior for conditionally independent categorical cues."""
    rows, steps = cues.shape
    probabilities = RELIABILITIES[reliability_index]
    wrong = (1.0 - probabilities) / (HYPOTHESES - 1)
    log_weights = np.zeros((rows, HYPOTHESES), dtype=np.float64)
    row = np.arange(rows)
    for t in range(steps):
        log_weights += np.log(wrong[:, t])[:, None]
        log_weights[row, cues[:, t]] += np.log(probabilities[:, t] / wrong[:, t])
    log_weights -= log_weights.max(axis=1, keepdims=True)
    weights = np.exp(log_weights)
    return weights / weights.sum(axis=1, keepdims=True)


def make_belief_batch(examples: int, rng: np.random.Generator, *, pre_evidence: int = 3,
                      late_evidence: int = 3, distractors_per_evidence: int = 1) -> BeliefBatch:
    if examples < 1 or pre_evidence < 1 or late_evidence < 1 or distractors_per_evidence < 0:
        raise ValueError("positive examples/pre/late evidence and nonnegative distractors required")
    total = pre_evidence + late_evidence
    truth = rng.integers(HYPOTHESES, size=examples)
    reliability_index = rng.integers(len(RELIABILITIES), size=(examples, total))
    reliability = RELIABILITIES[reliability_index]
    other = rng.integers(HYPOTHESES - 1, size=(examples, total))
    other += other >= truth[:, None]
    cues = np.where(rng.random((examples, total)) < reliability, truth[:, None], other)
    distractor_cues = rng.integers(HYPOTHESES, size=(examples, total, distractors_per_evidence))
    distractor_rel = rng.integers(len(RELIABILITIES), size=(examples, total, distractors_per_evidence))

    pre_index = pre_evidence * (1 + distractors_per_evidence)
    length = total * (1 + distractors_per_evidence) + 2
    x = np.zeros((examples, length, INPUT_SIZE), dtype=np.float32)
    row = np.arange(examples)
    position = 0
    for t in range(total):
        if t == pre_evidence:
            x[:, position, HYPOTHESES + len(RELIABILITIES) + QUERY] = 1
            position += 1
        x[row, position, cues[:, t]] = 1
        x[row, position, HYPOTHESES + reliability_index[:, t]] = 1
        x[:, position, HYPOTHESES + len(RELIABILITIES) + EVIDENCE] = 1
        position += 1
        for d in range(distractors_per_evidence):
            x[row, position, distractor_cues[:, t, d]] = 1
            x[row, position, HYPOTHESES + distractor_rel[:, t, d]] = 1
            x[:, position, HYPOTHESES + len(RELIABILITIES) + DISTRACTOR] = 1
            position += 1
    x[:, position, HYPOTHESES + len(RELIABILITIES) + QUERY] = 1
    assert position == length - 1
    return BeliefBatch(
        x=torch.from_numpy(x),
        pre_posterior=torch.from_numpy(_posterior(cues[:, :pre_evidence], reliability_index[:, :pre_evidence]).astype(np.float32)),
        final_posterior=torch.from_numpy(_posterior(cues, reliability_index).astype(np.float32)),
        truth=torch.from_numpy(truth.astype(np.int64)),
        pre_index=pre_index,
        evidence_cues=torch.from_numpy(cues.astype(np.int64)),
        evidence_reliabilities=torch.from_numpy(reliability_index.astype(np.int64)),
    )
