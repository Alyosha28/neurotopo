"""Exploratory Gate 1 benchmark calibration; never issues an architecture decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from neurotopo.belief_task import (HYPOTHESES, INPUT_SIZE, RELIABILITIES,
                                   BeliefBatch, make_belief_batch)
from neurotopo.models import RecurrentModel
from neurotopo.resources import resource_account
from neurotopo.run import configure, digest, git_output, source_fingerprints, write_json


MODEL_SEEDS = (101, 102, 103)
UPDATES = 1500
BATCH_SIZE = 128
LEARNING_RATE = 0.003
MONITOR_STEPS = (500, 1000, 1500)


def _nll(probabilities: np.ndarray, truth: np.ndarray) -> float:
    return float(-np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-12, 1)).mean())


def _accuracy(probabilities: np.ndarray, truth: np.ndarray) -> float:
    return float((probabilities.argmax(1) == truth).mean())


def _last_cue_posterior(batch: BeliefBatch) -> np.ndarray:
    count = len(batch.truth)
    cues = batch.evidence_cues[:, -1].numpy()
    reliability = RELIABILITIES[batch.evidence_reliabilities[:, -1].numpy()]
    result = np.repeat(((1 - reliability) / (HYPOTHESES - 1))[:, None], HYPOTHESES, axis=1)
    result[np.arange(count), cues] = reliability
    return result


def oracle_metrics(batch: BeliefBatch) -> dict[str, float]:
    truth = batch.truth.numpy()
    pre = batch.pre_posterior.numpy().astype(np.float64)
    final = batch.final_posterior.numpy().astype(np.float64)
    last = _last_cue_posterior(batch)
    changes = pre.argmax(1) != final.argmax(1)
    return {
        "oracle_pre_accuracy": _accuracy(pre, truth),
        "oracle_final_accuracy": _accuracy(final, truth),
        "oracle_final_nll": _nll(final, truth),
        "last_cue_accuracy": _accuracy(last, truth),
        "last_cue_nll": _nll(last, truth),
        "prefix_only_final_accuracy": _accuracy(pre, truth),
        "prefix_only_final_nll": _nll(pre, truth),
        "uniform_final_accuracy": _accuracy(np.full_like(final, 1 / HYPOTHESES), truth),
        "uniform_final_nll": float(np.log(HYPOTHESES)),
        "oracle_map_change_rate": float(changes.mean()),
        "oracle_changed_examples": int(changes.sum()),
        "oracle_final_entropy": float(-(final * np.log(final)).sum(1).mean()),
    }


def _soft_loss(model: RecurrentModel, batch: BeliefBatch) -> torch.Tensor:
    pre, final = model(batch.x, batch.pre_index)
    pre_loss = -(batch.pre_posterior * F.log_softmax(pre, dim=1)).sum(1).mean()
    final_loss = -(batch.final_posterior * F.log_softmax(final, dim=1)).sum(1).mean()
    return 0.5 * (pre_loss + final_loss)


@torch.no_grad()
def model_metrics(model: RecurrentModel, batch: BeliefBatch) -> dict[str, float]:
    model.eval()
    before, after = [], []
    for start in range(0, len(batch.x), 256):
        pre, final = model(batch.x[start:start + 256], batch.pre_index)
        before.append(F.softmax(pre, dim=1))
        after.append(F.softmax(final, dim=1))
    predicted_pre = torch.cat(before).numpy().astype(np.float64)
    predicted_final = torch.cat(after).numpy().astype(np.float64)
    target_pre = batch.pre_posterior.numpy().astype(np.float64)
    target_final = batch.final_posterior.numpy().astype(np.float64)
    truth = batch.truth.numpy()

    def kl(target: np.ndarray, predicted: np.ndarray) -> float:
        return float((target * (np.log(np.clip(target, 1e-12, 1)) -
                                np.log(np.clip(predicted, 1e-12, 1)))).sum(1).mean())

    changed = target_pre.argmax(1) != target_final.argmax(1)
    return {
        "pre_kl_to_oracle": kl(target_pre, predicted_pre),
        "final_kl_to_oracle": kl(target_final, predicted_final),
        "pre_oracle_map_agreement": float((predicted_pre.argmax(1) == target_pre.argmax(1)).mean()),
        "final_oracle_map_agreement": float((predicted_final.argmax(1) == target_final.argmax(1)).mean()),
        "final_changed_subset_oracle_map_agreement": float((predicted_final.argmax(1)[changed] ==
                                                               target_final.argmax(1)[changed]).mean()),
        "pre_true_accuracy": _accuracy(predicted_pre, truth),
        "final_true_accuracy": _accuracy(predicted_final, truth),
        "final_true_nll": _nll(predicted_final, truth),
    }


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    configure(101, 1)
    analytic = make_belief_batch(100_000, np.random.default_rng(19001))
    validation = make_belief_batch(1_024, np.random.default_rng(192001))
    calibration = make_belief_batch(8_192, np.random.default_rng(193001))
    analytic_scores = oracle_metrics(analytic)
    calibration_oracle = oracle_metrics(calibration)
    results = []
    for seed in MODEL_SEEDS:
        configure(seed, 1)
        model = RecurrentModel(INPUT_SIZE, 64, HYPOTHESES, groups=1, topology="dense")
        initial_sha256 = digest(b"".join(p.detach().numpy().tobytes() for p in model.parameters()))
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        rng = np.random.default_rng(191000 + seed)
        monitors = []
        started = time.perf_counter()
        for step in range(1, UPDATES + 1):
            model.train()
            batch = make_belief_batch(BATCH_SIZE, rng)
            optimizer.zero_grad(set_to_none=True)
            loss = _soft_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite calibration loss, seed={seed}, step={step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step in MONITOR_STEPS:
                monitors.append({"step": step, "train_batch_loss": float(loss.item()),
                                 "validation": model_metrics(model, validation)})
                print(json.dumps({"seed": seed, "step": step,
                                  "validation_final_kl": monitors[-1]["validation"]["final_kl_to_oracle"]}), flush=True)
        elapsed = time.perf_counter() - started
        results.append({"seed": seed, "initial_parameters_sha256": initial_sha256,
                        "final_parameters_sha256": digest(b"".join(p.detach().numpy().tobytes() for p in model.parameters())),
                        "monitor": monitors, "calibration": model_metrics(model, calibration),
                        "runtime_seconds": elapsed,
                        "resources": resource_account(model, calibration.x.shape[1], BATCH_SIZE)})
    shortcut_ok = (calibration_oracle["oracle_final_accuracy"] - calibration_oracle["last_cue_accuracy"] >= 0.10
                   and calibration_oracle["oracle_map_change_rate"] >= 0.10)
    oracle_range_ok = 0.75 <= calibration_oracle["oracle_final_accuracy"] <= 0.95
    learned = all(0.01 <= row["calibration"]["final_kl_to_oracle"] <= 0.20 and
                  row["calibration"]["final_oracle_map_agreement"] >= 0.70 for row in results)
    report = {
        "kind": "ENGINEERING_CALIBRATION_NOT_ARCHITECTURE_EVIDENCE",
        "plan": "research/design/gate1_belief_benchmark_calibration.md",
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--porcelain"),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "source_sha256": source_fingerprints(),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "streams": {"analytic": 19001, "model_seeds": list(MODEL_SEEDS),
                    "train_seed_formula": "191000 + model_seed", "validation": 192001,
                    "calibration": 193001},
        "configuration": {"hypotheses": HYPOTHESES, "reliabilities": RELIABILITIES.tolist(),
                          "early_evidence": 3, "late_evidence": 3, "distractors_per_evidence": 1,
                          "hidden": 64, "cell": "dense_gru", "updates": UPDATES,
                          "batch_size": BATCH_SIZE, "optimizer": "Adam", "learning_rate": LEARNING_RATE},
        "dataset_sha256": {"validation": digest(validation.x.numpy().tobytes()),
                           "calibration": digest(calibration.x.numpy().tobytes())},
        "analytic_scores": analytic_scores,
        "calibration_oracle": calibration_oracle,
        "model_results": results,
        "diagnostic_guards": {"shortcut_gap_and_flip_rate": shortcut_ok,
                              "oracle_accuracy_range": oracle_range_ok,
                              "all_seed_learning_with_headroom": learned},
        "qualified_for_formal_design": bool(shortcut_ok and oracle_range_ok and learned),
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/calibration/gate1_belief_v1.json"))
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"output": str(args.output), "qualified_for_formal_design":
                      result["qualified_for_formal_design"], "guards": result["diagnostic_guards"]}), flush=True)


if __name__ == "__main__":
    main()
