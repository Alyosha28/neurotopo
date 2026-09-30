"""Exploratory hidden-source benchmark calibration, without architecture claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from neurotopo.models import RecurrentModel
from neurotopo.resources import resource_account
from neurotopo.run import configure, digest, git_output, source_fingerprints, write_json
from neurotopo.source_belief_task import (BAD_RELIABILITY, GOOD_RELIABILITY,
                                          HYPOTHESES, INPUT_SIZE, REGIMES,
                                          SourceBeliefBatch, make_source_belief_batch)


MODEL_SEEDS = (201, 202, 203)
UPDATES = 1500
BATCH_SIZE = 128


def _accuracy(probabilities: np.ndarray, truth: np.ndarray) -> float:
    return float((probabilities.argmax(1) == truth).mean())


def _nll(probabilities: np.ndarray, truth: np.ndarray) -> float:
    return float(-np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-12, 1)).mean())


def oracle_metrics(batch: SourceBeliefBatch) -> dict[str, float]:
    pre_joint = batch.pre_joint.numpy().astype(np.float64)
    final_joint = batch.final_joint.numpy().astype(np.float64)
    pre_h = pre_joint.reshape(-1, HYPOTHESES, REGIMES).sum(2)
    final = final_joint.reshape(-1, HYPOTHESES, REGIMES)
    final_h, final_r = final.sum(2), final.sum(1)
    truth_h, truth_r = batch.truth_h.numpy(), batch.truth_r.numpy()
    truth_joint = truth_h * REGIMES + truth_r
    fixed_source_zero = final[:, :, 0] / final[:, :, 0].sum(1, keepdims=True)
    return {
        "pre_h_accuracy": _accuracy(pre_h, truth_h),
        "final_h_accuracy": _accuracy(final_h, truth_h),
        "final_r_accuracy": _accuracy(final_r, truth_r),
        "final_joint_accuracy": _accuracy(final_joint, truth_joint),
        "final_joint_nll": _nll(final_joint, truth_joint),
        "final_joint_entropy": float(-(final_joint * np.log(final_joint)).sum(1).mean()),
        "last_cue_h_accuracy": float((batch.evidence_cues[:, -1].numpy() == truth_h).mean()),
        "fixed_source_zero_h_accuracy": _accuracy(fixed_source_zero, truth_h),
        "fixed_source_zero_h_nll": _nll(fixed_source_zero, truth_h),
        "pre_only_final_h_accuracy": _accuracy(pre_h, truth_h),
        "h_map_change_rate": float((pre_h.argmax(1) != final_h.argmax(1)).mean()),
    }


def _loss(model: RecurrentModel, batch: SourceBeliefBatch) -> torch.Tensor:
    pre, final = model(batch.x, batch.pre_index)
    return 0.5 * (-(batch.pre_joint * F.log_softmax(pre, dim=1)).sum(1).mean()
                  - (batch.final_joint * F.log_softmax(final, dim=1)).sum(1).mean())


@torch.no_grad()
def model_metrics(model: RecurrentModel, batch: SourceBeliefBatch) -> dict[str, float]:
    model.eval()
    pre_rows, final_rows = [], []
    for start in range(0, len(batch.x), 256):
        pre, final = model(batch.x[start:start + 256], batch.pre_index)
        pre_rows.append(F.softmax(pre, dim=1))
        final_rows.append(F.softmax(final, dim=1))
    pre = torch.cat(pre_rows).numpy().astype(np.float64)
    final = torch.cat(final_rows).numpy().astype(np.float64)
    target_pre = batch.pre_joint.numpy().astype(np.float64)
    target_final = batch.final_joint.numpy().astype(np.float64)

    def kl(target: np.ndarray, predicted: np.ndarray) -> float:
        return float((target * (np.log(np.clip(target, 1e-12, 1)) -
                                np.log(np.clip(predicted, 1e-12, 1)))).sum(1).mean())

    predicted_h = final.reshape(-1, HYPOTHESES, REGIMES).sum(2)
    predicted_r = final.reshape(-1, HYPOTHESES, REGIMES).sum(1)
    target_h = target_final.reshape(-1, HYPOTHESES, REGIMES).sum(2)
    truth_h, truth_r = batch.truth_h.numpy(), batch.truth_r.numpy()
    truth_joint = truth_h * REGIMES + truth_r
    return {
        "pre_joint_kl_to_oracle": kl(target_pre, pre),
        "final_joint_kl_to_oracle": kl(target_final, final),
        "final_joint_oracle_map_agreement": float((target_final.argmax(1) == final.argmax(1)).mean()),
        "final_h_oracle_map_agreement": float((target_h.argmax(1) == predicted_h.argmax(1)).mean()),
        "final_h_true_accuracy": _accuracy(predicted_h, truth_h),
        "final_r_true_accuracy": _accuracy(predicted_r, truth_r),
        "final_joint_true_accuracy": _accuracy(final, truth_joint),
        "final_joint_true_nll": _nll(final, truth_joint),
    }


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    configure(201, 1)
    analytic = make_source_belief_batch(100_000, np.random.default_rng(29001))
    validation = make_source_belief_batch(1_024, np.random.default_rng(292001))
    calibration = make_source_belief_batch(8_192, np.random.default_rng(293001))
    analytic_scores = oracle_metrics(analytic)
    calibration_oracle = oracle_metrics(calibration)
    results = []
    for seed in MODEL_SEEDS:
        configure(seed, 1)
        model = RecurrentModel(INPUT_SIZE, 64, HYPOTHESES * REGIMES, groups=1, topology="dense")
        initial_sha256 = digest(b"".join(p.detach().numpy().tobytes() for p in model.parameters()))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        rng = np.random.default_rng(291000 + seed)
        monitors = []
        started = time.perf_counter()
        for step in range(1, UPDATES + 1):
            model.train()
            batch = make_source_belief_batch(BATCH_SIZE, rng)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss seed={seed} step={step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step in (500, 1000, 1500):
                metrics = model_metrics(model, validation)
                monitors.append({"step": step, "train_batch_loss": float(loss.item()), "validation": metrics})
                print(json.dumps({"seed": seed, "step": step,
                                  "validation_final_joint_kl": metrics["final_joint_kl_to_oracle"]}), flush=True)
        results.append({"seed": seed, "initial_parameters_sha256": initial_sha256,
                        "final_parameters_sha256": digest(b"".join(p.detach().numpy().tobytes() for p in model.parameters())),
                        "monitor": monitors, "calibration": model_metrics(model, calibration),
                        "runtime_seconds": time.perf_counter() - started,
                        "resources": resource_account(model, calibration.x.shape[1], BATCH_SIZE)})
    shortcut_guard = (calibration_oracle["final_h_accuracy"] - calibration_oracle["last_cue_h_accuracy"] >= 0.10
                      and calibration_oracle["h_map_change_rate"] >= 0.10)
    learned_with_headroom = all(0.01 <= row["calibration"]["final_joint_kl_to_oracle"] <= 0.20 and
                                row["calibration"]["final_joint_oracle_map_agreement"] >= 0.70
                                for row in results)
    report = {
        "kind": "ENGINEERING_CALIBRATION_NOT_ARCHITECTURE_EVIDENCE",
        "plan": "research/design/gate1_belief_benchmark_calibration_v2.md",
        "git_commit": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--porcelain"),
        "source_sha256": source_fingerprints(), "script_sha256": digest(Path(__file__).read_bytes()),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "streams": {"analytic": 29001, "model_seeds": list(MODEL_SEEDS),
                    "train_seed_formula": "291000 + model_seed", "validation": 292001, "calibration": 293001},
        "configuration": {"hypotheses": HYPOTHESES, "regimes": REGIMES,
                          "good_reliability": GOOD_RELIABILITY, "bad_reliability": BAD_RELIABILITY,
                          "early_evidence": 4, "late_evidence": 4,
                          "distractors_per_evidence": 1, "hidden": 64, "cell": "dense_gru",
                          "updates": UPDATES, "batch_size": BATCH_SIZE,
                          "optimizer": "Adam", "learning_rate": 0.003},
        "dataset_sha256": {"validation": digest(validation.x.numpy().tobytes()),
                           "calibration": digest(calibration.x.numpy().tobytes())},
        "analytic_scores": analytic_scores, "calibration_oracle": calibration_oracle,
        "model_results": results,
        "diagnostic_guards": {"shortcut_gap_and_flip_rate": shortcut_guard,
                              "all_seed_learning_with_headroom": learned_with_headroom},
        "qualified_for_further_benchmark_design": bool(shortcut_guard and learned_with_headroom),
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/calibration/gate1_source_belief_v1.json"))
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"output": str(args.output), "qualified_for_further_benchmark_design":
                      result["qualified_for_further_benchmark_design"], "guards": result["diagnostic_guards"]}), flush=True)


if __name__ == "__main__":
    main()
