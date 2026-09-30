"""Engineering-only calibration of posterior transfer across nuisance duration."""

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
from neurotopo.source_belief_task import (HYPOTHESES, INPUT_SIZE, REGIMES,
                                          SourceBeliefBatch, make_source_belief_batch)


MODEL_SEEDS = (301, 302, 303)
DELAYS = (1, 4, 8, 16)
UPDATES = 1500
BATCH_SIZE = 128


def _probabilities(model: RecurrentModel, batch: SourceBeliefBatch) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pres, finals = [], []
    with torch.no_grad():
        for start in range(0, len(batch.x), 256):
            pre, final = model(batch.x[start:start + 256], batch.pre_index)
            pres.append(F.softmax(pre, dim=1))
            finals.append(F.softmax(final, dim=1))
    return (torch.cat(pres).numpy().astype(np.float64),
            torch.cat(finals).numpy().astype(np.float64))


def _metrics(model: RecurrentModel, batch: SourceBeliefBatch) -> dict[str, float]:
    pre, final = _probabilities(model, batch)
    pre_oracle = batch.pre_joint.numpy().astype(np.float64)
    final_oracle = batch.final_joint.numpy().astype(np.float64)

    def kl(target: np.ndarray, predicted: np.ndarray) -> float:
        return float((target * (np.log(np.clip(target, 1e-12, 1)) -
                                np.log(np.clip(predicted, 1e-12, 1)))).sum(1).mean())

    truth_h, truth_r = batch.truth_h.numpy(), batch.truth_r.numpy()
    truth_joint = truth_h * REGIMES + truth_r
    predicted_h = final.reshape(-1, HYPOTHESES, REGIMES).sum(2)
    predicted_r = final.reshape(-1, HYPOTHESES, REGIMES).sum(1)
    return {
        "pre_joint_kl_to_oracle": kl(pre_oracle, pre),
        "final_joint_kl_to_oracle": kl(final_oracle, final),
        "pre_joint_oracle_map_agreement": float((pre.argmax(1) == pre_oracle.argmax(1)).mean()),
        "final_joint_oracle_map_agreement": float((final.argmax(1) == final_oracle.argmax(1)).mean()),
        "final_h_true_accuracy": float((predicted_h.argmax(1) == truth_h).mean()),
        "final_r_true_accuracy": float((predicted_r.argmax(1) == truth_r).mean()),
        "final_joint_true_accuracy": float((final.argmax(1) == truth_joint).mean()),
    }


def _oracle_metrics(batch: SourceBeliefBatch) -> dict[str, float]:
    pre = batch.pre_joint.numpy().astype(np.float64)
    final = batch.final_joint.numpy().astype(np.float64)
    pre_h = pre.reshape(-1, HYPOTHESES, REGIMES).sum(2)
    final_h = final.reshape(-1, HYPOTHESES, REGIMES).sum(2)
    final_r = final.reshape(-1, HYPOTHESES, REGIMES).sum(1)
    h, r = batch.truth_h.numpy(), batch.truth_r.numpy()
    fixed_zero = final.reshape(-1, HYPOTHESES, REGIMES)[:, :, 0].copy()
    fixed_zero /= fixed_zero.sum(1, keepdims=True)
    return {
        "pre_h_accuracy": float((pre_h.argmax(1) == h).mean()),
        "final_h_accuracy": float((final_h.argmax(1) == h).mean()),
        "final_r_accuracy": float((final_r.argmax(1) == r).mean()),
        "final_joint_accuracy": float((final.argmax(1) == h * REGIMES + r).mean()),
        "last_cue_h_accuracy": float((batch.evidence_cues[:, -1].numpy() == h).mean()),
        "fixed_source_zero_h_accuracy": float((fixed_zero.argmax(1) == h).mean()),
        "pre_only_final_h_accuracy": float((pre_h.argmax(1) == h).mean()),
        "h_map_change_rate": float((pre_h.argmax(1) != final_h.argmax(1)).mean()),
    }


def _loss(model: RecurrentModel, batch: SourceBeliefBatch) -> torch.Tensor:
    pre, final = model(batch.x, batch.pre_index)
    return 0.5 * (-(batch.pre_joint * F.log_softmax(pre, dim=1)).sum(1).mean()
                  - (batch.final_joint * F.log_softmax(final, dim=1)).sum(1).mean())


def _paired_batches(examples: int, seed: int) -> dict[int, SourceBeliefBatch]:
    batches = {delay: make_source_belief_batch(examples, np.random.default_rng(seed),
                                               distractors_per_evidence=delay)
               for delay in DELAYS}
    reference = batches[DELAYS[0]]
    for delay, batch in batches.items():
        for name in ("truth_h", "truth_r", "evidence_cues", "evidence_sources",
                     "pre_joint", "final_joint"):
            if not torch.equal(getattr(reference, name), getattr(batch, name)):
                raise ValueError(f"Pairing invariant failed for d={delay}, field={name}")
    return batches


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    configure(301, 1)
    validation = make_source_belief_batch(1024, np.random.default_rng(392001))
    evaluations = _paired_batches(8192, 393001)
    oracle = _oracle_metrics(evaluations[1])
    records = []
    checkpoint_dir = output.parent / "checkpoints" / "gate1_distractor_transfer_v1"
    if checkpoint_dir.exists():
        raise FileExistsError(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True)
    for seed in MODEL_SEEDS:
        configure(seed, 1)
        model = RecurrentModel(INPUT_SIZE, 64, HYPOTHESES * REGIMES, groups=1, topology="dense")
        initial_sha = digest(b"".join(p.detach().numpy().tobytes() for p in model.parameters()))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        rng = np.random.default_rng(391000 + seed)
        monitoring = []
        train_start = time.perf_counter()
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
                result = _metrics(model, validation)
                monitoring.append({"step": step, "validation": result,
                                   "train_batch_loss": float(loss.item())})
                print(json.dumps({"seed": seed, "step": step,
                                  "validation_final_kl": result["final_joint_kl_to_oracle"]}), flush=True)
        train_seconds = time.perf_counter() - train_start
        checkpoint = checkpoint_dir / f"seed-{seed}.pt"
        torch.save(model.state_dict(), checkpoint)
        final_sha = digest(b"".join(p.detach().numpy().tobytes() for p in model.parameters()))
        scores = {}
        for delay, batch in evaluations.items():
            start = time.perf_counter()
            metrics = _metrics(model, batch)
            scores[str(delay)] = {"metrics": metrics, "inference_seconds": time.perf_counter() - start,
                                  "resources": resource_account(model, batch.x.shape[1], BATCH_SIZE)}
        base_kl = scores["1"]["metrics"]["final_joint_kl_to_oracle"]
        for score in scores.values():
            score["delta_final_kl_vs_d1"] = score["metrics"]["final_joint_kl_to_oracle"] - base_kl
        records.append({"seed": seed, "initial_parameters_sha256": initial_sha,
                        "final_parameters_sha256": final_sha,
                        "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint.read_bytes()),
                        "training_seconds": train_seconds, "monitor": monitoring, "delays": scores})
    iid_learned = all(row["delays"]["1"]["metrics"]["final_joint_kl_to_oracle"] < 0.01 for row in records)
    eligible = [delay for delay in DELAYS[1:] if all(
        0.01 <= row["delays"][str(delay)]["metrics"]["final_joint_kl_to_oracle"] <= 0.20 and
        row["delays"][str(delay)]["metrics"]["pre_joint_kl_to_oracle"] <= 0.20 and
        row["delays"][str(delay)]["metrics"]["final_joint_oracle_map_agreement"] >= 0.70
        for row in records)]
    report = {
        "kind": "ENGINEERING_CALIBRATION_NOT_ARCHITECTURE_EVIDENCE",
        "plan": "research/design/gate1_distractor_transfer_calibration.md",
        "git_commit": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--porcelain"),
        "source_sha256": source_fingerprints(), "script_sha256": digest(Path(__file__).read_bytes()),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "streams": {"model_seeds": list(MODEL_SEEDS), "train_seed_formula": "391000 + model_seed",
                    "validation": 392001, "evaluation": 393001},
        "configuration": {"training_distractors_per_evidence": 1, "evaluation_distractors_per_evidence": list(DELAYS),
                          "examples_per_evaluation": 8192, "hidden": 64, "updates": UPDATES,
                          "batch_size": BATCH_SIZE, "optimizer": "Adam", "learning_rate": 0.003},
        "dataset_sha256": {str(delay): digest(batch.x.numpy().tobytes()) for delay, batch in evaluations.items()},
        "same_evidence_and_targets_across_delays": True,
        "oracle_and_shortcuts": oracle, "model_results": records,
        "iid_learning_guard": iid_learned,
        "eligible_ood_delays": eligible,
        "selected_smallest_eligible_delay": eligible[0] if iid_learned and eligible else None,
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/calibration/gate1_distractor_transfer_v1.json"))
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"output": str(args.output), "iid_learning_guard": result["iid_learning_guard"],
                      "selected_smallest_eligible_delay": result["selected_smallest_eligible_delay"]}), flush=True)


if __name__ == "__main__":
    main()
