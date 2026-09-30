"""Matched-compute engineering calibration of fixed versus varied nuisance cadence."""

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


SEEDS = (401, 402, 403)
EVAL_DELAYS = (1, 4, 8, 16)
UPDATES = 1500
BATCH_SIZE = 128


def varied_schedule() -> list[int]:
    rng = np.random.default_rng(490001)
    result: list[int] = []
    for _ in range(5):
        block = np.repeat(np.arange(3), 100)
        rng.shuffle(block)
        result.extend(int(value) for value in block)
    return result


def paired_batches(examples: int, seed: int) -> dict[int, SourceBeliefBatch]:
    batches = {d: make_source_belief_batch(examples, np.random.default_rng(seed),
                                           distractors_per_evidence=d)
               for d in EVAL_DELAYS}
    base = batches[1]
    for d, batch in batches.items():
        for field in ("truth_h", "truth_r", "evidence_cues", "evidence_sources", "pre_joint", "final_joint"):
            if not torch.equal(getattr(base, field), getattr(batch, field)):
                raise ValueError(f"Evaluation pairing failed: d={d}, field={field}")
    return batches


def same_informative_episode(a: SourceBeliefBatch, b: SourceBeliefBatch) -> bool:
    return all(torch.equal(getattr(a, field), getattr(b, field)) for field in
               ("truth_h", "truth_r", "evidence_cues", "evidence_sources", "pre_joint", "final_joint"))


def loss_on(model: RecurrentModel, batch: SourceBeliefBatch) -> torch.Tensor:
    pre, final = model(batch.x, batch.pre_index)
    return 0.5 * (-(batch.pre_joint * F.log_softmax(pre, dim=1)).sum(1).mean()
                  - (batch.final_joint * F.log_softmax(final, dim=1)).sum(1).mean())


def oracle_metrics(batch: SourceBeliefBatch) -> dict[str, float]:
    pre = batch.pre_joint.numpy().astype(np.float64).reshape(-1, HYPOTHESES, REGIMES)
    final = batch.final_joint.numpy().astype(np.float64).reshape(-1, HYPOTHESES, REGIMES)
    early_h, late_h = pre.sum(2), final.sum(2)
    truth_h, truth_r = batch.truth_h.numpy(), batch.truth_r.numpy()
    return {"pre_h_accuracy": float((early_h.argmax(1) == truth_h).mean()),
            "final_h_accuracy": float((late_h.argmax(1) == truth_h).mean()),
            "final_r_accuracy": float((final.sum(1).argmax(1) == truth_r).mean()),
            "final_joint_accuracy": float((final.reshape(-1, HYPOTHESES * REGIMES).argmax(1) ==
                                           truth_h * REGIMES + truth_r).mean()),
            "h_map_change_rate": float((early_h.argmax(1) != late_h.argmax(1)).mean())}


@torch.no_grad()
def metrics(model: RecurrentModel, batch: SourceBeliefBatch) -> dict[str, float]:
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

    def kl(target: np.ndarray, prediction: np.ndarray) -> float:
        return float((target * (np.log(np.clip(target, 1e-12, 1)) -
                                np.log(np.clip(prediction, 1e-12, 1)))).sum(1).mean())

    truth_h, truth_r = batch.truth_h.numpy(), batch.truth_r.numpy()
    h = final.reshape(-1, HYPOTHESES, REGIMES).sum(2)
    r = final.reshape(-1, HYPOTHESES, REGIMES).sum(1)
    return {"pre_joint_kl_to_oracle": kl(target_pre, pre),
            "final_joint_kl_to_oracle": kl(target_final, final),
            "final_joint_oracle_map_agreement": float((final.argmax(1) == target_final.argmax(1)).mean()),
            "final_h_true_accuracy": float((h.argmax(1) == truth_h).mean()),
            "final_r_true_accuracy": float((r.argmax(1) == truth_r).mean()),
            "final_joint_true_accuracy": float((final.argmax(1) == truth_h * REGIMES + truth_r).mean())}


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    configure(401, 1)
    schedule = varied_schedule()
    assert len(schedule) == UPDATES
    for end in range(300, UPDATES + 1, 300):
        assert sum(schedule[end - 300:end]) == 300
    validation = make_source_belief_batch(1024, np.random.default_rng(492001))
    evaluations = paired_batches(8192, 493001)
    oracle = oracle_metrics(evaluations[1])
    checkpoint_dir = output.parent / "checkpoints" / "gate1_cadence_v1"
    if checkpoint_dir.exists():
        raise FileExistsError(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True)
    records = []
    for seed in SEEDS:
        models, optimizers, starts = {}, {}, {}
        for arm in ("fixed", "varied"):
            configure(seed, 1)
            model = RecurrentModel(INPUT_SIZE, 64, HYPOTHESES * REGIMES, groups=1, topology="dense")
            models[arm] = model
            optimizers[arm] = torch.optim.Adam(model.parameters(), lr=0.003)
            starts[arm] = digest(b"".join(p.detach().numpy().tobytes() for p in model.parameters()))
        if starts["fixed"] != starts["varied"]:
            raise ValueError("Initial model parameters differ between paired arms")
        token_counts = {arm: 0 for arm in models}
        forward_flops = {arm: 0 for arm in models}
        train_seconds = {arm: 0.0 for arm in models}
        monitors = {arm: [] for arm in models}
        for step in range(1, UPDATES + 1):
            data_seed = 491000 + seed * 10_000 + step
            batches = {"fixed": make_source_belief_batch(BATCH_SIZE, np.random.default_rng(data_seed)),
                       "varied": make_source_belief_batch(BATCH_SIZE, np.random.default_rng(data_seed),
                                                           distractors_per_evidence=schedule[step - 1])}
            if not same_informative_episode(batches["fixed"], batches["varied"]):
                raise ValueError(f"Training data pairing failed seed={seed} step={step}")
            for arm in ("fixed", "varied"):
                model = models[arm]
                batch = batches[arm]
                started = time.perf_counter()
                model.train()
                optimizers[arm].zero_grad(set_to_none=True)
                loss = loss_on(model, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss seed={seed} arm={arm} step={step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizers[arm].step()
                train_seconds[arm] += time.perf_counter() - started
                token_counts[arm] += len(batch.x) * batch.x.shape[1]
                forward_flops[arm] += resource_account(model, batch.x.shape[1], len(batch.x))[
                    "dense_forward_linear_flops_per_batch"]
                if step in (300, 900, 1500):
                    value = metrics(model, validation)
                    monitors[arm].append({"step": step, "validation": value,
                                          "train_batch_loss": float(loss.item())})
                    print(json.dumps({"seed": seed, "arm": arm, "step": step,
                                      "validation_final_kl": value["final_joint_kl_to_oracle"]}), flush=True)
            if step % 300 == 0 and (token_counts["fixed"] != token_counts["varied"] or
                                    forward_flops["fixed"] != forward_flops["varied"]):
                raise ValueError(f"Cumulative token/compute mismatch seed={seed} step={step}")
        rows = {}
        for arm, model in models.items():
            checkpoint = checkpoint_dir / f"seed-{seed}-{arm}.pt"
            torch.save(model.state_dict(), checkpoint)
            scores = {}
            for d, batch in evaluations.items():
                started = time.perf_counter()
                value = metrics(model, batch)
                scores[str(d)] = {"metrics": value, "inference_seconds": time.perf_counter() - started,
                                  "resources": resource_account(model, batch.x.shape[1], BATCH_SIZE)}
            rows[arm] = {"initial_parameters_sha256": starts[arm],
                         "final_parameters_sha256": digest(b"".join(p.detach().numpy().tobytes() for p in model.parameters())),
                         "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint.read_bytes()),
                         "train_tokens": token_counts[arm], "train_forward_linear_flops": forward_flops[arm],
                         "train_seconds": train_seconds[arm], "monitor": monitors[arm], "delays": scores}
        records.append({"seed": seed, "arms": rows})
    iid_ok = all(all(row["arms"][arm]["delays"]["1"]["metrics"]["final_joint_kl_to_oracle"] < 0.01
                     for arm in ("fixed", "varied")) for row in records)
    effect_ok = all(
        0.01 <= row["arms"]["varied"]["delays"]["4"]["metrics"]["final_joint_kl_to_oracle"] <= 0.20
        and row["arms"]["varied"]["delays"]["4"]["metrics"]["final_joint_oracle_map_agreement"] >= 0.70
        and row["arms"]["fixed"]["delays"]["4"]["metrics"]["final_joint_kl_to_oracle"] > 0.20
        and row["arms"]["varied"]["delays"]["4"]["metrics"]["final_joint_kl_to_oracle"] -
            row["arms"]["fixed"]["delays"]["4"]["metrics"]["final_joint_kl_to_oracle"] <= -0.50
        for row in records)
    report = {
        "kind": "ENGINEERING_CALIBRATION_NOT_ARCHITECTURE_EVIDENCE",
        "plan": "research/design/gate1_cadence_calibration.md",
        "git_commit": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--porcelain"),
        "source_sha256": source_fingerprints(), "script_sha256": digest(Path(__file__).read_bytes()),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "streams": {"model_seeds": list(SEEDS), "schedule_seed": 490001,
                    "train_seed_formula": "491000 + model_seed*10000 + step",
                    "validation": 492001, "evaluation": 493001},
        "configuration": {"updates_per_arm": UPDATES, "batch_size": BATCH_SIZE,
                          "fixed_d": 1, "varied_d_counts": {"0": 500, "1": 500, "2": 500},
                          "evaluation_d": list(EVAL_DELAYS), "optimizer": "Adam", "learning_rate": 0.003},
        "schedule_sha256": digest(np.asarray(schedule, dtype=np.int8).tobytes()),
        "evaluation_dataset_sha256": {str(d): digest(batch.x.numpy().tobytes()) for d, batch in evaluations.items()},
        "training_pairing_and_block_compute_verified": True,
        "oracle": oracle,
        "model_results": records, "iid_learning_guard": iid_ok,
        "varied_d4_effect_and_headroom_guard": effect_ok,
        "qualified_for_further_benchmark_design": bool(iid_ok and effect_ok),
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/calibration/gate1_cadence_v1.json"))
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"output": str(args.output), "qualified_for_further_benchmark_design":
                      result["qualified_for_further_benchmark_design"],
                      "iid_learning_guard": result["iid_learning_guard"],
                      "varied_d4_effect_and_headroom_guard": result["varied_d4_effect_and_headroom_guard"]}), flush=True)


if __name__ == "__main__":
    main()
