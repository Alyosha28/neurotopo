"""EXP-0011: frozen paired-seed analysis and bounded field-package decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .pfstyle_task import make_batch
from .run import source_fingerprints, write_json


ARMS = ("K4_score", "S68_sparse", "S68_dense", "S64_score", "K4_unscored")
MAIN = ("K4_score", "S68_sparse", "S68_dense")
STOP_STATUSES = {"RESOURCE_STOP", "TASK_STOP", "QUALIFICATION_STOP", "ENGINEERING_ABORT",
                 "INCOMPLETE_BUDGET"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_summary(primary: np.ndarray, dense: np.ndarray,
                   bootstrap_seed: int, samples: int) -> dict[str, Any]:
    """Use the same resampled training-seed indices for both paired contrasts."""
    if primary.shape != dense.shape or primary.ndim != 1 or len(primary) < 2:
        raise ValueError("matching paired seed vectors required")
    if not np.isfinite(primary).all() or not np.isfinite(dense).all():
        raise ValueError("nonfinite paired effects")
    rng = np.random.default_rng(bootstrap_seed)
    indexes = rng.integers(len(primary), size=(samples, len(primary)))
    def one(values: np.ndarray) -> dict[str, Any]:
        means = values[indexes].mean(axis=1)
        return {"per_seed": values.tolist(), "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}
    return {"sparse_minus_k4": one(primary), "dense_minus_k4": one(dense),
            "bootstrap_seed": bootstrap_seed, "bootstrap_samples": samples,
            "unit": "paired model/training seed with independent test stream per seed"}


def _recompute_saved_metrics(result: dict[str, Any], predictions: Path) -> dict[str, np.ndarray]:
    """Reject summary KL unless it follows from the saved logits and targets."""
    with np.load(predictions, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    expected = {f"{condition}_{field}" for condition in ("iid16", "ood32")
                for field in ("pre_logits", "final_logits", "pre_target", "final_target", "truth")}
    if set(arrays) != expected:
        raise ValueError("prediction archive fields differ from registered output")
    def one(condition: str, query: str) -> tuple[float, np.ndarray]:
        logits = np.asarray(arrays[f"{condition}_{query}_logits"], dtype=np.float64)
        target = np.asarray(arrays[f"{condition}_{query}_target"], dtype=np.float64)
        if (logits.ndim != 2 or logits.shape != target.shape or logits.shape[1] != 4
                or logits.shape[0] != result["metrics"][condition]["examples"]
                or not np.isfinite(logits).all() or not np.isfinite(target).all()
                or np.any(target < 0) or not np.allclose(target.sum(axis=1), 1, atol=1e-6)):
            raise ValueError(f"malformed saved {condition}/{query} predictions")
        centered = logits - logits.max(axis=1, keepdims=True)
        log_model = centered - np.log(np.exp(centered).sum(axis=1, keepdims=True))
        log_truth = np.log(np.clip(target, np.finfo(np.float64).tiny, 1.0))
        kl = np.where(target > 0, target * (log_truth - log_model), 0).sum(axis=1)
        if not np.isfinite(kl).all():
            raise ValueError("nonfinite saved prediction KL")
        registered = result["metrics"][condition][f"{query}_joint_kl_to_oracle"]
        if abs(float(kl.mean()) - registered) > 1e-8:
            raise ValueError(f"summary KL does not match saved {condition}/{query} logits")
        return float(kl.mean()), kl
    for condition in ("iid16", "ood32"):
        one(condition, "pre")
        one(condition, "final")
        final_logits = np.asarray(arrays[f"{condition}_final_logits"], dtype=np.float64)
        centered = final_logits - final_logits.max(axis=1, keepdims=True)
        logp = centered - np.log(np.exp(centered).sum(axis=1, keepdims=True))
        probabilities = np.exp(logp)
        map_fraction = [(np.argmax(final_logits, axis=1) == mode).mean()
                        for mode in range(4)]
        entropy = float(np.mean(-np.sum(probabilities * logp, axis=1)))
        reported = result["metrics"][condition]
        if (not np.allclose(map_fraction, reported["predicted_mode_map_fraction"],
                                rtol=0, atol=1e-12)
                or abs(entropy - reported["predicted_entropy_final"]) > 1e-8):
            raise ValueError(f"saved {condition} logits do not match interpretation diagnostics")
        predicted_digest = hashlib.sha256(b"".join(
            arrays[key].tobytes() for key in sorted(arrays) if key.startswith(condition + "_"))).hexdigest()
        if predicted_digest != result["metrics"][condition]["prediction_sha256"]:
            raise ValueError(f"saved {condition} prediction digest differs from summary")
    return arrays


def _registered_test_identity(config: dict[str, Any], phase: str, seed: int,
                              result: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    block = config[phase]
    for condition, length in (("iid16", 16), ("ood32", 32)):
        batch = make_batch(block["test_examples_per_length"], length,
                           block["test_root_base"] + seed, Path(config["matrix_file"]))
        if (not np.array_equal(arrays[f"{condition}_pre_target"], batch.pre_target.numpy())
                or not np.array_equal(arrays[f"{condition}_final_target"], batch.final_target.numpy())
                or not np.array_equal(arrays[f"{condition}_truth"], batch.truth.numpy())):
            raise ValueError(f"saved {phase}/{seed}/{condition} targets differ from registered stream")
        data_digest = hashlib.sha256(b"".join((batch.x.numpy().tobytes(),
                                               batch.pre_target.numpy().tobytes(),
                                               batch.final_target.numpy().tobytes(),
                                               batch.truth.numpy().tobytes(),
                                               batch.mode_means.numpy().tobytes()))).hexdigest()
        if result["metrics"][condition]["dataset_sha256"] != data_digest:
            raise ValueError(f"registered dataset hash mismatch {phase}/{seed}/{condition}")


def _read_arm(run: Path, phase: str, seed: int, arm: str,
              config: dict[str, Any]) -> dict[str, Any]:
    destination = run / phase / f"seed-{seed}" / arm
    sealed_path = destination.parent / "test_stream_manifest.json"
    sealed = json.loads(sealed_path.read_text())
    block = config[phase]
    if (sealed["phase"] != phase or sealed["seed"] != seed
            or sealed["root"] != block["test_root_base"] + seed
            or sealed["sealed_before_training"] is not True
            or set(sealed["streams"]) != {"iid16", "ood32"}):
        raise ValueError(f"unregistered or unsealed final stream {phase}/{seed}")
    result = json.loads((destination / "result.json").read_text())
    if (result["status"] != "COMPLETED" or result["phase"] != phase or
            result["seed"] != seed or result["arm"] != arm):
        raise ValueError(f"misidentified or incomplete {phase}/{seed}/{arm}")
    predictions = run / result["predictions_file"]
    if (result["checkpoint_sha256"] != _sha(run / result["checkpoint_file"]) or
            result["predictions_sha256"] != _sha(predictions)):
        raise ValueError(f"artifact digest mismatch {phase}/{seed}/{arm}")
    arrays = _recompute_saved_metrics(result, predictions)
    _registered_test_identity(config, phase, seed, result, arrays)
    for condition, length in (("iid16", 16), ("ood32", 32)):
        stream = sealed["streams"][condition]
        if (stream["examples"] != block["test_examples_per_length"]
                or stream["length"] != length
                or stream["dataset_sha256"] != result["metrics"][condition]["dataset_sha256"]):
            raise ValueError(f"test stream changed after pre-training seal {phase}/{seed}/{condition}")
    return result


def _guard(values: list[float], window: list[float]) -> bool:
    return all(window[0] <= value <= window[1] for value in values)


def analyze(run: Path) -> dict[str, Any]:
    worker_failure = run / "worker_failure.json"
    if worker_failure.is_file():
        failure = json.loads(worker_failure.read_text())
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        result = {"experiment_id": "EXP-0011", "version": "v6", "decision": "INCONCLUSIVE",
                  "substatus": failure["substatus"], "run_status": "WORKER_FAILURE",
                  "worker_failure": failure,
                  "manifest_sha256_if_present": _sha(manifest_path) if manifest_path.is_file() else None,
                  "research_updates": manifest.get("research_updates"),
                  "completed_records": manifest.get("records", []),
                  "explanation": "Worker failure or timeout precludes a complete registered paired effect; no Gate 1 credit."}
        write_json(run / "analysis.json", result)
        return result
    manifest = json.loads((run / "manifest.json").read_text())
    c = manifest["config"]
    if (c["experiment_id"], c["version"], c["workflow"]) != ("EXP-0011", "v6", "pfstyle_field"):
        raise ValueError("wrong registered analysis input")
    if manifest["config_sha256"] != _sha(Path("configs/exp0011_v6.json")):
        raise ValueError("registered config digest mismatch")
    if (manifest["matrix_json_sha256"] != c["matrix_json_sha256"]
            or _sha(Path(c["matrix_file"])) != c["matrix_json_sha256"]):
        raise ValueError("matrix provenance mismatch")
    preflight_path = Path("artifacts/validation/exp0011_preflight.json")
    receipt_path = preflight_path.with_name(preflight_path.stem + "_receipt.json")
    if (manifest["source_sha256"] != source_fingerprints()
            or manifest["preflight_sha256"] != _sha(preflight_path)
            or manifest["preflight_receipt_sha256"] != _sha(receipt_path)):
        raise ValueError("source or preflight provenance mismatch")
    receipt = json.loads(receipt_path.read_text())
    expected_budget_status = ("INCOMPLETE_BUDGET" if manifest["status"] == "RESOURCE_STOP"
                              and manifest.get("stop_reason") == "INCOMPLETE_BUDGET" else "PASS")
    if (receipt != manifest["preflight_receipt"]
            or receipt["preflight_report_sha256"] != manifest["preflight_sha256"]
            or receipt["budget_status"] != expected_budget_status):
        raise ValueError("preflight receipt budget or content mismatch")
    if manifest["status"] in STOP_STATUSES:
        if manifest["status"] in {"RESOURCE_STOP", "TASK_STOP", "QUALIFICATION_STOP"}:
            if (run / "formal").exists() or (run / "replication").exists():
                raise ValueError("formal or replication artifacts exist after a registered stop")
        if manifest["status"] == "RESOURCE_STOP" and (
                (run / "task_information.json").exists() or (run / "qualification").exists()):
            raise ValueError("resource stop accessed task or control data")
        if manifest["status"] == "TASK_STOP" and (run / "qualification").exists():
            raise ValueError("task information stop trained controls")
        result = {"experiment_id": "EXP-0011", "version": "v6", "decision": "INCONCLUSIVE",
                  "substatus": manifest.get("stop_reason", manifest["status"]),
                  "run_status": manifest["status"], "research_updates": manifest["research_updates"],
                  "task_information": manifest.get("task_information"),
                  "qualification": manifest.get("qualification"),
                  "completed_records": manifest["records"],
                  "explanation": "Registered staged gate stopped before a complete formal paired effect; no Gate 1 credit."}
        write_json(run / "analysis.json", result)
        return result
    if manifest["status"] != "COMPLETED" or manifest["research_updates"] != 97_200:
        raise ValueError("complete registered study or valid stop status required")
    if manifest["preflight"]["resource_status"] != "PASS":
        raise ValueError("completed run lacks resource qualification")
    q = c["qualification"]
    if manifest["task_information"]["status"] != "PASS" or manifest["qualification"]["status"] != "PASS":
        raise ValueError("completed run lacks task/control qualification")
    for phase, seeds, arms in (("qualification", q["seeds"], q["controls"]),
                               ("formal", c["formal"]["seeds"], ARMS),
                               ("replication", c["replication"]["seeds"], ARMS)):
        expected = {(phase, seed, arm) for seed in seeds for arm in arms}
        actual_rows = [(row["phase"], row["seed"], row["arm"]) for row in manifest["records"]
                       if row["phase"] == phase]
        if set(actual_rows) != expected or len(actual_rows) != len(expected):
            raise ValueError(f"{phase} record set differs from registered seeds and arms")
    results = {(phase, seed, arm): _read_arm(run, phase, seed, arm, c)
               for phase, seeds, arms in (("qualification", q["seeds"], q["controls"]),
                                           ("formal", c["formal"]["seeds"], ARMS),
                                           ("replication", c["replication"]["seeds"], ARMS))
               for seed in seeds for arm in arms}
    expected_updates = c["training"]["updates_per_arm_seed"]
    for (phase, seed, arm), result in results.items():
        if result["train_updates"] != expected_updates or result["train_examples"] != (
                expected_updates * c["training"]["batch_size"]):
            raise ValueError(f"training budget mismatch {phase}/{seed}/{arm}")
        expected_tokens = expected_updates * c["training"]["batch_size"] * c["task"]["iid_observations"]
        expected_linear_flops = (result["resources"]["iid16"]["dense_forward_linear_macs_per_example"]
                                 * 2 * c["training"]["batch_size"] * expected_updates)
        if (result["train_tokens"] != expected_tokens
                or result["training_forward_linear_flops_proxy"] != expected_linear_flops):
            raise ValueError(f"token or forward-linear compute account mismatch {phase}/{seed}/{arm}")
        for label in ("iid16", "ood32"):
            if not np.isfinite(result["metrics"][label]["final_joint_kl_to_oracle"]):
                raise ValueError(f"nonfinite KL {phase}/{seed}/{arm}/{label}")
    for phase, seeds, arms in (("qualification", q["seeds"], q["controls"]),
                               ("formal", c["formal"]["seeds"], ARMS),
                               ("replication", c["replication"]["seeds"], ARMS)):
        for seed in seeds:
            if len({results[phase, seed, arm]["training_data_sha256"] for arm in arms}) != 1:
                raise ValueError(f"unpaired training data {phase}/{seed}")
            for label in ("iid16", "ood32"):
                if len({results[phase, seed, arm]["metrics"][label]["dataset_sha256"]
                        for arm in arms}) != 1:
                    raise ValueError(f"unpaired test data {phase}/{seed}/{label}")
    formal_seeds = c["formal"]["seeds"]
    replication_seeds = c["replication"]["seeds"]
    def kl(phase: str, seed: int, arm: str, condition: str = "ood32") -> float:
        return results[phase, seed, arm]["metrics"][condition]["final_joint_kl_to_oracle"]
    sparse_effect = np.asarray([kl("formal", seed, "S68_sparse") - kl("formal", seed, "K4_score")
                                for seed in formal_seeds], dtype=np.float64)
    dense_effect = np.asarray([kl("formal", seed, "S68_dense") - kl("formal", seed, "K4_score")
                               for seed in formal_seeds], dtype=np.float64)
    paired = paired_summary(sparse_effect, dense_effect, c["formal"]["bootstrap_seed"],
                            c["formal"]["bootstrap_samples"])
    adequacy = all(kl("formal", seed, arm, "iid16") < q["iid_kl_each_seed_max_exclusive"]
                   for seed in formal_seeds for arm in MAIN)
    sparse_ood = [kl("formal", seed, "S68_sparse") for seed in formal_seeds]
    dense_ood = [kl("formal", seed, "S68_dense") for seed in formal_seeds]
    headroom = (sum(_guard([value], q["sparse_ood_each_kl_window"]) for value in sparse_ood)
                >= c["formal"]["sparse_seeds_in_window_min"]
                and _guard([float(np.mean(sparse_ood))], q["sparse_ood_mean_kl_window"])
                and _guard([float(np.mean(dense_ood))], q["dense_ood_mean_kl_window"]))
    rep_sparse = np.asarray([kl("replication", seed, "S68_sparse") -
                             kl("replication", seed, "K4_score") for seed in replication_seeds])
    rep_dense = np.asarray([kl("replication", seed, "S68_dense") -
                            kl("replication", seed, "K4_score") for seed in replication_seeds])
    replication_guard = bool(
        int(np.sum(rep_sparse > 0)) >= c["replication"]["positive_seeds_per_control_min"]
        and int(np.sum(rep_dense > 0)) >= c["replication"]["positive_seeds_per_control_min"]
        and rep_sparse.mean() > c["replication"]["mean_gain_each_control_min_exclusive"]
        and rep_dense.mean() > c["replication"]["mean_gain_each_control_min_exclusive"]
    )
    sparse_mean = paired["sparse_minus_k4"]["mean"]
    sparse_low, sparse_high = paired["sparse_minus_k4"]["ci95"]
    dense_mean = paired["dense_minus_k4"]["mean"]
    dense_low, _ = paired["dense_minus_k4"]["ci95"]
    useful = (sparse_mean >= c["formal"]["primary_minimum_mean_kl_gain"] and sparse_low > 0)
    if not adequacy or not headroom:
        decision = "INCONCLUSIVE"
        reason = "formal_learning_or_actual_control_headroom_failed"
    elif useful and dense_mean > 0 and dense_low > 0 and replication_guard:
        decision = "SUPPORTED"
        reason = "scoped_field_package_effect_with_strong_control_and_replication"
    elif useful:
        decision = "PARTIALLY_SUPPORTED"
        reason = "primary_field_package_effect_without_all_strong_guards"
    elif sparse_high < c["formal"]["primary_minimum_mean_kl_gain"]:
        decision = "FALSIFIED"
        reason = "registered_0.05nat_primary_useful_gain_excluded"
    else:
        decision = "INCONCLUSIVE"
        reason = "paired_effect_interval_did_not_resolve_registered_threshold"
    result = {
        "experiment_id": "EXP-0011", "version": "v6", "decision": decision, "reason": reason,
        "scope": "Fixed 8D task, field-reference PF-style architecture package; no project Gate 1 credit",
        "qualification": manifest["qualification"], "task_information": manifest["task_information"],
        "resource_status": manifest["preflight"]["resource_status"],
        "formal_iid_adequacy": adequacy, "formal_primary_control_headroom": headroom,
        "formal_sparse_ood_kl": sparse_ood, "formal_dense_ood_kl": dense_ood,
        "paired_effect": paired,
        "formal_interpretation_diagnostics_by_seed": {
            str(seed): {arm: {"predicted_mode_map_fraction": results["formal", seed, arm]["metrics"]["ood32"]["predicted_mode_map_fraction"],
                              "predicted_entropy_final": results["formal", seed, arm]["metrics"]["ood32"]["predicted_entropy_final"],
                              "score_diagnostics": results["formal", seed, arm]["metrics"]["ood32"].get("score_diagnostics")}
                        for arm in ARMS}
            for seed in formal_seeds
        },
        "replication_sparse_minus_k4": rep_sparse.tolist(),
        "replication_dense_minus_k4": rep_dense.tolist(),
        "replication_guard": replication_guard,
        "research_updates": manifest["research_updates"],
        "train_seconds_by_arm": {arm: sum(results[phase, seed, arm]["training_seconds_including_monitor"]
                                          for phase, seeds in (("formal", formal_seeds),
                                                               ("replication", replication_seeds))
                                          for seed in seeds) for arm in ARMS},
        "interpretation_limit": "No pure-K causal estimate, semantic hypothesis specialization or cross-task generalization",
    }
    write_json(run / "analysis.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    print(json.dumps({"decision": result["decision"], "run": str(args.run)}, sort_keys=True))


if __name__ == "__main__":
    main()
