"""EXP-0011 v6 clean-checkout, zero-update replay of frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .pfstyle_models import build_model
from .pfstyle_study import _batch_sha, _parameter_sha, evaluate, task_information
from .pfstyle_task import make_batch
from .run import configure, git_output, source_fingerprints


CONFIG_PATH = Path("configs/exp0011_v6.json")
MATRIX_VERIFY_PATH = Path("artifacts/validation/exp0011_matrix_verify.json")
PREFLIGHT_PATH = Path("artifacts/validation/exp0011_preflight.json")
PREFLIGHT_RECEIPT_PATH = Path("artifacts/validation/exp0011_preflight_receipt.json")
PAIR = ("K4_score", "S68_sparse")
REGISTERED_ROOTS = {"task_information": 8_050_001,
                    "qualification_seed": 12_101, "qualification_test": 8_200_000,
                    "formal_seed": 12_201, "formal_test": 8_300_000}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(report: dict[str, Any], label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        report["failures"].append({"check": label, "expected": expected, "observed": observed})


def _deadline(started: float, cap: float) -> None:
    if time.monotonic() - started > cap:
        raise TimeoutError("EXP-0011 replay exceeded registered wall-clock cap")


def _safe_artifact(run: Path, relative: str, expected: Path) -> Path:
    path = run / relative
    if path.resolve() != expected.resolve() or not path.is_file():
        raise ValueError(f"unsafe or missing replay artifact: {relative}")
    return path


def _registered_training_sha(config: dict[str, Any], seed: int,
                             started: float, cap: float) -> str:
    """Replay training inputs only; do not instantiate an optimizer or update weights."""
    hasher = hashlib.sha256()
    train = config["training"]
    for step in range(1, train["updates_per_arm_seed"] + 1):
        if step % 50 == 0:
            _deadline(started, cap)
        batch = make_batch(train["batch_size"], config["task"]["iid_observations"],
                           train["train_root_base"] + seed * 10_000 + step,
                           Path(config["matrix_file"]))
        hasher.update(bytes.fromhex(_batch_sha(batch)))
    return hasher.hexdigest()


def _replay_arm(run: Path, config: dict[str, Any], phase: str, seed: int,
                arm: str, report: dict[str, Any], started: float, cap: float,
                training_sha: str) -> None:
    label = f"{phase}/seed-{seed}/{arm}"
    directory = run / phase / f"seed-{seed}" / arm
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    report["original_hashes"][f"{label}/result.json"] = _sha(result_path)
    for field, expected in (("phase", phase), ("seed", seed), ("arm", arm),
                            ("status", "COMPLETED"),
                            ("train_updates", config["training"]["updates_per_arm_seed"])):
        _check(report, f"{label}/{field}", result.get(field), expected)
    _check(report, f"{label}/training_data_sha256", result.get("training_data_sha256"),
           training_sha)
    checkpoint = _safe_artifact(run, result["checkpoint_file"], directory / "final.pt")
    predictions = _safe_artifact(run, result["predictions_file"], directory / "predictions.npz")
    for kind, path in (("checkpoint", checkpoint), ("predictions", predictions)):
        actual = _sha(path)
        report["original_hashes"][f"{label}/{path.name}"] = actual
        _check(report, f"{label}/{kind}_sha256", actual, result[f"{kind}_sha256"])
    _deadline(started, cap)
    # A checkpoint with an invalid byte digest is never deserialized.
    if _sha(checkpoint) != result["checkpoint_sha256"]:
        return
    configure(seed, config["training"]["threads"])
    model = build_model(arm, seed, config["training"]["graph_seed_base"] + seed)
    _check(report, f"{label}/initial_parameters_sha256", _parameter_sha(model),
           result.get("initial_parameters_sha256"))
    _check(report, f"{label}/graph_mask_sha256",
           hashlib.sha256(model.mask.detach().cpu().numpy().tobytes()).hexdigest(),
           result.get("graph_mask_sha256"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    _check(report, f"{label}/final_parameters_sha256", _parameter_sha(model),
           result.get("final_parameters_sha256"))
    with np.load(predictions, allow_pickle=False) as saved:
        saved_names = set(saved.files)
        for condition, length in (("iid16", 16), ("ood32", 32)):
            _deadline(started, cap)
            block = config[phase]
            batch = make_batch(block["test_examples_per_length"], length,
                               block["test_root_base"] + seed, Path(config["matrix_file"]))
            metric, arrays = evaluate(model, batch)
            recorded = result["metrics"][condition]
            _check(report, f"{label}/{condition}/dataset_sha256",
                   _batch_sha(batch), recorded.get("dataset_sha256"))
            _check(report, f"{label}/{condition}/prediction_sha256",
                   metric["prediction_sha256"], recorded.get("prediction_sha256"))
            for key in sorted(metric):
                if key not in ("dataset_sha256", "prediction_sha256"):
                    _check(report, f"{label}/{condition}/{key}", metric[key], recorded.get(key))
            for key, array in arrays.items():
                name = f"{condition}_{key}"
                if name not in saved_names:
                    report["failures"].append({"check": f"{label}/{name}", "error": "missing saved array"})
                    continue
                observed = saved[name]
                if observed.dtype != array.dtype or observed.shape != array.shape or observed.tobytes() != array.tobytes():
                    report["failures"].append({"check": f"{label}/{name}", "error": "saved array differs byte-for-byte"})
                saved_names.remove(name)
        if saved_names:
            report["failures"].append({"check": f"{label}/predictions.npz",
                                       "error": "unexpected arrays", "names": sorted(saved_names)})
    report["replayed_arms"].append(label)


def _selected_phase(manifest: dict[str, Any], config: dict[str, Any]) -> tuple[str, int | None]:
    status = manifest["status"]
    if status == "RESOURCE_STOP":
        return "resource", None
    if status == "TASK_STOP":
        return "task_information", None
    records = {(row["phase"], row["seed"], row["arm"]) for row in manifest["records"]}
    formal = config["formal"]["seeds"]
    arms = config["models"]
    if all(("formal", seed, arm) in records for seed in formal for arm in arms):
        return "formal", formal[0]
    if status == "COMPLETED":
        raise ValueError("completed manifest lacks the full formal record set")
    qualification = config["qualification"]["seeds"]
    if all(("qualification", seed, arm) in records for seed in qualification
           for arm in config["qualification"]["controls"]):
        return "qualification", qualification[0]
    raise ValueError("no completed registered pair to replay")


def _verify(run: Path, report: dict[str, Any], started: float) -> None:
    if not run.is_absolute() or not run.is_dir():
        raise ValueError("--run must name an existing absolute run directory")
    if git_output("status", "--porcelain"):
        raise RuntimeError("EXP-0011 replay requires a clean checkout")
    report["git_commit"] = git_output("rev-parse", "HEAD")
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    report["original_hashes"]["manifest.json"] = _sha(manifest_path)
    config = json.loads(CONFIG_PATH.read_text())
    if (config["experiment_id"], config["version"], config["workflow"]) != (
            "EXP-0011", "v6", "pfstyle_field"):
        raise ValueError("wrong registered replay config")
    cap = config["resource"]["replay_wall_seconds_cap"]
    _check(report, "replay_wall_seconds_cap", cap, 300)
    _check(report, "manifest/config", manifest.get("config"), config)
    _check(report, "manifest/git_commit", report["git_commit"], manifest.get("git_commit"))
    _check(report, "task_information/root", config["task"]["task_information_root"],
           REGISTERED_ROOTS["task_information"])
    _check(report, "qualification/first_seed", config["qualification"]["seeds"][0],
           REGISTERED_ROOTS["qualification_seed"])
    _check(report, "qualification/test_root_base", config["qualification"]["test_root_base"],
           REGISTERED_ROOTS["qualification_test"])
    _check(report, "formal/first_seed", config["formal"]["seeds"][0],
           REGISTERED_ROOTS["formal_seed"])
    _check(report, "formal/test_root_base", config["formal"]["test_root_base"],
           REGISTERED_ROOTS["formal_test"])
    hashes = report["original_hashes"]
    hashes["config.json"] = _sha(CONFIG_PATH)
    hashes["matrix.json"] = _sha(Path(config["matrix_file"]))
    hashes["matrix_verify.json"] = _sha(MATRIX_VERIFY_PATH)
    hashes["preflight.json"] = _sha(PREFLIGHT_PATH)
    hashes["preflight_receipt.json"] = _sha(PREFLIGHT_RECEIPT_PATH)
    hashes["uv.lock"] = _sha(Path("uv.lock"))
    _check(report, "manifest/config_sha256", hashes["config.json"], manifest.get("config_sha256"))
    _check(report, "manifest/matrix_json_sha256", hashes["matrix.json"], manifest.get("matrix_json_sha256"))
    _check(report, "config/matrix_json_sha256", hashes["matrix.json"], config["matrix_json_sha256"])
    _check(report, "config/matrix_verify_report_sha256", hashes["matrix_verify.json"],
           config["matrix_verify_report_sha256"])
    _check(report, "manifest/preflight_sha256", hashes["preflight.json"], manifest.get("preflight_sha256"))
    _check(report, "manifest/preflight_receipt_sha256", hashes["preflight_receipt.json"],
           manifest.get("preflight_receipt_sha256"))
    _check(report, "manifest/lock_sha256", hashes["uv.lock"], manifest.get("lock_sha256"))
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    receipt = json.loads(PREFLIGHT_RECEIPT_PATH.read_text())
    _check(report, "manifest/preflight_receipt", manifest.get("preflight_receipt"), receipt)
    _check(report, "preflight_receipt/report_sha256", receipt.get("preflight_report_sha256"),
           hashes["preflight.json"])
    expected_preflight_budget = ("INCOMPLETE_BUDGET" if manifest["status"] == "RESOURCE_STOP"
                                 and manifest.get("stop_reason") == "INCOMPLETE_BUDGET" else "PASS")
    _check(report, "preflight_receipt/budget_status", receipt.get("budget_status"),
           expected_preflight_budget)
    _check(report, "preflight/config_sha256", preflight.get("config_sha256"), hashes["config.json"])
    _check(report, "preflight/matrix_json_sha256", preflight.get("matrix_json_sha256"), hashes["matrix.json"])
    report["source_sha256"] = source_fingerprints()
    _check(report, "manifest/source_sha256", report["source_sha256"], manifest.get("source_sha256"))
    _check(report, "preflight/source_sha256", preflight.get("source_sha256"), report["source_sha256"])
    _check(report, "manifest/research_updates", manifest.get("research_updates"),
           0 if manifest["status"] in ("RESOURCE_STOP", "TASK_STOP") else manifest.get("research_updates"))
    if manifest["status"] == "COMPLETED":
        _check(report, "completed/research_updates", manifest.get("research_updates"), 97_200)
    if manifest["status"] == "QUALIFICATION_STOP":
        _check(report, "qualification_stop/research_updates", manifest.get("research_updates"), 7_200)
    if report["failures"]:
        return
    phase, seed = _selected_phase(manifest, config)
    report["selected_phase"] = phase
    report["selected_seed"] = seed
    _deadline(started, cap)
    if phase == "resource":
        _check(report, "resource/preflight", manifest.get("preflight"),
               json.loads(PREFLIGHT_PATH.read_text()))
        if preflight.get("resource_status") == "PASS" and manifest.get("stop_reason") != "INCOMPLETE_BUDGET":
            report["failures"].append({"check": "resource/stop_reason",
                                       "error": "resource stop has a PASS preflight"})
        for forbidden in ("task_information.json", "qualification", "formal", "replication"):
            if (run / forbidden).exists():
                report["failures"].append({"check": f"resource/forbidden/{forbidden}",
                                           "error": "later-stage artifact exists"})
        return
    if phase == "task_information":
        info_path = run / "task_information.json"
        report["original_hashes"]["task_information.json"] = _sha(info_path)
        info = json.loads(info_path.read_text())
        _check(report, "task_information/manifest", manifest.get("task_information"), info)
        _check(report, "task_information/status", info.get("status"), "TASK_UNINFORMATIVE")
        for forbidden in ("qualification", "formal", "replication"):
            if (run / forbidden).exists():
                report["failures"].append({"check": f"task_information/forbidden/{forbidden}",
                                           "error": "later-stage artifact exists"})
        configure(config["task"]["task_information_root"], config["training"]["threads"])
        _check(report, "task_information/replay", task_information(config), info)
        _deadline(started, cap)
        return
    assert seed is not None
    training_sha = _registered_training_sha(config, seed, started, cap)
    report["recomputed_training_data_sha256"] = training_sha
    for arm in PAIR if phase == "formal" else config["qualification"]["controls"]:
        try:
            _replay_arm(run, config, phase, seed, arm, report, started, cap, training_sha)
        except TimeoutError:
            raise
        except Exception as error:
            report["failures"].append({"check": f"{phase}/seed-{seed}/{arm}",
                                       "error": f"{type(error).__name__}: {error}"})
    _deadline(started, cap)


def replay(run: Path, output: Path) -> dict[str, Any]:
    """Create exactly one immutable report; every checked mismatch is retained."""
    if output.exists():
        raise FileExistsError(output)
    receipt_path = output.with_name(output.stem + "_receipt.json")
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"experiment_id": "EXP-0011", "version": "v6",
                              "run": str(run), "original_hashes": {}, "source_sha256": None,
                              "selected_phase": None, "selected_seed": None,
                              "replayed_arms": [], "optimizer_updates": 0, "failures": []}
    started = time.monotonic()
    with output.open("x", encoding="utf-8") as stream:
        try:
            _verify(run, report, started)
        except TimeoutError as error:
            report["failures"].append({"check": "replay_wall_seconds", "error": str(error)})
            report["status"] = "INCOMPLETE_BUDGET"
        except Exception as error:
            report["failures"].append({"check": "replay", "error": f"{type(error).__name__}: {error}"})
        report["elapsed_seconds_before_report_fsync"] = time.monotonic() - started
        if report["elapsed_seconds_before_report_fsync"] > 300:
            report["failures"].append({"check": "replay_wall_seconds", "error": "exceeded 300 seconds"})
            report["status"] = "INCOMPLETE_BUDGET"
        report.setdefault("status", "PASS" if not report["failures"] else "FAIL")
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    elapsed_after_report_fsync = time.monotonic() - started
    receipt = {"id": "EXP-0011_REPLAY_RECEIPT",
               "replay_report_sha256": _sha(output),
               "seconds_after_report_fsync": elapsed_after_report_fsync,
               "budget_status": "PASS" if elapsed_after_report_fsync <= 300 else "INCOMPLETE_BUDGET",
               "publication_metadata_fsync_excluded": True}
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    report["receipt_path"] = str(receipt_path)
    report["receipt_budget_status"] = receipt["budget_status"]
    if receipt["budget_status"] != "PASS":
        report["status"] = "INCOMPLETE_BUDGET"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = replay(args.run, args.output)
    print(json.dumps({"status": report["status"], "selected_phase": report["selected_phase"],
                      "failures": len(report["failures"]), "output": str(args.output),
                      "receipt": report["receipt_path"]}, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
