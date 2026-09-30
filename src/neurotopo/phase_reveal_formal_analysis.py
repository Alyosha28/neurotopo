"""Array-first EXP-0014 v3 analysis and EXP-0013 source recomputation."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .phase_reveal_models import build_model
from .phase_reveal_task import (PhaseRevealBatch, cue_only_posterior,
                                last_frame_posterior, make_batch,
                                truncated_prehistory_posterior, validate_oracles)
from .phase_reveal_qualification import _array_hash, _state_hash, configure_cpu


CONFIG_SHA256 = "912065f0d38cea31cecef94bb933a56d35c4e56534276c520a39f6d1ad42041d"


def kl_rows(oracle: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Per-example KL(p_oracle || softmax(logits)) in float64."""
    p = np.asarray(oracle, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 4 or z.shape != p.shape:
        raise ValueError("expected matching [examples, 4] oracle and logits")
    if (not np.isfinite(p).all() or not np.isfinite(z).all()
            or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0)):
        raise ValueError("invalid oracle probabilities or logits")
    maximum = z.max(axis=1, keepdims=True)
    log_q = z - maximum - np.log(np.exp(z - maximum).sum(axis=1, keepdims=True))
    log_p = np.zeros_like(p)
    np.log(p, out=log_p, where=p > 0)
    return np.sum(np.where(p > 0, p * (log_p - log_q), 0), axis=1)


def paired_bootstrap(values: np.ndarray, seed: int, repeats: int = 20_000,
                     indices: np.ndarray | None = None) -> dict[str, Any]:
    """Seed-paired percentile interval; callers share ``indices`` across arms."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all() or repeats < 1:
        raise ValueError("finite one-dimensional seed values required")
    if indices is None:
        indices = np.random.Generator(np.random.PCG64(seed)).integers(
            0, x.size, size=(repeats, x.size))
    if indices.shape != (repeats, x.size) or np.any(indices < 0) or np.any(indices >= x.size):
        raise ValueError("invalid common bootstrap indices")
    means = x[indices].mean(axis=1)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "ci95": [float(v) for v in np.quantile(means, [0.025, 0.975])],
            "seed": seed, "resamples": repeats}


def correction_mask(H: np.ndarray, cue: np.ndarray,
                    oracle_pre: np.ndarray, oracle_final: np.ndarray) -> np.ndarray:
    """Registered wrong-cue, early cue-MAP, late true-H revision subset."""
    h, c = np.asarray(H), np.asarray(cue)
    if h.ndim != 1 or c.shape != h.shape or oracle_pre.shape != (h.size, 4) or oracle_final.shape != (h.size, 4):
        raise ValueError("invalid correction slice arrays")
    return ((h != c) & (oracle_pre.argmax(axis=1) == c)
            & (oracle_final.argmax(axis=1) == h))


def _safe_relative(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path must be project-relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("artifact escapes project")
    return resolved


def _npz(root: Path, path: str, digest: str,
         fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    file = _safe_relative(root, path)
    if not file.is_file() or file.is_symlink() or sha256(file.read_bytes()).hexdigest() != digest:
        raise ValueError(f"missing or changed raw array: {path}")
    with np.load(file, allow_pickle=False) as data:
        if not set(fields) <= set(data.files):
            raise ValueError(f"missing raw fields: {path}")
        return {name: np.array(data[name]) for name in fields}


def _replay_control_prediction(model: Any, x: np.ndarray,
                               arrays: dict[str, np.ndarray]) -> None:
    """Zero-update forward replay binds archived logits to regenerated input."""
    replay_pre, replay_final = [], []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            lp, lf = model(torch.from_numpy(x[start:start + 512]), pre_index=8)
            replay_pre.append(lp.numpy().astype(np.float64))
            replay_final.append(lf.numpy().astype(np.float64))
    if (not np.array_equal(arrays["logits_pre"], np.concatenate(replay_pre))
            or not np.array_equal(arrays["logits_final"], np.concatenate(replay_final))):
        raise ValueError("qualification logits differ on frozen-root input replay")


def recheck_qualification(root: Path, config: dict[str, Any],
                          manifest: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    """Recompute selected control LR and disjoint qualification metrics.

    Any unavailable array, source ambiguity, altered root or missing numeric
    status is a hard failure.  The independent decision remains a separate gate.
    """
    q = config["qualification"]
    seeds = q["model_seeds"]
    menu = q["lr_menu"]
    if manifest.get("experiment_id") != "EXP-0013" or manifest.get("version") != "v3":
        raise ValueError("not the frozen qualification manifest")
    if manifest.get("config") != config or manifest.get("gate1_claim") is not False:
        raise ValueError("qualification manifest does not embed frozen config/boundary")
    if manifest.get("research_updates") != config["resource"]["research_update_cap"]:
        raise ValueError("qualification research update count differs")
    if manifest.get("status") != "COMPLETED_PENDING_ANALYSIS":
        raise ValueError("qualification manifest is not complete")
    if (_safe_relative(root, str(Path(config["output"]) / "RS4"))).exists():
        raise ValueError("qualification output contains RS4 task artifacts")
    if manifest.get("rs4_task_updates") != 0:
        raise ValueError("qualification trained RS4")
    if analysis.get("qualification_status") != "NUMERIC_GUARDS_PASS_PENDING_REVIEW":
        raise ValueError("qualification analysis is not a passing pending-review result")
    if analysis.get("suggested_decision") != "INCONCLUSIVE" or analysis.get("review_state") != "NOT_REVIEWED":
        raise ValueError("qualification analysis preempted independent decision")
    arms = manifest.get("arms", {})
    if set(arms) != {"SC64", "D64_full"}:
        raise ValueError("qualification arms do not match frozen controls")
    matrix = _safe_relative(root, config["matrix_file"])
    if sha256(matrix.read_bytes()).hexdigest() != config["matrix_sha256"]:
        raise ValueError("qualification matrix SHA differs")
    configure_cpu()
    generated: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    loaded_models: dict[tuple[str, float, int], Any] = {}
    result: dict[str, Any] = {}
    for arm in ("SC64", "D64_full"):
        candidates: dict[float, dict[str, list[float]]] = {}
        if set(arms[arm]) != {str(lr) for lr in menu}:
            raise ValueError(f"{arm} LR menu differs")
        for lr in menu:
            records = arms[arm][str(lr)]
            if set(records) != {str(seed) for seed in seeds}:
                raise ValueError(f"{arm} seed menu differs")
            values = {split: [] for split in ("tune_iid", "tune_ood", "qual_iid", "qual_ood")}
            for seed in seeds:
                row = records[str(seed)]
                if (row.get("arm") != arm or row.get("seed") != seed
                        or row.get("lr") != lr or row.get("status") != "COMPLETED"
                        or row.get("updates") != config["training"]["updates_per_arm_seed"]):
                    raise ValueError("qualification update count differs")
                if [cp.get("step") for cp in row.get("checkpoints", [])] != config["training"]["monitor_steps"]:
                    raise ValueError("qualification monitor checkpoint steps differ")
                for cp in row["checkpoints"]:
                    for file_key, sha_key in (("checkpoint_file", "checkpoint_sha256"),
                                              ("prediction_file", "prediction_sha256")):
                        path = _safe_relative(root, str(Path(config["output"]) / arm
                                                        / str(lr) / str(seed) / cp[file_key]))
                        if sha256(path.read_bytes()).hexdigest() != cp[sha_key]:
                            raise ValueError("qualification checkpoint/monitor hash differs")
                evaluations = row.get("evaluations", {})
                required = {"tune_iid", "tune_ood"}
                if float(lr) == float(manifest.get("selected_lr", {}).get(arm, -1)):
                    required |= {"qual_iid", "qual_ood"}
                if set(evaluations) != required:
                    raise ValueError("qualification split missing or extra")
                for split in required:
                    entry = evaluations[split]
                    base = q["tune_iid_root_base" if split == "tune_iid" else
                             "tune_ood_root_base" if split == "tune_ood" else
                             "qual_iid_root_base" if split == "qual_iid" else
                             "qual_ood_root_base"]
                    if entry.get("root") != base + seed or entry.get("examples") != q["eval_examples"]:
                        raise ValueError("qualification frozen evaluation root/count mismatch")
                    prediction_file = str(Path(config["output"]) / arm / str(lr) / str(seed)
                                          / entry["predictions_file"])
                    arrays = _npz(root, prediction_file, entry["predictions_sha256"],
                                  ("oracle_pre", "oracle_final", "logits_pre", "logits_final"))
                    if arrays["oracle_final"].shape != (q["eval_examples"], 4):
                        raise ValueError("qualification evaluation shape mismatch")
                    cache_key = (split, seed)
                    if cache_key not in generated:
                        length = (config["task"]["train_length"] if split.endswith("iid")
                                  else config["task"]["ood_length"])
                        rebuilt = make_batch(q["eval_examples"], length, base + seed, matrix,
                                             cue_reliability=config["task"]["cue_reliability"],
                                             switch_step=config["task"]["switch_step"])
                        generated[cache_key] = (rebuilt.x, rebuilt.oracle_pre,
                                                rebuilt.oracle_final)
                    x_rebuilt, pre_rebuilt, final_rebuilt = generated[cache_key]
                    if entry.get("input_sha256") != _array_hash(x_rebuilt):
                        raise ValueError("qualification eval input hash differs from frozen root")
                    if (not np.array_equal(arrays["oracle_pre"], pre_rebuilt)
                            or not np.array_equal(arrays["oracle_final"], final_rebuilt)):
                        raise ValueError("qualification eval oracle differs from frozen root")
                    model_key = (arm, float(lr), seed)
                    if model_key not in loaded_models:
                        model = build_model(arm, seed,
                                            config["model"]["graph_seed_base"] + seed)
                        final_cp = row["checkpoints"][-1]
                        cp_path = _safe_relative(root, str(Path(config["output"]) / arm
                                                          / str(lr) / str(seed)
                                                          / final_cp["checkpoint_file"]))
                        checkpoint = torch.load(cp_path, map_location="cpu", weights_only=True)
                        if (checkpoint["arm"] != arm or checkpoint["seed"] != seed
                                or checkpoint["step"] != config["training"]["updates_per_arm_seed"]):
                            raise ValueError("qualification final checkpoint metadata differs")
                        model.load_state_dict(checkpoint["state_dict"])
                        if _state_hash(model) != row["final_state_sha256"]:
                            raise ValueError("qualification loaded final state differs")
                        loaded_models[model_key] = model
                    _replay_control_prediction(loaded_models[model_key], x_rebuilt, arrays)
                    measured = float(kl_rows(arrays["oracle_final"], arrays["logits_final"]).mean())
                    pre = float(kl_rows(arrays["oracle_pre"], arrays["logits_pre"]).mean())
                    if not np.isclose(measured, entry["final_kl"], rtol=0, atol=1e-10) or not np.isclose(
                        pre, entry["pre_kl"], rtol=0, atol=1e-10
                    ):
                        raise ValueError("qualification summary differs from original predictions")
                    values[split].append(measured)
            candidates[float(lr)] = values
        eligible = [lr for lr in menu if all(v < q["iid_kl_per_seed_max_exclusive"]
                    for v in candidates[float(lr)]["tune_iid"])
                    and np.mean(candidates[float(lr)]["tune_iid"]) < q["iid_kl_mean_max_exclusive"]]
        if not eligible:
            raise ValueError(f"{arm} no eligible LR")
        chosen = min(eligible, key=lambda lr: (np.mean(candidates[float(lr)]["tune_ood"]), lr))
        if float(manifest.get("selected_lr", {}).get(arm, -1)) != float(chosen):
            raise ValueError(f"{arm} selected LR is not independent best OOD")
        qualified = candidates[float(chosen)]
        iid = qualified["qual_iid"]
        ood = qualified["qual_ood"]
        if not all(v < q["iid_kl_per_seed_max_exclusive"] for v in iid) or not np.mean(iid) < q["iid_kl_mean_max_exclusive"]:
            raise ValueError(f"{arm} did not learn on independent qualification IID")
        low, high = q["ood_mean_windows"][arm]
        seed_low, seed_high = q["ood_per_seed_windows"][arm]
        if not (low <= np.mean(ood) <= high and all(seed_low <= v <= seed_high for v in ood)):
            raise ValueError(f"{arm} not in independent headroom window")
        result[arm] = {"chosen_lr": float(chosen), "qual_iid": iid, "qual_ood": ood}
    if analysis.get("controls", {}).get("selected_lr") != {arm: result[arm]["chosen_lr"] for arm in result}:
        raise ValueError("qualification analysis and raw selected LRs differ")
    generation = manifest.get("generation_p95_seconds")
    if not isinstance(generation, (int, float)) or not np.isfinite(generation) or generation <= 0:
        raise ValueError("missing measured task generation p95")
    task = manifest.get("task_information", {})
    if task.get("pass") is not True:
        raise ValueError("qualification task-information source is not PASS")
    task_file = str(Path(config["output"]) / task["arrays_file"])
    raw_task = _npz(root, task_file, task["arrays_sha256"],
                    ("H", "cue", "p_pre", "p_full", "p_cue", "p_last", "p_truncated",
                     "retained_H", "retained_cue", "retained_pre", "retained_full",
                     "retained_cue_posterior"))
    _replay_task_information(raw_task, config, matrix, task)
    _recheck_task_information(raw_task, config)
    return {"status": "NUMERIC_GUARDS_PASS_PENDING_REVIEW",
            "selected_lr": {arm: result[arm]["chosen_lr"] for arm in result},
            "controls": result, "data_generation_p95_seconds": float(generation)}


def _replay_task_information(raw: dict[str, np.ndarray], config: dict[str, Any],
                             matrix: Path, record: dict[str, Any]) -> None:
    """Bytewise replay of both registered task-information roots and oracles."""
    info, task = config["task_information"], config["task"]
    if record.get("root") != info["root"] or record.get("retained_root") != info["retained_root"]:
        raise ValueError("task-information root metadata differs")
    main = make_batch(info["examples"], task["ood_length"], info["root"], matrix,
                      cue_reliability=task["cue_reliability"], switch_step=task["switch_step"])
    retained = make_batch(info["retained_examples"], task["ood_length"],
                          info["retained_root"], matrix,
                          cue_reliability=task["retained_cue_reliability"],
                          switch_step=task["retained_switch_step"])
    main_prior = cue_only_posterior(main.cue, task["cue_reliability"])
    main_last = last_frame_posterior(main.u, main.y, main.cue, matrix,
                                     switch_step=task["switch_step"],
                                     cue_reliability=task["cue_reliability"])
    main_trunc = truncated_prehistory_posterior(main.u, main.y, main.cue, matrix,
                                               switch_step=task["switch_step"],
                                               cue_reliability=task["cue_reliability"])
    retained_prior = cue_only_posterior(retained.cue, task["retained_cue_reliability"])
    expected = {"H": main.H, "cue": main.cue, "p_pre": main.oracle_pre,
                "p_full": main.oracle_final, "p_cue": main_prior,
                "p_last": main_last, "p_truncated": main_trunc,
                "retained_H": retained.H, "retained_cue": retained.cue,
                "retained_pre": retained.oracle_pre,
                "retained_full": retained.oracle_final,
                "retained_cue_posterior": retained_prior}
    for name, value in expected.items():
        if not np.array_equal(raw[name], value):
            raise ValueError(f"qualification task-information {name} differs from frozen root")
    # The task-information NPZ does not archive x/u/eps/y. Rebuild them from
    # the fixed roots and independently check the source/oracle implementation.
    subset = PhaseRevealBatch(*(getattr(main, field)[:8] for field in
                                ("x", "oracle_pre", "oracle_final", "H", "cue", "u", "eps", "y")),
                              main.switch_step, main.cue_reliability)
    reference = validate_oracles(subset, matrix)
    if reference != record.get("oracle_check"):
        raise ValueError("qualification independent oracle receipt differs")


def _prob_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("invalid task posterior arrays")
    if np.any(p < 0) or np.any(q <= 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0) or not np.allclose(q.sum(axis=1), 1, atol=1e-10, rtol=0):
        raise ValueError("task posterior probabilities malformed")
    return np.sum(np.where(p > 0, p * (np.log(np.maximum(p, 1e-300)) - np.log(q)), 0), axis=1)


def _bootstrap_lower_large(values: np.ndarray, rng: np.random.Generator, repeats: int) -> float:
    draws = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 64):
        end = min(repeats, start + 64)
        indices = rng.integers(0, len(values), size=(end - start, len(values)))
        draws[start:end] = values[indices].mean(axis=1)
    return float(np.quantile(draws, 0.025))


def _recheck_task_information(raw: dict[str, np.ndarray], config: dict[str, Any]) -> None:
    info = config["task_information"]
    task = config["task"]
    H, cue = raw["H"], raw["cue"]
    full, pre, prior = raw["p_full"], raw["p_pre"], raw["p_cue"]
    if H.shape != (info["examples"],) or cue.shape != H.shape or full.shape != (len(H), 4):
        raise ValueError("task information size mismatch")
    fractions = np.bincount(H, minlength=4) / len(H)
    if (not all(info["h_fraction_window"][0] <= f <= info["h_fraction_window"][1] for f in fractions)
            or not info["cue_hit_window"][0] <= np.mean(H == cue) <= info["cue_hit_window"][1]
            or np.max(np.abs(pre - prior)) > 1e-10
            or np.mean(_prob_kl(full, np.full_like(full, 0.25))) < info["oracle_uniform_kl_min"]):
        raise ValueError("task information balance/oracle guard failed")
    wrong = int(np.sum(H != cue))
    corrected = int(np.sum(correction_mask(H, cue, pre, full)))
    z = 1.959963984540054
    p = corrected / wrong
    wilson = (p + z*z/(2*wrong) - z*np.sqrt(p*(1-p)/wrong + z*z/(4*wrong*wrong))) / (1+z*z/wrong)
    if corrected < info["wrong_cue_correction_count_min"] or wilson < info["wrong_cue_correction_wilson_lower_min"]:
        raise ValueError("task wrong-cue correction guard failed")
    rng = np.random.Generator(np.random.PCG64(info["bootstrap_seed"]))
    for name, field in (("cue", "p_cue"), ("last", "p_last"),
                        ("truncated_prehistory", "p_truncated")):
        lower = _bootstrap_lower_large(_prob_kl(full, raw[field]), rng, info["bootstrap_resamples"])
        if lower < info["shortcut_kl_ci_lower_min"][name]:
            raise ValueError(f"task {name} shortcut guard failed")
    rH, rcue = raw["retained_H"], raw["retained_cue"]
    if (rH.shape != (info["retained_examples"],)
            or np.sum(correction_mask(rH, rcue, raw["retained_pre"], raw["retained_full"]))
            < info["retained_correction_count_min"]
            or np.mean(_prob_kl(raw["retained_full"], raw["retained_cue_posterior"]))
            < info["retained_cue_kl_min"]):
        raise ValueError("retained task information guard failed")


def linear_cka_blocks(h: np.ndarray, denominator_min: float = 1e-12) -> dict[str, Any]:
    """Four t48 state blocks: variance and maximum pairwise linear CKA."""
    data = np.asarray(h, dtype=np.float64)
    if data.ndim != 3 or data.shape[1:] != (4, 16) or not np.isfinite(data).all():
        raise ValueError("expected finite [examples,4,16] hidden states")
    centered = data - data.mean(axis=0, keepdims=True)
    variance = centered.var(axis=0).mean(axis=1)
    ckas = []
    for i in range(4):
        x = centered[:, i]
        for j in range(i + 1, 4):
            y = centered[:, j]
            denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
            if denominator <= denominator_min:
                return {"variance_by_block": variance.tolist(), "max_cka": None,
                        "collapsed": True}
            ckas.append(float(np.linalg.norm(x.T @ y) ** 2 / denominator))
    return {"variance_by_block": variance.tolist(), "max_cka": max(ckas),
            "collapsed": False}


def classify_complete(primary_ci: tuple[float, float], dense_ci: tuple[float, float],
                      correction_ci: tuple[float, float], *, operation: str,
                      retained_positive: bool, replication_positive: bool,
                      learned_and_headroom: bool, raw_complete: bool) -> dict[str, str]:
    """Frozen v3 priority after all per-seed quantities were recomputed."""
    if not raw_complete:
        return {"decision": "INCONCLUSIVE", "substatus": "MISSING_RAW_ARRAYS"}
    if not learned_and_headroom:
        return {"decision": "INCONCLUSIVE", "substatus": "LEARNING_OR_HEADROOM_FAILURE"}
    if primary_ci[1] < 0.05:
        return {"decision": "FALSIFIED", "substatus": "PRIMARY_USEFUL_GAIN_FALSIFIED"}
    if primary_ci[0] > 0.05:
        if operation != "PASS":
            return {"decision": "INCONCLUSIVE", "substatus": "OPERATION_FAILED_OR_UNRESOLVED"}
        if (dense_ci[0] > 0 and correction_ci[0] > 0.03
                and retained_positive and replication_positive):
            return {"decision": "SUPPORTED", "substatus": "BOUNDED_ARCHITECTURE_EFFECT"}
        return {"decision": "PARTIALLY_SUPPORTED", "substatus": "LOCAL_EFFECT_GUARD_FAILED"}
    return {"decision": "INCONCLUSIVE", "substatus": "PRIMARY_CI_BORDERLINE"}


def _archived_npz(run: Path, relative: str, digest: str,
                  fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    return _npz(run, relative, digest, fields)


def _read_sealed(run: Path, record: dict[str, Any], expected_root: int,
                 expected_examples: int, expected_length: int,
                 matrix: Path, expected_switch: int,
                 expected_reliability: float) -> dict[str, np.ndarray]:
    if (record.get("root") != expected_root or record.get("examples") != expected_examples
            or record.get("length") != expected_length):
        raise ValueError("sealed stream root/shape metadata differs")
    raw = _archived_npz(run, record["file"], record["sha256"],
                        ("x", "oracle_pre", "oracle_final", "H", "cue", "u", "eps", "y"))
    if raw["x"].shape != (expected_examples, expected_length, 16):
        raise ValueError("sealed x has incorrect shape")
    if any(raw[key].shape != (expected_examples, 4) for key in ("oracle_pre", "oracle_final")):
        raise ValueError("sealed oracle has incorrect shape")
    if raw["H"].shape != (expected_examples,) or raw["cue"].shape != (expected_examples,):
        raise ValueError("sealed metadata has incorrect shape")
    if raw["u"].shape != (expected_examples, expected_length, 4):
        raise ValueError("sealed u has incorrect shape")
    if raw["y"].shape != raw["u"].shape or raw["eps"].shape != raw["u"].shape:
        raise ValueError("sealed y/eps has incorrect shape")
    if (record.get("switch_step") != expected_switch
            or record.get("cue_reliability") != expected_reliability):
        raise ValueError("sealed task parameter metadata differs")
    from .phase_reveal_task import make_batch

    regenerated = make_batch(expected_examples, expected_length, expected_root, matrix,
                             cue_reliability=expected_reliability,
                             switch_step=expected_switch)
    for key in raw:
        if not np.array_equal(raw[key], getattr(regenerated, key)):
            raise ValueError(f"sealed {key} differs from frozen root replay")
    return raw


def _read_prediction(run: Path, row: dict[str, Any], split: str,
                     sealed: dict[str, Any], raw: dict[str, np.ndarray]) -> dict[str, Any]:
    record = row["evaluations"][split]
    if record["root"] != sealed["root"] or record["examples"] != sealed["examples"]:
        raise ValueError("prediction root/example mismatch")
    arrays = _archived_npz(run, record["predictions_file"], record["predictions_sha256"],
                           ("oracle_pre", "oracle_final", "logits_pre", "logits_final"))
    for key in ("oracle_pre", "oracle_final"):
        if not np.array_equal(arrays[key], raw[key]):
            raise ValueError("prediction target differs from sealed oracle")
    pre = float(kl_rows(arrays["oracle_pre"], arrays["logits_pre"]).mean())
    final = float(kl_rows(arrays["oracle_final"], arrays["logits_final"]).mean())
    if (not np.isclose(pre, record["pre_kl"], rtol=0, atol=1e-10)
            or not np.isclose(final, record["final_kl"], rtol=0, atol=1e-10)):
        raise ValueError("archived prediction summary differs from raw arrays")
    return {"pre_kl": pre, "final_kl": final, "arrays": arrays}


def _operation_seed(run: Path, row: dict[str, Any], oracle: np.ndarray,
                    rs_logits: np.ndarray, cfg: dict[str, Any],
                    diagnostic: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    op = cfg["operation"]
    records = row.get("operation", {})
    if set(records) != {"0", "1", "2", "3"}:
        raise ValueError("four archived operation blocks required")
    summaries: dict[str, Any] = {}
    reference_h48: np.ndarray | None = None
    for block in range(4):
        record = records[str(block)]
        if record.get("permutation_seed") != op["prefix_permutation_seed_base"] + block:
            raise ValueError("operation permutation seed differs")
        permutation = np.random.Generator(np.random.PCG64(record["permutation_seed"])).permutation(len(oracle))
        if record.get("permutation_sha256") != sha256(np.asarray(permutation, dtype="<i8").tobytes()).hexdigest():
            raise ValueError("operation permutation hash differs")
        arrays = _archived_npz(run, str(Path(row["operation_dir"]) / record["arrays_file"]),
                               record["arrays_sha256"],
                               ("original_logits", "transplant_logits", "lesion_logits",
                                "h16_original", "h16_transplanted", "h48_original",
                                "h48_transplanted"))
        orig = arrays["original_logits"]
        if not np.allclose(orig, rs_logits, atol=1e-6, rtol=0):
            raise ValueError("operation original logits differ from OOD predictions")
        h16, h16cf = arrays["h16_original"], arrays["h16_transplanted"]
        h48, h48cf = arrays["h48_original"], arrays["h48_transplanted"]
        if any(value.shape != (len(oracle), 64) for value in (h16, h16cf, h48, h48cf)):
            raise ValueError("operation state shape differs")
        if reference_h48 is None:
            reference_h48 = h48
        elif not np.array_equal(reference_h48, h48):
            raise ValueError("operation original state differs across block records")
        if diagnostic is not None and (
            not np.allclose(h16, diagnostic["h16"], atol=1e-6, rtol=0)
            or not np.allclose(h48, diagnostic["h48"], atol=1e-6, rtol=0)
        ):
            raise ValueError("operation original state differs from archived OOD trace")
        lo, hi = 16 * block, 16 * (block + 1)
        target_change = float(np.linalg.norm(h16cf[:, lo:hi] - h16[:, lo:hi], axis=1).mean() / 4)
        untouched16 = (h16cf - h16).copy()
        untouched16[:, lo:hi] = 0
        untouched48 = (h48cf - h48).copy()
        untouched48[:, lo:hi] = 0
        centered = orig - arrays["transplant_logits"]
        centered -= centered.mean(axis=1, keepdims=True)
        centered -= centered.mean(axis=0, keepdims=True)
        logit_change = float(np.linalg.norm(centered, axis=1).mean() / 2)
        transplant_loss = float(np.mean(kl_rows(oracle, arrays["transplant_logits"])
                                        - kl_rows(oracle, orig)))
        lesion_loss = float(np.mean(kl_rows(oracle, arrays["lesion_logits"])
                                    - kl_rows(oracle, orig)))
        summary = {"state_change": target_change,
                   "non_target_t16_max": float(np.max(np.abs(untouched16))),
                   "non_target_t48_max": float(np.max(np.abs(untouched48))),
                   "centered_logit_change": logit_change,
                   "transplant_kl_increase": transplant_loss,
                   "lesion_kl_increase": lesion_loss}
        if any(not np.isclose(summary[name], record[name], rtol=0, atol=1e-10) for name in summary):
            raise ValueError("operation summary differs from raw arrays")
        summary["pass"] = bool(
            target_change >= op["state_change_l2_per_sqrt_width_min"]
            and summary["non_target_t16_max"] <= op["non_target_max_abs_change_max"]
            and summary["non_target_t48_max"] <= op["non_target_max_abs_change_max"]
            and logit_change >= op["centered_logit_change_l2_per_sqrt_classes_min"]
            and transplant_loss >= op["state_transplant_kl_increase_min"]
            and lesion_loss >= op["readout_lesion_kl_increase_min"])
        summaries[str(block)] = summary
    assert reference_h48 is not None
    redundancy = linear_cka_blocks(reference_h48.reshape(len(oracle), 4, 16),
                                   op["cka_denominator_min"])
    redundancy["pass"] = bool(not redundancy["collapsed"]
                              and all(v >= op["state_variance_mean_min"]
                                      for v in redundancy["variance_by_block"])
                              and redundancy["max_cka"] < op["pairwise_linear_cka_max_exclusive"])
    return {"blocks": summaries, "redundancy": redundancy,
            "zero_leakage": all(
                row["non_target_t16_max"] <= op["non_target_max_abs_change_max"]
                and row["non_target_t48_max"] <= op["non_target_max_abs_change_max"]
                for row in summaries.values())}


def _replay_operation_from_checkpoint(run: Path, row: dict[str, Any], seed: int,
                                      x: np.ndarray, oracle: np.ndarray,
                                      cfg: dict[str, Any]) -> None:
    """Re-execute all four causal interventions with frozen final RS weights."""
    from .phase_reveal_formal import transplant_diagnostic

    configure_cpu()
    if x.shape != (len(oracle), cfg["task"]["ood_length"], 16):
        raise ValueError("operation replay input shape differs")
    model = build_model("RS4", seed, cfg["model"]["graph_seed_base"] + seed)
    cp = row["checkpoints"][-1]
    cp_path = _safe_relative(run, cp["checkpoint_file"])
    if sha256(cp_path.read_bytes()).hexdigest() != cp["checkpoint_sha256"]:
        raise ValueError("RS final checkpoint hash differs before operation replay")
    checkpoint = torch.load(cp_path, map_location="cpu", weights_only=True)
    if checkpoint["arm"] != "RS4" or checkpoint["seed"] != seed or checkpoint["step"] != cfg["training"]["updates_per_arm_seed"]:
        raise ValueError("RS checkpoint metadata differs before operation replay")
    model.load_state_dict(checkpoint["state_dict"])
    if _state_hash(model) != row["final_state_sha256"]:
        raise ValueError("RS final weights differ before operation replay")
    tensor_x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    with torch.no_grad():
        original = model.run_trace(tensor_x, (16, 48))
    fields = ("original_logits", "transplant_logits", "lesion_logits",
              "h16_original", "h16_transplanted", "h48_original",
              "h48_transplanted")
    for block in range(4):
        record = row["operation"][str(block)]
        replayed = transplant_diagnostic(model, tensor_x, oracle, block,
                                         cfg["operation"]["prefix_permutation_seed_base"] + block,
                                         original)
        archived = _archived_npz(run, str(Path(row["operation_dir"]) / record["arrays_file"]),
                                 record["arrays_sha256"], fields)
        for field in fields:
            if not np.array_equal(replayed[field], archived[field]):
                raise ValueError(f"RS block {block} {field} differs from checkpoint replay")
        for name, value in replayed.items():
            if isinstance(value, np.ndarray):
                continue
            if (isinstance(value, (int, float)) and not isinstance(value, bool)):
                if not np.isclose(value, record[name], rtol=0, atol=1e-10):
                    raise ValueError(f"RS block {block} {name} summary differs from replay")
            elif value != record[name]:
                raise ValueError(f"RS block {block} {name} provenance differs from replay")


def _complete_effect(run: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    cfg = manifest["config"]
    formal = cfg["formal"]
    if (manifest.get("config_sha256") != CONFIG_SHA256
            or cfg.get("experiment_id") != "EXP-0014" or cfg.get("version") != "v3"):
        raise ValueError("formal config is not the frozen v3 document")
    preflight = manifest.get("engineering_preflight", {})
    report_file = Path(preflight["report_file"])
    terminal_file = Path(preflight["terminal_file"])
    if (sha256(report_file.read_bytes()).hexdigest() != preflight["report_sha256"]
            or sha256(terminal_file.read_bytes()).hexdigest() != preflight["terminal_sha256"]
            or not manifest.get("qualification_admission")
            or manifest.get("engineering", {}).get("status") != "PASS"):
        raise ValueError("formal engineering provenance is missing or changed")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_file.read_text(encoding="utf-8"))
    if (report.get("engineering") != manifest["engineering"]
            or report.get("qualification_admission") != manifest["qualification_admission"]
            or report.get("config_sha256") != CONFIG_SHA256
            or terminal.get("status") != "PASS"
            or terminal.get("report_sha256") != preflight["report_sha256"]
            or terminal.get("receipt_sha256") != preflight["receipt_sha256"]
            or terminal.get("seconds_after_receipt_fsync", float("inf"))
            > cfg["resource"]["preflight_wall_seconds_cap"]):
        raise ValueError("formal preflight report and manifest disagree")
    if manifest.get("research_updates") != cfg["resource"]["research_update_cap"]:
        raise ValueError("formal research update count incomplete")
    if manifest.get("formal_task_roots_read") != 3 * (len(formal["model_seeds"]) + len(formal["replication_seeds"])):
        raise ValueError("formal sealed-root count differs")
    matrix = Path(manifest["project_root"]) / cfg["matrix_file"]
    if sha256(matrix.read_bytes()).hexdigest() != cfg["matrix_sha256"]:
        raise ValueError("analysis matrix SHA differs")
    values: dict[str, Any] = {"formal": {}, "replication": {}}
    for stage in values:
        seeds = formal["model_seeds" if stage == "formal" else "replication_seeds"]
        base = (("iid", "iid_root_base"), ("ood", "ood_root_base"),
                ("retained", "retained_root_base")) if stage == "formal" else (
                    ("iid", "rep_iid_root_base"), ("ood", "rep_ood_root_base"),
                    ("retained", "rep_retained_root_base"))
        if set(manifest["sealed"][stage]) != {str(s) for s in seeds} or set(manifest["arms"][stage]) != {str(s) for s in seeds}:
            raise ValueError(f"{stage} incomplete seed set")
        for seed in seeds:
            records = manifest["sealed"][stage][str(seed)]
            rows = manifest["arms"][stage][str(seed)]
            if set(rows) != {"RS4", "SC64", "D64_full"} or set(records) != {"iid", "ood", "retained"}:
                raise ValueError("formal arm/split set incomplete")
            raw = {split: _read_sealed(
                run, records[split], formal[key] + seed, formal["eval_examples"],
                cfg["task"]["train_length"] if split == "iid" else cfg["task"]["ood_length"],
                matrix,
                cfg["task"]["retained_switch_step"] if split == "retained" else cfg["task"]["switch_step"],
                cfg["task"]["retained_cue_reliability"] if split == "retained" else cfg["task"]["cue_reliability"])
                for split, key in base}
            checked: dict[str, Any] = {}
            diagnostics: dict[str, dict[str, np.ndarray]] = {}
            for arm, row in rows.items():
                if (row["arm"] != arm or row["seed"] != seed or row["status"] != "COMPLETED"
                        or row["updates"] != cfg["training"]["updates_per_arm_seed"]
                        or [cp["step"] for cp in row["checkpoints"]] != cfg["training"]["monitor_steps"]):
                    raise ValueError("formal training row incomplete")
                admission = manifest["qualification_admission"]
                expected_lr = (admission["lr_dense"] if arm == "D64_full"
                               else admission["lr_rs_sc"])
                if row["lr"] != expected_lr:
                    raise ValueError("formal learning rate differs from qualification")
                for checkpoint in row["checkpoints"]:
                    for name, checksum in (("checkpoint_file", "checkpoint_sha256"),
                                           ("prediction_file", "prediction_sha256")):
                        path = _safe_relative(run, checkpoint[name])
                        if sha256(path.read_bytes()).hexdigest() != checkpoint[checksum]:
                            raise ValueError("checkpoint/monitor hash mismatch")
                final_checkpoint = _safe_relative(run, row["checkpoints"][-1]["checkpoint_file"])
                saved = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
                if (saved.get("arm") != arm or saved.get("seed") != seed
                        or saved.get("step") != cfg["training"]["updates_per_arm_seed"]):
                    raise ValueError("final checkpoint metadata differs")
                state_hash = sha256()
                for name, tensor in saved["state_dict"].items():
                    state_hash.update(name.encode())
                    state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                if state_hash.hexdigest() != row["final_state_sha256"]:
                    raise ValueError("final checkpoint state differs from recorded tensors")
                checked[arm] = {split: _read_prediction(run, row, split, records[split], raw[split])
                                for split in ("iid", "ood", "retained")}
                diag_record = row["state_diagnostic"]
                diagnostic = _archived_npz(run, diag_record["file"], diag_record["sha256"],
                                           ("h16", "h48", "logits16", "logits48"))
                if (diagnostic["h16"].shape != (formal["eval_examples"], 64)
                        or diagnostic["h48"].shape != (formal["eval_examples"], 64)
                        or diagnostic["logits16"].shape != (formal["eval_examples"], 4)
                        or diagnostic["logits48"].shape != (formal["eval_examples"], 4)
                        or not np.allclose(diagnostic["logits48"],
                                           checked[arm]["ood"]["arrays"]["logits_final"],
                                           atol=1e-6, rtol=0)):
                    raise ValueError("OOD hidden-state diagnostic differs")
                diagnostics[arm] = diagnostic
            ood = raw["ood"]
            retained = raw["retained"]
            ood_mask = correction_mask(ood["H"], ood["cue"], ood["oracle_pre"], ood["oracle_final"])
            retained_mask = correction_mask(retained["H"], retained["cue"],
                                            retained["oracle_pre"], retained["oracle_final"])
            operation = _operation_seed(run, rows["RS4"], ood["oracle_final"],
                                        checked["RS4"]["ood"]["arrays"]["logits_final"],
                                        cfg, diagnostics["RS4"])
            _replay_operation_from_checkpoint(run, rows["RS4"], seed,
                                              ood["x"], ood["oracle_final"], cfg)
            sc_partition = linear_cka_blocks(diagnostics["SC64"]["h48"].reshape(formal["eval_examples"], 4, 16),
                                             cfg["operation"]["cka_denominator_min"])
            def diff(control: str, split: str, mask: np.ndarray | None = None) -> float:
                oracle = raw[split]["oracle_final"]
                left = kl_rows(oracle, checked[control][split]["arrays"]["logits_final"])
                right = kl_rows(oracle, checked["RS4"][split]["arrays"]["logits_final"])
                return float((left - right)[mask].mean()) if mask is not None else float((left - right).mean())
            values[stage][str(seed)] = {
                "kl": {arm: {split: checked[arm][split]["final_kl"] for split in checked[arm]}
                       for arm in checked},
                "d": diff("SC64", "ood"), "q": diff("D64_full", "ood"),
                "correction_count": int(ood_mask.sum()),
                "retained_correction_count": int(retained_mask.sum()),
                "correction_d": diff("SC64", "ood", ood_mask) if ood_mask.any() else None,
                "retained_d": diff("SC64", "retained"),
                "retained_q": diff("D64_full", "retained"),
                "operation": operation,
                "sc_fixed_group_partition_diagnostic": sc_partition,
            }
    frows = list(values["formal"].values())
    rrows = list(values["replication"].values())
    indices = np.random.Generator(np.random.PCG64(cfg["effects"]["bootstrap_seed"])).integers(
        0, len(frows), size=(cfg["effects"]["bootstrap_resamples"], len(frows)))
    d = paired_bootstrap([row["d"] for row in frows], cfg["effects"]["bootstrap_seed"],
                         cfg["effects"]["bootstrap_resamples"], indices)
    q = paired_bootstrap([row["q"] for row in frows], cfg["effects"]["bootstrap_seed"],
                         cfg["effects"]["bootstrap_resamples"], indices)
    enough_slice = all(row["correction_count"] >= formal["correction_slice_min_per_seed"]
                       and row["retained_correction_count"] >= formal["correction_slice_min_per_seed"]
                       for row in frows)
    correction = (paired_bootstrap([row["correction_d"] for row in frows],
                                   cfg["effects"]["bootstrap_seed"],
                                   cfg["effects"]["bootstrap_resamples"], indices)
                  if enough_slice else None)
    admission = manifest["qualification_admission"]
    control_windows = admission["ood_mean_windows"]
    seed_windows = admission["ood_per_seed_windows"]
    learned = all(row["kl"][arm]["iid"] < 0.15
                  for row in frows for arm in cfg["arms"])
    headroom = all(
        control_windows[arm][0] <= np.mean([row["kl"][arm]["ood"] for row in frows]) <= control_windows[arm][1]
        and sum(seed_windows[arm][0] <= row["kl"][arm]["ood"] <= seed_windows[arm][1]
                for row in frows) >= 8
        for arm in ("SC64", "D64_full"))
    retained_positive = all(np.mean([row[key] for row in frows]) > 0
                            for key in ("retained_d", "retained_q"))
    replication_positive = all(sum(row[key] > 0 for row in rrows) >= cfg["effects"]["replication_positive_min_per_comparison"]
                               and np.mean([row[key] for row in rrows]) > 0
                               for key in ("d", "q"))
    operation = _operation_verdict(values, cfg)
    if not learned or not headroom:
        verdict = {"decision": "INCONCLUSIVE", "substatus": "LEARNING_OR_HEADROOM_FAILURE"}
    elif not enough_slice:
        verdict = {"decision": "INCONCLUSIVE", "substatus": "SLICE_UNINFORMATIVE"}
    else:
        verdict = classify_complete(tuple(d["ci95"]), tuple(q["ci95"]),
                                    tuple(correction["ci95"]), operation=operation["status"],
                                    retained_positive=retained_positive,
                                    replication_positive=replication_positive,
                                    learned_and_headroom=learned and headroom,
                                    raw_complete=True)
    return {**verdict, "experiment_id": "EXP-0014", "version": "v3",
            "effect_computed": True, "seed_metrics": values,
            "primary_d": d, "dense_q": q, "correction_d": correction,
            "correction_slice_informative": enough_slice,
            "learning_pass": learned, "headroom_pass": headroom,
            "retained_positive": retained_positive,
            "replication_positive": replication_positive,
            "operation": operation, "gate1_credit_before_independent_review": False}


def _operation_verdict(values: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    op = cfg["operation"]
    formal = list(values["formal"].values())
    replication = list(values["replication"].values())
    seed_pass_counts: dict[str, int] = {}
    intervals: dict[str, Any] = {}
    indices = np.random.Generator(np.random.PCG64(op["bootstrap_seed"])).integers(
        0, len(formal), size=(op["bootstrap_resamples"], len(formal)))
    all_ok = True
    for block in range(4):
        b = str(block)
        for stage, rows, minimum in (("formal", formal, op["formal_seed_pass_min"]),
                                     ("replication", replication, op["replication_seed_pass_min"])):
            guards = {
                "state_change": lambda v: v >= op["state_change_l2_per_sqrt_width_min"],
                "centered_logit_change": lambda v: v >= op["centered_logit_change_l2_per_sqrt_classes_min"],
                "transplant_kl_increase": lambda v: v >= op["state_transplant_kl_increase_min"],
                "lesion_kl_increase": lambda v: v >= op["readout_lesion_kl_increase_min"],
            }
            for key, passes in guards.items():
                count = sum(bool(passes(row["operation"]["blocks"][b][key])) for row in rows)
                seed_pass_counts[f"{stage}_block_{b}_{key}"] = count
                all_ok &= count >= minimum
        for effect in ("transplant_kl_increase", "lesion_kl_increase"):
            name = f"block_{b}_{effect}"
            stats = paired_bootstrap([row["operation"]["blocks"][b][effect]
                                      for row in formal], op["bootstrap_seed"],
                                     op["bootstrap_resamples"], indices)
            intervals[name] = stats
            all_ok &= stats["ci95"][0] > 0
    for stage, rows, minimum in (("formal", formal, op["formal_seed_pass_min"]),
                                 ("replication", replication, op["replication_seed_pass_min"])):
        for block in range(4):
            count = sum(row["operation"]["redundancy"]["variance_by_block"][block]
                        >= op["state_variance_mean_min"] for row in rows)
            seed_pass_counts[f"{stage}_block_{block}_variance"] = count
            all_ok &= count >= minimum
        count = sum(row["operation"]["redundancy"]["pass"] for row in rows)
        seed_pass_counts[f"{stage}_noncollapse_cka"] = count
        all_ok &= count >= minimum
    all_ok &= all(row["operation"]["zero_leakage"] for row in formal + replication)
    return {"status": "PASS" if all_ok else "FAIL", "seed_pass_counts": seed_pass_counts,
            "intervals": intervals,
            "zero_leakage_all_seeds": all(row["operation"]["zero_leakage"]
                                           for row in formal + replication)}


def analyze(run: Path) -> dict[str, Any]:
    """Write a conservative terminal report for stopped or incomplete runs.

    Complete effect analysis requires all 15 paired-seed raw archives and is
    deliberately unavailable until that archive contract is implemented.
    No partial set is promoted to a scientific effect claim.
    """
    run = Path(run)
    worker_failure = run / "worker_failure.json"
    manifest_path = run / "manifest.json"
    if worker_failure.is_file():
        failure = json.loads(worker_failure.read_text(encoding="utf-8"))
        substatus = failure.get("substatus", "ENGINEERING_ABORT")
        if substatus not in {"INCOMPLETE_BUDGET", "ENGINEERING_ABORT"}:
            substatus = "ENGINEERING_ABORT"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        result: dict[str, Any] = {
            "experiment_id": "EXP-0014", "version": "v3", "decision": "INCONCLUSIVE",
            "substatus": substatus, "research_updates": manifest.get("research_updates", 0),
            "source": "worker_failure", "effect_computed": False,
        }
        if manifest_path.is_file():
            result["manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    elif manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = manifest.get("status", "ENGINEERING_ABORT")
        if status == "COMPLETED":
            try:
                result = _complete_effect(run, manifest)
            except (OSError, KeyError, ValueError, FloatingPointError, AssertionError) as exc:
                result = {"experiment_id": "EXP-0014", "version": "v3",
                          "decision": "INCONCLUSIVE", "substatus": "MISSING_OR_INVALID_RAW_ARRAYS",
                          "evidence_error": str(exc), "effect_computed": False}
        else:
            result = {"experiment_id": "EXP-0014", "version": "v3",
                      "decision": "INCONCLUSIVE", "substatus": status,
                      "research_updates": manifest.get("research_updates", 0),
                      "source": "manifest", "effect_computed": False}
        result["manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    else:
        raise FileNotFoundError("formal manifest and worker failure record both missing")
    path = run / "analysis.json"
    if path.exists():
        raise FileExistsError(f"formal analysis already exists: {path}")
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
