"""EXP0033 v1: DELAY-2 carrier confirmation (control-only) and the exploratory SC-REL-v4 vs matched-control probe."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .arch0001_control_sweep import training_cfg
from .arch0001_core import op_estimate, param_count
from .arch0001_core_v4 import candidate_metrics, evaluate_v4, fit_v4
from .arch0001_memory_requalification import calibration_stream, meaningful_metrics, oracle_score, reference_scores
from .arch0001_memory_task import evaluate_memory, evaluation_stream, fit_memory, generate_memory
from .frozen_binary_probe import save_json
from .order_observation import sha
from .run import git_output, source_fingerprints


def _seed(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return data


def do_carrier_run(cfg: dict[str, Any], width: int, blocks: int, seed: int, out: Path) -> dict[str, Any]:
    path = out / "carrier" / f"w{width}_b{blocks}_seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    c = training_cfg(cfg, cfg["task"]["budget"], cfg["task"]["curriculum"], width=width, blocks=blocks)
    result = fit_memory(seed, cfg["task"]["d"], "PLAIN", c)
    data = evaluation_stream(c, seed, cfg["task"]["d"], cfg["task"]["primary_length"])
    record = dict(width=width, blocks=blocks, seed=seed, d=cfg["task"]["d"], budget=cfg["task"]["budget"],
                  primary=meaningful_metrics(evaluate_memory(result["model"], data), data, cfg["task"]["d"]),
                  references=reference_scores(data, cfg["task"]["d"], calibration_stream(c, cfg["task"]["d"])),
                  oracle=oracle_score(data, cfg["task"]["d"]),
                  parameters=param_count("INV_COMP", width, blocks),
                  inference_ops=op_estimate("INV_COMP", width, blocks)["inference"],
                  final_hash=result["record"]["final_hash"],
                  training_data_sha256=result["record"]["training_data_sha256"],
                  seconds=result["record"]["seconds"])
    save_json(path, record)
    return record


def run_carrier(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    records = []
    for width, blocks in cfg["carrier_menu"]:
        for seed in cfg["carrier_seeds"]:
            records.append(do_carrier_run(cfg, width, blocks, seed, out))
            print(json.dumps(dict(phase="carrier", w=width, b=blocks, seed=seed,
                                  meaningful=round(records[-1]["primary"]["meaningful_accuracy"], 4)), ensure_ascii=False), flush=True)
    band = cfg["band"]
    by_config: dict[str, Any] = {}
    for width, blocks in cfg["carrier_menu"]:
        rows = [r for r in records if r["width"] == width and r["blocks"] == blocks]
        acc = [r["primary"]["meaningful_accuracy"] for r in rows]
        band_ok = (float(np.mean(acc)) >= band["learn_mean_min"] and float(np.min(acc)) >= band["learn_seed_min"]
                   and float(np.mean(acc)) <= band["nonceiling_mean_max"] and float(np.max(acc)) <= band["nonceiling_seed_max"])
        refs = max(max(r["references"].values()) for r in rows)
        by_config[f"w{width}b{blocks}"] = dict(width=width, blocks=blocks, accuracy=acc, mean=float(np.mean(acc)),
                                               min=float(np.min(acc)), max=float(np.max(acc)),
                                               references_max=refs, parameters=rows[0]["parameters"],
                                               inference_ops=rows[0]["inference_ops"],
                                               qualified=bool(band_ok and refs <= band["memoryless_max"]))
    eligible = [v for v in by_config.values() if v["qualified"]]
    selected = None
    if eligible:
        selected = max(eligible, key=lambda v: (v["mean"], -abs(np.log(v["inference_ops"] / cfg["candidate_inference_ops"]))))
        selected = dict(width=selected["width"], blocks=selected["blocks"], mean=selected["mean"],
                        inference_ops=selected["inference_ops"])
    summary = dict(by_config=by_config, selected=selected, band=band, seeds=cfg["carrier_seeds"])
    save_json(out / "carrier_summary.json", summary)
    return summary


def run_comparison(cfg: dict[str, Any], out: Path, carrier: dict[str, Any]) -> dict[str, Any]:
    d = cfg["task"]["d"]
    length = cfg["task"]["primary_length"]
    conf = training_cfg(cfg, cfg["task"]["budget"], cfg["task"]["curriculum"],
                        width=carrier["width"], blocks=carrier["blocks"])
    arms = {a: [] for a in cfg["arms"]}
    for seed in cfg["comparison_seeds"]:
        data = evaluation_stream(conf, seed, d, length)
        # SEL_INV
        inv = fit_memory(seed, d, "PLAIN", conf)
        inv_metrics = meaningful_metrics(evaluate_memory(inv["model"], data), data, d)
        inv_correct = None
        # per-example correctness for the bootstrap
        parts = evaluate_memory(inv["model"], data)
        target = data["targets"] >= .5
        sel = slice(d + 1, length + 1)
        inv_correct = ((parts["logits"].numpy() >= 0) == target)[:, sel].mean(axis=1)
        # CAND_V4
        cand = fit_v4(seed, d, conf, generate=generate_memory)
        cand_parts = evaluate_v4(cand["model"], data)
        cand_metrics = candidate_metrics(cand_parts, data, d, lambda dd, ll: slice(dd + 1, ll + 1))
        cand_correct = ((cand_parts["logits"].numpy() >= 0) == target)[:, sel].mean(axis=1)
        # EXTRA_V4 (same supervision, no correction)
        extra = fit_v4(seed, d, conf, generate=generate_memory, correct=False)
        extra_parts = evaluate_v4(extra["model"], data)
        extra_metrics = candidate_metrics(extra_parts, data, d, lambda dd, ll: slice(dd + 1, ll + 1))
        extra_correct = ((extra_parts["logits"].numpy() >= 0) == target)[:, sel].mean(axis=1)
        record = dict(seed=seed, d=d, budget=cfg["task"]["budget"], carrier=carrier,
                      arms=dict(CAND_V4=dict(metrics=cand_metrics, record=cand["record"], correct=cand_correct.tolist()),
                                SEL_INV=dict(metrics=inv_metrics, record=inv["record"], correct=inv_correct.tolist(),
                                             parameters=param_count("INV_COMP", carrier["width"], carrier["blocks"]),
                                             inference_ops=op_estimate("INV_COMP", carrier["width"], carrier["blocks"])["inference"]),
                                EXTRA_V4=dict(metrics=extra_metrics, record=extra["record"], correct=extra_correct.tolist(),
                                              inference_ops=op_estimate("EXTRA_SUP")["inference"])))
        save_json(out / "comparison" / f"seed{seed}.json", record)
        arms["CAND_V4"].append(record)
        arms["SEL_INV"].append(record)
        arms["EXTRA_V4"].append(record)
        print(json.dumps(dict(phase="comparison", seed=seed,
                              cand=round(cand_metrics["meaningful_accuracy"], 4),
                              sel=round(inv_metrics["meaningful_accuracy"], 4),
                              extra=round(extra_metrics["meaningful_accuracy"], 4)), ensure_ascii=False), flush=True)
    cand_acc = np.array([r["arms"]["CAND_V4"]["metrics"]["meaningful_accuracy"] for r in arms["CAND_V4"]])
    sel_acc = np.array([r["arms"]["SEL_INV"]["metrics"]["meaningful_accuracy"] for r in arms["SEL_INV"]])
    extra_acc = np.array([r["arms"]["EXTRA_V4"]["metrics"]["meaningful_accuracy"] for r in arms["EXTRA_V4"]])
    rng = np.random.default_rng(20260929)
    draws = []
    n_seed = len(cand_acc)
    for _ in range(cfg["bootstrap"]):
        picks = rng.integers(0, n_seed, size=n_seed)
        diffs = []
        for p in picks:
            cand_correct = np.array(arms["CAND_V4"][p]["arms"]["CAND_V4"]["correct"])
            sel_correct = np.array(arms["SEL_INV"][p]["arms"]["SEL_INV"]["correct"])
            idx = rng.integers(0, len(cand_correct), size=len(cand_correct))
            diffs.append(float((cand_correct[idx] - sel_correct[idx]).mean()))
        draws.append(float(np.mean(diffs)))
    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    effect = float(np.mean(cand_acc - sel_acc))
    band = cfg["band"]
    control_in_band = bool(float(sel_acc.mean()) >= band["learn_mean_min"] and float(sel_acc.min()) >= band["learn_seed_min"]
                           and float(sel_acc.mean()) <= band["nonceiling_mean_max"] and float(sel_acc.max()) <= band["nonceiling_seed_max"])
    if effect >= cfg["delta_pp"] / 100. and ci[0] > 0 and control_in_band:
        decision, substatus = "SUPPORTED", "EXPLORATORY_ARCHITECTURE_ADVANTAGE"
    elif effect <= -cfg["delta_pp"] / 100.:
        decision, substatus = "FALSIFIED", "CONTROL_BETTER"
    else:
        decision, substatus = "FALSIFIED", "NO_EXPLORATORY_ADVANTAGE"
    summary = dict(carrier=carrier, candidate_contract="arch0001_successor_compatible_core_v4",
                   seeds=cfg["comparison_seeds"], budget=cfg["task"]["budget"], task=f"DELAY-{d}",
                   accuracy=dict(CAND_V4=cand_acc.tolist(), SEL_INV=sel_acc.tolist(), EXTRA_V4=extra_acc.tolist()),
                   means=dict(CAND_V4=float(cand_acc.mean()), SEL_INV=float(sel_acc.mean()), EXTRA_V4=float(extra_acc.mean())),
                   effect_pp=effect * 100., bootstrap_ci95_pp=[ci[0] * 100., ci[1] * 100.],
                   control_in_band=control_in_band, delta_pp=cfg["delta_pp"],
                   resources=dict(candidate_parameters=int(np.mean([r["arms"]["CAND_V4"]["record"]["parameters"] for r in arms["CAND_V4"]])),
                                  candidate_inference_ops=cfg["candidate_inference_ops"],
                                  control_parameters=carrier.get("parameters"), control_inference_ops=carrier["inference_ops"],
                                  extra_inference_ops=op_estimate("EXTRA_SUP")["inference"]),
                   decision=decision, substatus=substatus)
    save_json(out / "comparison_summary.json", summary)
    return summary


def validate_config(c: dict[str, Any]) -> None:
    assert c["experiment_id"] in ("EXP-0033", "EXP-0034") and c["version"] == "v1"
    if c["experiment_id"] == "EXP-0033":
        assert c["carrier_menu"] == [[32, 3], [48, 2], [48, 3], [64, 2], [96, 2], [96, 3]]
        assert c["carrier_seeds"] == [29301, 29302, 29303]
        assert c["comparison_seeds"] == [29311, 29312, 29313, 29314, 29315]
    else:
        assert c["carrier_override"]["width"] == 48 and c["carrier_override"]["blocks"] == 2
        assert abs(c["carrier_override"]["ops_ratio_vs_candidate"] - 1) <= 0.10
        assert c["comparison_seeds"] == [29321, 29322, 29323, 29324, 29325]
    assert c["arms"] == ["CAND_V4", "SEL_INV", "EXTRA_V4"]
    assert c["task"] == dict(d=2, primary_length=12, budget=2000,
                             curriculum=[[500, 2], [1000, 4], [1500, 8], [2000, 12]])
    assert c["delta_pp"] == 3.0 and c["bootstrap"] == 4000 and c["candidate_inference_ops"] == 1161
    assert c["band"] == dict(learn_mean_min=.65, learn_seed_min=.58, memoryless_margin_mean=.05,
                             memoryless_margin_seed=.02, nonceiling_mean_max=.95, nonceiling_seed_max=.97,
                             oracle_min=.99, memoryless_max=.55)


def manifest(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    status = git_output("status", "--porcelain")
    frozen = [config_path, Path(cfg["protocol"]), Path(cfg["candidate_contract"]),
              Path("src/neurotopo/arch0001_a2_probe.py"), Path("src/neurotopo/arch0001_core_v4.py"),
              Path("configs/exp0032_v2.json"), Path("uv.lock"), Path("artifacts/validation/suite.json"),
              Path("tests/test_arch0001_a2_probe.py")]
    return dict(experiment_id=cfg["experiment_id"], version=cfg["version"], gate=cfg["gate"], config=cfg,
                git_commit=git_output("rev-parse", "HEAD"), git_dirty=bool(status.strip()),
                git_status=status, source_sha256=source_fingerprints(),
                frozen_hashes={str(p): sha(p) for p in frozen},
                python=sys.version, numpy=np.__version__, platform=platform.platform(),
                scope="Exploratory probe: carrier confirmation plus candidate-vs-control comparison; no A2 gate credit.")


def run(config_path: Path, output: Path, phases: str = "all") -> dict[str, Any]:
    cfg = json.loads(config_path.read_text())
    validate_config(cfg)
    payload = json.loads(Path("artifacts/validation/suite.json").read_text())
    if not payload["pass"] or payload["source_sha256"] != source_fingerprints():
        raise RuntimeError("Validation suite does not match current sources; run scripts/verify.py")
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "manifest.json").exists():
        save_json(output / "manifest.json", manifest(cfg, config_path))
    started = time.monotonic()
    todo = ["carrier", "compare", "analyze"] if phases == "all" else phases.split(",")
    carrier = json.loads((output / "carrier_summary.json").read_text()) if (output / "carrier_summary.json").exists() else None
    comparison = json.loads((output / "comparison_summary.json").read_text()) if (output / "comparison_summary.json").exists() else None
    for phase in todo:
        if time.monotonic() - started > cfg["resource"]["wall_seconds_cap"]:
            raise TimeoutError("Shared EXP-0033 wall budget")
        if phase == "carrier":
            if cfg.get("carrier_override"):
                carrier = dict(by_config={}, selected=dict(cfg["carrier_override"]),
                               band=cfg["band"], seeds=[], source="frozen carrier override from the replication protocol")
                save_json(output / "carrier_summary.json", carrier)
            else:
                carrier = run_carrier(cfg, output)
        elif phase == "compare":
            carrier = carrier or json.loads((output / "carrier_summary.json").read_text())
            if carrier["selected"] is not None:
                comparison = run_comparison(cfg, output, carrier["selected"])
        elif phase == "analyze":
            carrier = carrier or json.loads((output / "carrier_summary.json").read_text())
            if comparison is None and (output / "comparison_summary.json").exists():
                comparison = json.loads((output / "comparison_summary.json").read_text())
            if carrier["selected"] is None:
                decision, substatus = "INCONCLUSIVE", "NO_CARRIER"
            else:
                decision, substatus = comparison["decision"], comparison["substatus"]
            analysis = dict(experiment_id=cfg["experiment_id"], version=cfg["version"], gate=cfg["gate"],
                            decision=decision, substatus=substatus, carrier=carrier, comparison=comparison,
                            seconds=time.monotonic() - started,
                            scope=("Exploratory probe only: even SUPPORTED here is not an A2 gate pass and needs "
                                   "independent replication; no A1 or old Gate 1 credit."))
            save_json(output / "analysis.json", analysis)
            return analysis
        else:
            raise ValueError(phase)
    return dict(phases=todo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phases", default="all")
    args = parser.parse_args()
    outcome = run(args.config, args.output, args.phases)
    print(json.dumps({k: v for k, v in outcome.items() if k not in ("carrier", "comparison")}, ensure_ascii=False, default=str))
