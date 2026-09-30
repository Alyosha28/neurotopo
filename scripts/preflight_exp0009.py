"""No-outcome EXP-0009 model, graph, stream and resource acceptance checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from neurotopo.resources import resource_account
from neurotopo.run import configure, digest, source_fingerprints, write_json
from neurotopo.source_belief_study import (batch_sha256, build_arm, cross_block_edges,
                                           paired_evaluation_batches, parameter_sha256,
                                           schedule, update_batch_hash)
from neurotopo.source_belief_task import make_source_belief_batch


def preflight(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text())
    if (config["experiment_id"], config["version"], config["workflow"],
            config["models"], config["seeds"], config["updates"],
            config["batch_size"], config["eval_distractors_per_evidence"]) != (
            "EXP-0009", "v1", "source_belief", ["multi_gru", "random_gru", "gru_wide"],
            list(range(9401, 9411)), 1500, 128, [1, 4, 8]):
        raise ValueError("Config does not match registered EXP-0009 v1 values")
    suite_path = Path("artifacts/validation/suite.json")
    suite = json.loads(suite_path.read_text())
    source_hashes = source_fingerprints()
    problems = []
    if not suite["pass"] or suite["source_sha256"] != source_hashes:
        problems.append("source-hashed engineering validation is missing or stale")
    delays = schedule(config["schedule_seed"], config["updates"])
    schedule_sha = digest(np.asarray(delays, dtype=np.int8).tobytes())
    if len(delays) != 1500 or any(delays[start:start + 300].count(d) != 100
                                 for start in range(0, 1500, 300) for d in (0, 1, 2)):
        problems.append("varied schedule is not balanced within each 300-update block")
    configure(config["seeds"][0], config["threads"])
    monitor = make_source_belief_batch(config["monitor_examples"],
                                       np.random.default_rng(config["monitor_seed"]))
    monitor_hash = batch_sha256(monitor)
    final = paired_evaluation_batches(config)
    final_hashes = {f"d{d}": batch_sha256(batch) for d, batch in final.items()}
    rows = []
    for seed in config["seeds"]:
        models = {name: build_arm(name, config, seed) for name in config["models"]}
        initial = {name: parameter_sha256(model) for name, model in models.items()}
        if initial["multi_gru"] != initial["random_gru"]:
            problems.append(f"seed{seed}: primary initial weights differ")
        model_data = {}
        for name, model in models.items():
            graph = model.graph_info
            crossed = cross_block_edges(model)
            if name in ("multi_gru", "random_gru") and (
                    graph["diagonal_connections"] != 64 or
                    any(value != 16 for value in graph["in_degree"] + graph["out_degree"]) or
                    (name == "random_gru" and not graph["strongly_connected"])):
                problems.append(f"seed{seed}/{name}: graph degree/self-loop/connectedness mismatch")
            if (name == "multi_gru" and crossed != 0) or (name == "random_gru" and crossed <= 0):
                problems.append(f"seed{seed}/{name}: cross-block edge descriptor mismatch")
            resources = {f"d{d}": resource_account(model, final[d].x.shape[1], config["batch_size"])
                         for d in config["eval_distractors_per_evidence"]}
            if resources["d4"]["allocated_parameters"] != 14920 or (
                    name in ("multi_gru", "random_gru") and resources["d4"]["effective_parameters"] != 5704):
                problems.append(f"seed{seed}/{name}: parameter count mismatch")
            smoke = final[4].x[:16]
            times = []
            with torch.no_grad():
                for _ in range(2):
                    model(smoke, final[4].pre_index)
                for _ in range(5):
                    start = time.perf_counter()
                    model(smoke, final[4].pre_index)
                    times.append(time.perf_counter() - start)
            model_data[name] = {
                "initial_parameters_sha256": initial[name],
                "mask_sha256": digest(model.mask.numpy().tobytes()),
                "graph": graph, "cross_original_block_edges": crossed,
                "resources": resources,
                "median_smoke_forward_seconds_batch16_d4": statistics.median(times),
            }
        for d in config["eval_distractors_per_evidence"]:
            key = f"d{d}"
            a, b, wide = (model_data[name]["resources"][key] for name in config["models"])
            for field in ("allocated_parameters", "effective_parameters", "active_recurrent_parameters",
                          "hidden_state_floats", "dense_forward_linear_macs_per_example",
                          "active_forward_linear_macs_per_example", "dense_forward_linear_flops_per_batch"):
                if a[field] != b[field]:
                    problems.append(f"seed{seed}/{key}: primary resource {field} mismatch")
            if (a["allocated_parameters"] != wide["allocated_parameters"] or
                    a["hidden_state_floats"] != wide["hidden_state_floats"] or
                    a["dense_forward_linear_flops_per_batch"] != wide["dense_forward_linear_flops_per_batch"]):
                problems.append(f"seed{seed}/{key}: contextual dense allocated/state/execution mismatch")
        training_hash = hashlib.sha256()
        tokens = 0
        flops = 0
        for step, d in enumerate(delays, 1):
            batch = make_source_belief_batch(config["batch_size"],
                                             np.random.default_rng(config["train_seed_base"] + seed * 10_000 + step),
                                             distractors_per_evidence=d)
            update_batch_hash(training_hash, batch)
            tokens += len(batch.x) * batch.x.shape[1]
            flops += resource_account(models["multi_gru"], batch.x.shape[1], len(batch.x))[
                "dense_forward_linear_flops_per_batch"]
        if tokens != 3_456_000:
            problems.append(f"seed{seed}: expected train token count changed")
        rows.append({"seed": seed, "models": model_data,
                     "training_dataset_sha256": training_hash.hexdigest(),
                     "training_tokens": tokens, "training_forward_linear_flops": flops})
    common_tokens = {row["training_tokens"] for row in rows}
    common_flops = {row["training_forward_linear_flops"] for row in rows}
    if len(common_tokens) != 1 or len(common_flops) != 1:
        problems.append("training compute proxy differs across planned seeds")
    report = {
        "pass": not problems, "problems": problems,
        "experiment_id": "EXP-0009", "version": "v1",
        "source_sha256": source_hashes,
        "script_sha256": digest(Path(__file__).read_bytes()),
        "config_sha256": digest(config_path.read_bytes()),
        "validation_sha256": digest(suite_path.read_bytes()),
        "schedule_sha256": schedule_sha,
        "monitor_dataset_sha256": monitor_hash,
        "final_dataset_sha256": final_hashes,
        "training_tokens_per_arm": rows[0]["training_tokens"],
        "training_forward_linear_flops_per_arm": rows[0]["training_forward_linear_flops"],
        "seeds": rows,
        "compute_scope": "Forward linear analytical proxy; complete training FLOPs and peak memory not measured.",
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp0009_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation/exp0009_preflight.json"))
    args = parser.parse_args()
    result = preflight(args.config, args.output)
    print(json.dumps({"output": str(args.output), "pass": result["pass"],
                      "problems": result["problems"], "seeds": len(result["seeds"])}), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
