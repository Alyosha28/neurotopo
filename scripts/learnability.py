"""Registered baseline diagnostic, isolated from architectural hypothesis tests."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from neurotopo.models import build_model
from neurotopo.resources import resource_account
from neurotopo.run import configure, digest, evaluate, git_output, loss_on, source_fingerprints, write_json
from neurotopo.task import make_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    base = json.loads(Path(c["base_config"]).read_text())
    if git_output("status", "--porcelain"):
        raise RuntimeError("Commit the registered diagnostic before running")
    validation = json.loads(Path("artifacts/validation/suite.json").read_text())
    if not validation["pass"] or validation["source_sha256"] != source_fingerprints():
        raise RuntimeError("Current source validation required")
    output = Path(c["output"])
    if output.exists():
        raise FileExistsError(output)
    manifest = {"status": "RUNNING", "purpose": "BASELINE_DIAGNOSTIC_NOT_ARCHITECTURE_COMPARISON",
                "config": c, "base_config": base, "git_commit": git_output("rev-parse", "HEAD"),
                "source_sha256": source_fingerprints(), "config_sha256": digest(args.config.read_bytes()),
                "script_sha256": digest(Path(__file__).read_bytes()),
                "lock_sha256": digest(Path("uv.lock").read_bytes()), "validation_pass": True,
                "platform": platform.platform(), "machine": platform.machine(),
                "python": sys.version, "torch": torch.__version__, "numpy": np.__version__}
    write_json(output / "manifest.json", manifest)
    rows = []
    try:
        for delay in c["delays"]:
            for seed in c["seeds"]:
                configure(seed, base["threads"])
                model = build_model(c["model"], base, seed)
                optimizer = torch.optim.Adam(model.parameters(), lr=base["learning_rate"])
                rng = np.random.default_rng(c["train_seed_offset"] + seed)
                evaluation = make_batch(c["evaluation_pairs"], delay, np.random.default_rng(c["evaluation_seed_offset"] + delay))
                checkpoints, losses = {}, []
                started = time.perf_counter()
                for step in range(1, c["updates"] + 1):
                    model.train()
                    batch = make_batch(base["batch_size"] // 2, delay, rng)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_on(model, batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"nonfinite loss: {delay}/{seed}/{step}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), base["grad_clip"], error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(loss.item())
                    if step in c["checkpoints"]:
                        checkpoints[str(step)], _ = evaluate(model, evaluation)
                        print(json.dumps({"seed": seed, "delay": delay, "update": step,
                                          "pre": checkpoints[str(step)]["pre_accuracy"],
                                          "joint": checkpoints[str(step)]["joint_pair_accuracy"]}), flush=True)
                row = {"seed": seed, "delay": delay, "checkpoints": checkpoints, "losses": losses,
                       "runtime_seconds_including_eval": time.perf_counter() - started,
                       "resources": resource_account(model, evaluation.x.shape[1]),
                       "train_examples": c["updates"] * base["batch_size"],
                       "train_tokens": c["updates"] * base["batch_size"] * evaluation.x.shape[1],
                       "evaluation_sha256": digest(evaluation.x.numpy().tobytes() + evaluation.pre_target.numpy().tobytes() + evaluation.final_target.numpy().tobytes())}
                write_json(output / f"delay-{delay}-seed-{seed}.json", row)
                rows.append(row)
        def passes(delay: int) -> bool:
            return all(row["checkpoints"][str(c["updates"])]["pre_accuracy"] >= c["minimum_pre_accuracy"] and
                       row["checkpoints"][str(c["updates"])]["joint_pair_accuracy"] >= c["minimum_joint_accuracy"]
                       for row in rows if row["delay"] == delay)
        decision = "SUPPORTED" if passes(8) else ("PARTIALLY_SUPPORTED" if passes(0) else "INCONCLUSIVE")
        summary = {"decision": decision, "scope": "H64 dense GRU feasibility at fixed 6000-update budget only",
                   "gate1_advanced": False,
                   "final_metrics": [{"seed": row["seed"], "delay": row["delay"], **row["checkpoints"][str(c["updates"]) ]} for row in rows]}
        write_json(output / "analysis.json", summary)
        manifest["status"] = "COMPLETED"
    except BaseException as exc:
        manifest.update(status="FAILED", error=repr(exc))
        raise
    finally:
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
