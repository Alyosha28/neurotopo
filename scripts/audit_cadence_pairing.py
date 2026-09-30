"""Independently regenerate paired training inputs and block compute for CAL-0004."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from neurotopo.models import RecurrentModel
from neurotopo.resources import resource_account
from neurotopo.run import digest, source_fingerprints, write_json
from neurotopo.source_belief_task import (HYPOTHESES, INPUT_SIZE, REGIMES,
                                          make_source_belief_batch)


FIELDS = ("truth_h", "truth_r", "evidence_cues", "evidence_sources", "pre_joint", "final_joint")


def run(output: Path, result_path: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    original = json.loads(result_path.read_text())
    schedule_rng = np.random.default_rng(490001)
    schedule: list[int] = []
    for _ in range(5):
        block = np.repeat(np.arange(3), 100)
        schedule_rng.shuffle(block)
        schedule.extend(int(value) for value in block)
    schedule_hash = digest(np.asarray(schedule, dtype=np.int8).tobytes())
    if schedule_hash != original["schedule_sha256"]:
        raise ValueError("Schedule digest differs from frozen result")
    torch.manual_seed(0)
    model = RecurrentModel(INPUT_SIZE, 64, HYPOTHESES * REGIMES, groups=1, topology="dense")
    rows = []
    for seed in (401, 402, 403):
        content_hashes = {"fixed": hashlib.sha256(), "varied": hashlib.sha256()}
        blocks = []
        for first in range(1, 1501, 300):
            counts = {str(d): 0 for d in (0, 1, 2)}
            tokens = {"fixed": 0, "varied": 0}
            flops = {"fixed": 0, "varied": 0}
            for step in range(first, first + 300):
                d = schedule[step - 1]
                counts[str(d)] += 1
                data_seed = 491000 + seed * 10_000 + step
                batches = {
                    "fixed": make_source_belief_batch(128, np.random.default_rng(data_seed)),
                    "varied": make_source_belief_batch(128, np.random.default_rng(data_seed),
                                                       distractors_per_evidence=d),
                }
                for field in FIELDS:
                    if not torch.equal(getattr(batches["fixed"], field), getattr(batches["varied"], field)):
                        raise ValueError(f"Mismatch seed={seed} step={step} field={field}")
                for arm, batch in batches.items():
                    for field in FIELDS:
                        content_hashes[arm].update(getattr(batch, field).numpy().tobytes())
                    tokens[arm] += len(batch.x) * batch.x.shape[1]
                    flops[arm] += resource_account(model, batch.x.shape[1], len(batch.x))[
                        "dense_forward_linear_flops_per_batch"]
            if counts != {"0": 100, "1": 100, "2": 100} or tokens["fixed"] != tokens["varied"] or flops["fixed"] != flops["varied"]:
                raise ValueError(f"Block mismatch seed={seed} start={first}")
            blocks.append({"updates": [first, first + 299], "d_counts": counts,
                           "tokens_per_arm": tokens["fixed"], "forward_linear_flops_per_arm": flops["fixed"]})
        if content_hashes["fixed"].hexdigest() != content_hashes["varied"].hexdigest():
            raise ValueError(f"Full effective-content digest mismatch seed={seed}")
        original_row = next(row for row in original["model_results"] if row["seed"] == seed)
        for arm in ("fixed", "varied"):
            if (sum(block["tokens_per_arm"] for block in blocks) != original_row["arms"][arm]["train_tokens"] or
                    sum(block["forward_linear_flops_per_arm"] for block in blocks) != original_row["arms"][arm]["train_forward_linear_flops"]):
                raise ValueError(f"Audit differs from frozen resource result seed={seed} arm={arm}")
        rows.append({"seed": seed, "effective_input_target_sha256": content_hashes["fixed"].hexdigest(),
                     "blocks": blocks})
    report = {"kind": "CAL-0004_INDEPENDENT_DETERMINISTIC_PAIR_AUDIT",
              "source_result_sha256": digest(result_path.read_bytes()),
              "source_sha256": source_fingerprints(),
              "script_sha256": digest(Path(__file__).read_bytes()),
              "schedule_sha256": schedule_hash,
              "rows": rows, "all_pairing_and_block_compute_pass": True}
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=Path("artifacts/calibration/gate1_cadence_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/calibration/gate1_cadence_pair_audit.json"))
    args = parser.parse_args()
    result = run(args.output, args.result)
    print(json.dumps({"output": str(args.output), "pass": result["all_pairing_and_block_compute_pass"],
                      "seeds": [row["seed"] for row in result["rows"]]}), flush=True)


if __name__ == "__main__":
    main()
