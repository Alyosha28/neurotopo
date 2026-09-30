"""Independent-product posterior checks and current forward-state resource audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from neurotopo.belief_task import (HYPOTHESES, INPUT_SIZE as BELIEF_INPUT,
                                    RELIABILITIES, make_belief_batch)
from neurotopo.models import RecurrentModel
from neurotopo.resources import resource_account
from neurotopo.source_belief_study import build_arm
from neurotopo.source_belief_task import (INPUT_SIZE as SOURCE_INPUT,
                                          make_source_belief_batch)


FILES = (
    "src/neurotopo/belief_task.py", "src/neurotopo/source_belief_task.py",
    "src/neurotopo/models.py", "src/neurotopo/resources.py",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def categorical_product(cues: np.ndarray, reliabilities: np.ndarray,
                        end: int) -> np.ndarray:
    result = np.empty((len(cues), HYPOTHESES), dtype=np.float64)
    for example in range(len(cues)):
        weights = np.full(HYPOTHESES, 1 / HYPOTHESES, dtype=np.float64)
        for t in range(end):
            p = float(RELIABILITIES[reliabilities[example, t]])
            likelihood = np.full(HYPOTHESES, (1 - p) / (HYPOTHESES - 1))
            likelihood[cues[example, t]] = p
            weights *= likelihood
        result[example] = weights / weights.sum()
    return result


def joint_product(cues: np.ndarray, sources: np.ndarray, end: int) -> np.ndarray:
    result = np.empty((len(cues), 8), dtype=np.float64)
    for example in range(len(cues)):
        weights = np.full((4, 2), 1 / 8, dtype=np.float64)
        for t in range(end):
            for regime in range(2):
                p = 0.8 if sources[example, t] == regime else 0.4
                likelihood = np.full(4, (1 - p) / 3)
                likelihood[cues[example, t]] = p
                weights[:, regime] *= likelihood
        result[example] = (weights / weights.sum()).reshape(8)
    return result


def check_tags(x: np.ndarray, cues: np.ndarray, second: np.ndarray,
               *, categories: int, second_categories: int, early: int) -> bool:
    position = 0
    for t in range(cues.shape[1]):
        if t == early:
            if not np.all(x[:, position, categories + second_categories + 2] == 1):
                return False
            position += 1
        if not np.all(x[np.arange(len(x)), position, cues[:, t]] == 1):
            return False
        if not np.all(x[np.arange(len(x)), position, categories + second[:, t]] == 1):
            return False
        if not np.all(x[:, position, categories + second_categories] == 1):
            return False
        position += 1
        if not np.all(x[:, position, categories + second_categories + 1] == 1):
            return False
        position += 1
    return position == x.shape[1] - 1 and bool(
        np.all(x[:, position, categories + second_categories + 2] == 1)
    )


def resource_slice(model: RecurrentModel, sequence_length: int) -> dict:
    account = resource_account(model, sequence_length, 128)
    return {
        "parameter_dtype": str(model.weight_ih.dtype),
        "recurrent_hidden_floats": account["hidden_state_floats"],
        "pre_readout_cache_floats": account["pre_readout_cache_floats_per_example"],
        "pre_readout_cache_bytes": account["pre_readout_cache_bytes_per_example"],
        "memory_limit": account["memory_limit"],
    }


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    belief = make_belief_batch(128, np.random.default_rng(710001))
    source = make_source_belief_batch(128, np.random.default_rng(720001))
    b_cues = belief.evidence_cues.numpy()
    b_rels = belief.evidence_reliabilities.numpy()
    s_cues = source.evidence_cues.numpy()
    s_sources = source.evidence_sources.numpy()
    diffs = {
        "cal0001_pre": float(np.max(np.abs(categorical_product(b_cues, b_rels, 3) - belief.pre_posterior.numpy()))),
        "cal0001_final": float(np.max(np.abs(categorical_product(b_cues, b_rels, 6) - belief.final_posterior.numpy()))),
        "cal0002_pre": float(np.max(np.abs(joint_product(s_cues, s_sources, 4) - source.pre_joint.numpy()))),
        "cal0002_final": float(np.max(np.abs(joint_product(s_cues, s_sources, 8) - source.final_joint.numpy()))),
    }
    tags_ok = {
        "cal0001": check_tags(belief.x.numpy(), b_cues, b_rels,
                              categories=4, second_categories=3, early=3),
        "cal0002": check_tags(source.x.numpy(), s_cues, s_sources,
                              categories=4, second_categories=2, early=4),
    }
    source_early_balanced = bool(np.all((s_sources[:, :4] == 0).sum(1) == 2))
    source_final_balanced = bool(np.all((s_sources == 0).sum(1) == 4))
    torch.manual_seed(810001)
    cal_model = RecurrentModel(BELIEF_INPUT, 64, 4, topology="dense")
    cal_account = resource_slice(cal_model, belief.x.shape[1])
    with Path("configs/exp0009_v1.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    arms = {name: build_arm(name, config, 9401) for name in ("multi_gru", "random_gru")}
    arm_accounts = {name: resource_slice(model, source.x.shape[1]) for name, model in arms.items()}
    actual_pre_shapes = {name: list(model(source.x, source.pre_index)[0].shape)
                         for name, model in arms.items()}
    checks = {
        "all_posterior_differences_below_1e_6": all(value < 1e-6 for value in diffs.values()),
        "all_input_evidence_and_query_tags_match": all(tags_ok.values()),
        "balanced_source_counts_at_queries": source_early_balanced and source_final_balanced,
        "cal0001_pre_cache_matches_output": cal_account["pre_readout_cache_floats"] == 4
                                            and cal_account["pre_readout_cache_bytes"] == 16,
        "exp0009_pre_cache_matches_both_outputs": all(
            account["recurrent_hidden_floats"] == 64
            and account["pre_readout_cache_floats"] == 8
            and account["pre_readout_cache_bytes"] == 32
            and arm_accounts[name]["parameter_dtype"] == "torch.float32"
            and actual_pre_shapes[name] == [128, 8]
            for name, account in arm_accounts.items()
        ),
    }
    report = {
        "kind": "TH0001_POST_REFLECTION_IMPLEMENTATION_SANITY_NOT_MODEL_EFFECT",
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "source_sha256": {name: sha256(Path(name).read_bytes()) for name in FILES},
        "examples_per_task": 128,
        "rng_seeds": {"cal0001": 710001, "cal0002": 720001, "model": 810001},
        "max_absolute_posterior_differences": diffs,
        "input_tag_checks": tags_ok,
        "source_balanced_at_pre": source_early_balanced,
        "source_balanced_at_final": source_final_balanced,
        "resource_accounts": {"cal0001_dense": cal_account, **arm_accounts},
        "actual_exp0009_pre_shapes": actual_pre_shapes,
        "checks": checks, "pass": all(checks.values()),
        "scope": "Independent direct likelihood product on fixed synthetic examples; current float32 resource fields; no training, no frozen EXP rewrite, no peak memory measurement.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("research/reflections/TH-0001/implementation_audit.json"))
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({"output": str(args.output), "pass": report["pass"],
                      "max_difference": max(report["max_absolute_posterior_differences"].values())}))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
