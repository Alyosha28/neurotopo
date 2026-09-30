"""Graph-only existence witnesses; not a scientific null sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neurotopo.models import recurrent_mask, strongly_connected
from neurotopo.run import digest, source_fingerprints, write_json


def cross_count(adjacency: np.ndarray) -> int:
    blocks = np.arange(64) // 16
    return int((adjacency & (blocks[:, None] != blocks[None, :])).sum())


def reciprocity(adjacency: np.ndarray) -> float:
    off = adjacency.copy()
    np.fill_diagonal(off, False)
    return float((off & off.T).sum() / off.sum())


def second_eigenvalue(adjacency: np.ndarray) -> float:
    row_degree = adjacency.sum(1)
    return float(np.sort(np.abs(np.linalg.eigvals(adjacency.astype(np.float64) / row_degree[:, None])))[::-1][1])


def full_mask_sha256(adjacency: np.ndarray) -> str:
    return digest(torch.from_numpy(adjacency.astype(np.float32)).repeat(3, 1).numpy().tobytes())


def witness(adjacency: np.ndarray, seed: int, max_attempts: int = 100_000) -> dict:
    before = adjacency.astype(bool)
    if before.shape != (64, 64) or not strongly_connected(before):
        raise ValueError("Expected a strongly connected H64 directed graph")
    original_cross = cross_count(before)
    original_reciprocity = reciprocity(before)
    original_in, original_out = before.sum(1), before.sum(0)
    original_diagonal = np.diag(before).copy()
    edges = np.argwhere(before & ~np.eye(64, dtype=bool))
    rng = np.random.default_rng(seed)
    blocks = np.arange(64) // 16
    for attempt in range(1, max_attempts + 1):
        first, second = rng.integers(len(edges), size=2)
        i, j = (int(value) for value in edges[first])
        k, l = (int(value) for value in edges[second])
        if i == k or j == l or i == l or k == j or before[i, l] or before[k, j]:
            continue
        old_cross = int(blocks[i] != blocks[j]) + int(blocks[k] != blocks[l])
        new_cross = int(blocks[i] != blocks[l]) + int(blocks[k] != blocks[j])
        if old_cross != new_cross:
            continue
        changed = before.copy()
        changed[i, j] = changed[k, l] = False
        changed[i, l] = changed[k, j] = True
        changed_reciprocity = reciprocity(changed)
        if changed_reciprocity == original_reciprocity or not strongly_connected(changed):
            continue
        if (not np.array_equal(changed.sum(1), original_in) or
                not np.array_equal(changed.sum(0), original_out) or
                not np.array_equal(np.diag(changed), original_diagonal) or
                cross_count(changed) != original_cross):
            raise RuntimeError("Graph invariant failed after candidate swap")
        return {"found": True, "attempts": attempt,
                "removed_edges_receiver_sender": [[i, j], [k, l]],
                "added_edges_receiver_sender": [[i, l], [k, j]],
                "original_mask_sha256": full_mask_sha256(before),
                "changed_mask_sha256": full_mask_sha256(changed),
                "original_cross_edges": original_cross, "changed_cross_edges": cross_count(changed),
                "original_reciprocity": original_reciprocity,
                "changed_reciprocity": changed_reciprocity,
                "original_second_eigenvalue": second_eigenvalue(before),
                "changed_second_eigenvalue": second_eigenvalue(changed),
                "degree_diagonal_connectivity_preserved": True}
    return {"found": False, "attempts": max_attempts,
            "original_mask_sha256": full_mask_sha256(before),
            "original_cross_edges": original_cross,
            "original_reciprocity": original_reciprocity}


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for seed in range(9401, 9411):
        mask, _ = recurrent_mask(64, 4, "random", 695000 + seed)
        original_sha = digest(mask.repeat(3, 1).numpy().tobytes())
        result_path = Path("artifacts/EXP-0009/v1") / f"seed-{seed}" / "random_gru" / "result.json"
        recorded = json.loads(result_path.read_text())
        if original_sha != recorded["mask_sha256"]:
            raise ValueError(f"Frozen source mask mismatch seed={seed}")
        result = witness(mask.numpy().astype(bool), 505000 + seed)
        rows.append({"seed": seed, "result_sha256": digest(result_path.read_bytes()), **result})
    successes = sum(row["found"] for row in rows)
    report = {
        "kind": "GRAPH_ONLY_EXISTENCE_WITNESSES_NOT_NULL_MODEL_OR_TASK_EVIDENCE",
        "plan": "research/design/h1_reciprocity_feasibility.md",
        "source_sha256": source_fingerprints(),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "experiment": "EXP-0009", "seeds": list(range(9401, 9411)),
        "max_attempts_per_seed": 100_000,
        "witnesses_found": successes,
        "at_least_eight_witnesses": successes >= 8,
        "rows": rows,
        "interpretation_limit": "One local degree/self-loop/cross-edge/connectedness preserving swap per seed; no uniform null, no spectrum matching, no model training.",
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/analysis/h1_reciprocity_feasibility_v1.json"))
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"output": str(args.output), "found": result["witnesses_found"],
                      "at_least_eight": result["at_least_eight_witnesses"]}), flush=True)


if __name__ == "__main__":
    main()
