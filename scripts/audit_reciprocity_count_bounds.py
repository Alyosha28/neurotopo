"""Count-only reciprocity upper bounds on frozen C0 graphs; generates no new graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from neurotopo.models import recurrent_mask, strongly_connected


SEEDS = tuple(range(9401, 9411))
WIDTH = 16
NODES = 64
EDGES = 1024
OFF_DIAGONAL_EDGES = 960
TARGET_ADDED_RECIPROCAL_EDGES = 96  # +0.10 of 960 off-diagonal directed edges
DESIGN = Path("research/design/h1_reciprocity_count_bound_protocol.md")
PRIOR_WITNESS = Path("artifacts/analysis/h1_reciprocity_feasibility_v1.json")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def edge_counts(adjacency: np.ndarray) -> np.ndarray:
    return np.array(
        [[int(adjacency[a * WIDTH:(a + 1) * WIDTH,
                         b * WIDTH:(b + 1) * WIDTH].sum())
          for b in range(4)] for a in range(4)], dtype=np.int64,
    )


def bound_row(adjacency: np.ndarray, seed: int, formal_mask_sha: str,
              formal_result_sha: str, witness: dict) -> dict:
    if adjacency.shape != (NODES, NODES) or int(adjacency.sum()) != EDGES:
        raise ValueError(f"seed={seed}: graph size/edge count mismatch")
    if not np.all(adjacency.sum(axis=0) == 16) or not np.all(adjacency.sum(axis=1) == 16):
        raise ValueError(f"seed={seed}: per-node in/out degree mismatch")
    if not np.all(np.diag(adjacency)) or not strongly_connected(adjacency):
        raise ValueError(f"seed={seed}: self-loops or strong connectivity mismatch")
    counts = edge_counts(adjacency)
    cross = int(counts.sum() - np.trace(counts))
    off = adjacency.copy()
    np.fill_diagonal(off, False)
    reciprocal_edges = int((off & off.T).sum())
    if int(off.sum()) != OFF_DIAGONAL_EDGES or reciprocal_edges % 2:
        raise ValueError(f"seed={seed}: off-diagonal/reciprocity mismatch")
    current_mask_sha = sha256(np.tile(adjacency.astype(np.float32), (3, 1)).tobytes())
    if current_mask_sha != formal_mask_sha or current_mask_sha != witness["original_mask_sha256"]:
        raise ValueError(f"seed={seed}: formal/witness mask identity mismatch")
    if cross != witness["original_cross_edges"] or not np.isclose(
        reciprocal_edges / OFF_DIAGONAL_EDGES, witness["original_reciprocity"],
        atol=1e-15, rtol=0
    ):
        raise ValueError(f"seed={seed}: witness metrics mismatch")
    total_bound = 2 * (cross // 2) + 2 * ((OFF_DIAGONAL_EDGES - cross) // 2)
    within_bound = sum(2 * ((int(counts[a, a]) - WIDTH) // 2) for a in range(4))
    between_bound = sum(
        2 * min(int(counts[a, b]), int(counts[b, a]))
        for a in range(4) for b in range(a + 1, 4)
    )
    block_bound = within_bound + between_bound
    target_edges = reciprocal_edges + TARGET_ADDED_RECIPROCAL_EDGES
    return {
        "seed": seed,
        "formal_result_sha256": formal_result_sha,
        "formal_mask_sha256": formal_mask_sha,
        "block_edge_counts_receiver_by_sender": counts.tolist(),
        "cross_edges": cross,
        "fixed_partition_q": 0.75 - cross / EDGES,
        "original_reciprocal_directed_edges": reciprocal_edges,
        "original_reciprocity": reciprocal_edges / OFF_DIAGONAL_EDGES,
        "target_reciprocal_directed_edges": target_edges,
        "target_reciprocity": target_edges / OFF_DIAGONAL_EDGES,
        "total_cross_count_upper_bound_edges": total_bound,
        "full_block_matrix_upper_bound_edges": block_bound,
        "total_cross_count_bound_excludes_target": total_bound < target_edges,
        "full_block_matrix_bound_excludes_target": block_bound < target_edges,
    }


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    witness_data = json.loads(PRIOR_WITNESS.read_text(encoding="utf-8"))
    witnesses = {row["seed"]: row for row in witness_data["rows"]}
    if set(witnesses) != set(SEEDS):
        raise ValueError("Prior witness seed set mismatch")
    rows = []
    for seed in SEEDS:
        mask, _ = recurrent_mask(NODES, 4, "random", 695000 + seed)
        adjacency = mask.numpy().astype(bool)
        result_path = Path("artifacts/EXP-0009/v1") / f"seed-{seed}" / "random_gru" / "result.json"
        raw = result_path.read_bytes()
        formal = json.loads(raw)
        if formal.get("seed") != seed or formal.get("model") != "random_gru":
            raise ValueError(f"seed={seed}: formal result identity mismatch")
        rows.append(bound_row(adjacency, seed, formal["mask_sha256"],
                              sha256(raw), witnesses[seed]))
    excluded_total = sum(row["total_cross_count_bound_excludes_target"] for row in rows)
    excluded_block = sum(row["full_block_matrix_bound_excludes_target"] for row in rows)
    if excluded_total >= 8:
        label = "TARGET_EXCLUDED_UNDER_CURRENT_COUNT_CONSTRAINTS"
    elif excluded_block >= 8:
        label = "TARGET_EXCLUDED_ONLY_UNDER_STRONGER_BLOCK_COUNTS"
    else:
        label = "BOUND_NONDECISIVE"
    report = {
        "kind": "GRAPH_ONLY_NECESSARY_COUNT_BOUNDS_NOT_REACHABILITY_OR_NULL_EVIDENCE",
        "design": str(DESIGN), "design_sha256": sha256(DESIGN.read_bytes()),
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "prior_witness_sha256": sha256(PRIOR_WITNESS.read_bytes()),
        "seeds": list(SEEDS), "target_added_reciprocity": 0.10,
        "target_added_reciprocal_directed_edges": TARGET_ADDED_RECIPROCAL_EDGES,
        "excluded_by_total_count": excluded_total,
        "excluded_by_full_block_matrix": excluded_block,
        "decision_label": label, "rows": rows,
        "scope": "Necessary count-only upper bounds. Full block matrix is stronger than the registered total-cross constraint. No new graphs, sampling, spectrum matching, model training or task scores.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/analysis/h1_reciprocity_count_bound_v1.json"))
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"output": str(args.output), "label": result["decision_label"],
                      "excluded_total": result["excluded_by_total_count"],
                      "excluded_block": result["excluded_by_full_block_matrix"]}), flush=True)


if __name__ == "__main__":
    main()
