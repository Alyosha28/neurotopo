"""Read-only matched-graph confounding audit for frozen EXP-0004 and EXP-0009."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neurotopo.models import recurrent_mask, strongly_connected
from neurotopo.run import digest, source_fingerprints, write_json


def graph_measures(mask: np.ndarray) -> dict[str, float | int | bool | list[int]]:
    if mask.shape != (64, 64):
        raise ValueError("This audit is frozen to H64")
    adjacency = mask.astype(bool)
    groups = np.arange(64) // 16
    within = groups[:, None] == groups[None, :]
    indegree = adjacency.sum(1)
    outdegree = adjacency.sum(0)
    edges = int(adjacency.sum())
    expected_within = sum(int(indegree[groups == group].sum()) *
                          int(outdegree[groups == group].sum()) / edges
                          for group in range(4))
    modularity = (float((adjacency & within).sum()) - expected_within) / edges
    off = adjacency.copy()
    np.fill_diagonal(off, False)
    reciprocal_fraction = float((off & off.T).sum() / off.sum())
    transition = adjacency.astype(np.float64) / indegree[:, None]
    eigenvalues = np.sort(np.abs(np.linalg.eigvals(transition)))[::-1]
    return {
        "nodes": 64, "edges": edges, "self_loops": int(np.trace(adjacency)),
        "in_degree": indegree.astype(int).tolist(), "out_degree": outdegree.astype(int).tolist(),
        "cross_original_block_edges": int((adjacency & ~within).sum()),
        "directed_modularity_original_blocks": float(modularity),
        "reciprocal_offdiagonal_edge_fraction": reciprocal_fraction,
        "strongly_connected": bool(strongly_connected(adjacency)),
        "second_absolute_eigenvalue_row_normalized": float(eigenvalues[1]),
    }


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    studies = []
    for experiment, version, config_name, seeds, offset in (
            ("EXP-0004", "v1", "configs/exp0004_v1.json", range(9101, 9111), 200000),
            ("EXP-0009", "v1", "configs/exp0009_v1.json", range(9401, 9411), 695000)):
        config_path = Path(config_name)
        config = json.loads(config_path.read_text())
        if experiment == "EXP-0004":
            base = json.loads(Path(config["base_config"]).read_text())
            if base["graph_seed_offset"] != offset:
                raise ValueError("EXP-0004 graph seed offset changed")
        elif config["graph_seed_offset"] != offset:
            raise ValueError("EXP-0009 graph seed offset changed")
        if config["seeds"] != list(seeds):
            raise ValueError(f"{experiment} seed set changed")
        rows = []
        for seed in seeds:
            paired = {}
            for name, topology in (("multi_gru", "block"), ("random_gru", "random")):
                mask, metadata = recurrent_mask(64, 4, topology, offset + seed)
                mask_sha = digest(mask.repeat(3, 1).numpy().tobytes())
                result_path = Path("artifacts") / experiment / version / f"seed-{seed}" / name / "result.json"
                result = json.loads(result_path.read_text())
                if mask_sha != result["mask_sha256"]:
                    raise ValueError(f"Frozen mask mismatch {experiment}/{seed}/{name}")
                metrics = graph_measures(mask.numpy())
                if metrics["self_loops"] != metadata["diagonal_connections"] or (
                        metrics["in_degree"] != metadata["in_degree"] or
                        metrics["out_degree"] != metadata["out_degree"]):
                    raise ValueError(f"Graph metadata mismatch {experiment}/{seed}/{name}")
                if experiment == "EXP-0009" and metrics["cross_original_block_edges"] != result["cross_original_block_edges"]:
                    raise ValueError(f"Frozen cross-block count mismatch {seed}/{name}")
                paired[name] = {"mask_sha256": mask_sha, "result_sha256": digest(result_path.read_bytes()),
                                "measures": metrics}
            a, b = paired["multi_gru"]["measures"], paired["random_gru"]["measures"]
            if any(a[key] != b[key] for key in ("nodes", "edges", "self_loops", "in_degree", "out_degree")):
                raise ValueError(f"Degree/edge/self-loop match failed {experiment}/{seed}")
            rows.append({"seed": seed, "models": paired})
        studies.append({"experiment": experiment, "version": version,
                        "config_sha256": digest(config_path.read_bytes()), "rows": rows})
    report = {
        "kind": "READ_ONLY_GRAPH_CONFOUNDING_AUDIT_NOT_PERFORMANCE_EVIDENCE",
        "plan": "research/design/h1_h2_existing_evidence_audit.md",
        "source_sha256": source_fingerprints(), "script_sha256": digest(Path(__file__).read_bytes()),
        "all_frozen_mask_and_matched_count_checks_pass": True,
        "formula_scope": "Directed modularity on fixed original four-block partition; off-diagonal reciprocal directed-edge fraction; second absolute eigenvalue of row-normalized adjacency. No performance correlation or causal mediation claim.",
        "studies": studies,
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/analysis/h1_h2_graph_audit_v1.json"))
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"output": str(args.output), "studies": len(result["studies"]),
                      "pairs": sum(len(study["rows"]) for study in result["studies"]),
                      "pass": result["all_frozen_mask_and_matched_count_checks_pass"]}), flush=True)


if __name__ == "__main__":
    main()
