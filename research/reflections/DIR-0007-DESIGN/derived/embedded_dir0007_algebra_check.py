"""Deterministic design identities only: no task samples, training, or graph search."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    n = 16
    permutation = [0, 5, 11, 2, 14, 7, 3, 12, 8, 1, 15, 6, 10, 4, 13, 9]
    q = np.eye(n)[permutation]
    s = np.roll(np.eye(n), 1, axis=0)
    powers = [np.linalg.matrix_power(s, k) for k in range(n)]

    def project(w: np.ndarray) -> np.ndarray:
        return sum(g @ w @ g.T for g in powers) / n

    def conjugate_project(w: np.ndarray) -> np.ndarray:
        return q @ project(q.T @ w @ q) @ q.T

    # Full basis checks, not random witnesses.
    basis = np.eye(n * n).reshape(n * n, n, n)
    p = np.stack([project(w).ravel() for w in basis], axis=1)
    pq = np.stack([conjugate_project(w).ravel() for w in basis], axis=1)
    eig = np.linalg.eigvalsh(p)
    eigq = np.linalg.eigvalsh(pq)
    transition = .7 * np.eye(n) + .2 * s + .1 * s.T
    transformed = q @ transition @ q.T
    matrix = (np.arange(n * n).reshape(n, n) % 17 - 8).astype(float) / 17
    penalty = lambda w, fn: float(np.square(w - fn(w)).sum())
    nparam = 7 * n * n + 7 * n
    checks = {
        "projector_idempotence": float(np.max(np.abs(p @ p - p))),
        "projector_self_adjoint": float(np.max(np.abs(p - p.T))),
        "conjugate_idempotence": float(np.max(np.abs(pq @ pq - pq))),
        "conjugate_self_adjoint": float(np.max(np.abs(pq - pq.T))),
        "projector_spectrum_difference": float(np.max(np.abs(eig - eigq))),
        "penalty_conjugacy_error": abs(penalty(matrix, project) - penalty(q @ matrix @ q.T, conjugate_project)),
        "transition_stochastic_error": float(np.max(np.abs(transition.sum(axis=0) - 1))),
        "aligned_transition_projection_error": float(np.max(np.abs(project(transition) - transition))),
        "conjugate_transition_projection_error": float(np.max(np.abs(conjugate_project(transformed) - transformed))),
    }
    mismatch = float(np.linalg.norm(transition - conjugate_project(transition)))
    # Independent entry-wise circular-diagonal implementation.
    direct = np.empty_like(matrix)
    for i in range(n):
        for j in range(n):
            direct[i, j] = sum(matrix[(i + k) % n, (j + k) % n] for k in range(n)) / n
    checks["independent_projection_error"] = float(np.max(np.abs(project(matrix) - direct)))
    result = {
        "scope": "DETERMINISTIC_ALGEBRA_ONLY_NOT_RESEARCH_EFFECT",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy": np.__version__, "n": n, "permutation": permutation,
        "parameter_count": nparam, "projector_rank_per_matrix": int(np.sum(eig > .5)),
        "full_projector_rank": 7*n+7, "residual_rank": nparam-(7*n+7),
        "regularizer_hessian_nonzero_eigenvalue_lambda1": 2/nparam,
        "checks": checks, "mismatched_transition_projection_frobenius": mismatch,
        "checks_pass": max(checks.values()) < 1e-12 and mismatch > .1 and int(np.sum(eig > .5)) == n,
        "task_samples": 0, "optimization_updates": 0,
        "not_established": ["control_headroom", "learning", "effect", "power", "executed_compute_matching", "Gate1_eligibility"],
    }
    output = Path("research/design/dir0007_algebra_check.json")
    if output.exists():
        raise FileExistsError("Preserve prior output; use a new protocol version for reruns")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if not result["checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
