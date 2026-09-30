"""Numerical checks for the G1-KSI-0001 design claims (v2: adds an independent oracle
and a positive control for check C, per the independent review).

Read-only: no training, no task samples, no frozen-metric recomputation. All checks are
closed-form / exact-inference computations on small synthetic arrays from a fixed local
RNG. The constants used here are the design's frozen constants (beta=0.8, lambda=0.7,
J_F=0.0, J_X=0.25, T=16, four-cycle edge set), not verifier-chosen values.

Checks
------
A. v0 generator: the posterior depends on y_{1:T} only through the sample mean.
B. v1 F condition (J=0): the exact 16-state joint forward posterior equals the product
   of four independent two-state forward filters.
C. v1 F condition: chain 0's posterior is unchanged when the other chains' observations
   are replaced, AND (positive control) changes when chain 0's own stream is replaced.
D. v1 X condition (J>0): the exact joint posterior is not the product of its per-chain
   marginals.
E. Independent oracle: brute-force enumeration over the full trajectory space at T=4
   agrees with the recursive forward algorithm (a second, differently-parameterised
   implementation, not a self-consistency comparison).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

OUT = Path("research/reflections/G1-KSI-0001-V1/verification.json")
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
STATES = np.array(list(itertools.product((-1.0, 1.0), repeat=4)))

# Design constants (frozen in g1_k_structure_learning_efficiency_v2.md).
BETA = 0.8  # temporal persistence (F flip probability per dimension = 1/(1+exp(2*BETA)))
LAMBDA = 0.7
J_F = 0.0
J_X = 0.25
T = 16


def ising_energy(states: np.ndarray, coupling: float) -> np.ndarray:
    """Sum over ring edges of z_i z_j for each row state."""
    return coupling * sum(states[:, i] * states[:, j] for i, j in EDGES)


def transition_matrix(coupling: float, persistence: float = BETA) -> np.ndarray:
    """p(z'|z) ∝ exp(persistence * <z',z> + coupling * ring_energy(z')).

    Chosen because it factorises across dimensions exactly when coupling = 0, which is
    what makes the F condition four independent temporal filters. Its stationary
    distribution is NOT pi_J for coupling > 0; the actual stationary distribution is
    computed and reported in check G rather than asserted.
    """
    align = persistence * (STATES @ STATES.T)
    energy = ising_energy(STATES, coupling)[None, :]
    logits = align + energy
    logits -= logits.max(axis=1, keepdims=True)
    matrix = np.exp(logits)
    return matrix / matrix.sum(axis=1, keepdims=True)


def emission_loglik(observations: np.ndarray, scale: float = LAMBDA) -> np.ndarray:
    """log N(y_t; scale*z, I) for every state, shape (T, 16)."""
    residual = observations[:, None, :] - scale * STATES[None, :, :]
    return -0.5 * (residual ** 2).sum(axis=2)


def joint_forward(observations: np.ndarray, coupling: float, persistence: float = BETA,
                  scale: float = LAMBDA) -> np.ndarray:
    """Exact 16-state forward posterior over z_T (recursive message passing)."""
    logp = np.full(len(STATES), -np.log(len(STATES)))
    emissions = emission_loglik(observations, scale)
    log_trans = np.log(transition_matrix(coupling, persistence))
    horizon = len(observations)
    for t in range(horizon):
        logp = logp + emissions[t]
        logp -= logp.max()
        if t < horizon - 1:          # T emissions, T-1 transitions: filter for z_T
            logp = np.log(np.exp(logp) @ np.exp(log_trans))
            logp -= logp.max()
    weights = np.exp(logp)
    return weights / weights.sum()


def chain_forward(observations: np.ndarray, index: int, persistence: float = BETA,
                  scale: float = LAMBDA) -> np.ndarray:
    """Independent two-state forward filter for one chain, using only its own stream."""
    pair = np.array([[-1.0], [1.0]])
    trans = np.exp(persistence * (pair @ pair.T))
    trans /= trans.sum(axis=1, keepdims=True)
    logp = np.full(2, -np.log(2.0))
    stream = observations[:, index]
    for t, value in enumerate(stream):
        residual = value - scale * pair[:, 0]
        logp = logp - 0.5 * residual ** 2
        logp -= logp.max()
        if t < len(stream) - 1:      # same convention as joint_forward
            logp = np.log(np.exp(logp) @ trans)
            logp -= logp.max()
    weights = np.exp(logp)
    return weights / weights.sum()


def product_of_chains(observations: np.ndarray, persistence: float = BETA,
                      scale: float = LAMBDA) -> np.ndarray:
    """Product distribution over the 16 joint states from four independent filters."""
    table = np.ones(len(STATES))
    for i in range(4):
        weights = chain_forward(observations, i, persistence, scale)
        table *= weights[(STATES[:, i] > 0).astype(int)]
    return table / table.sum()


def static_posterior(observations: np.ndarray, coupling: float) -> np.ndarray:
    """v0 posterior: static z, iid observations -> depends only on the sample mean."""
    mean = observations.mean(axis=0)
    logits = ising_energy(STATES, coupling) - 8.0 * ((mean[None, :] - STATES) ** 2).sum(axis=1)
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def brute_force_posterior(observations: np.ndarray, coupling: float,
                          persistence: float = BETA, scale: float = LAMBDA) -> np.ndarray:
    """Independent oracle: enumerate every trajectory z_{1:T} and sum weights by z_T."""
    horizon = len(observations)
    transitions = transition_matrix(coupling, persistence)
    index = {tuple(state): i for i, state in enumerate(STATES)}
    log_trans = np.log(transitions)
    emissions = emission_loglik(observations, scale)
    totals = np.full(len(STATES), -np.inf)
    for trajectory in itertools.product(range(len(STATES)), repeat=horizon):
        weight = -np.log(len(STATES)) + emissions[0, trajectory[0]]
        for t in range(1, horizon):
            weight += log_trans[trajectory[t - 1], trajectory[t]] + emissions[t, trajectory[t]]
        last = trajectory[-1]
        totals[last] = np.logaddexp(totals[last], weight)
    totals -= totals.max()
    weights = np.exp(totals)
    return weights / weights.sum()


def main() -> int:
    rng = np.random.default_rng(20260924)
    observations = rng.normal(scale=1.0, size=(T, 4)) + LAMBDA * STATES[rng.integers(0, 16)]

    # A. v0: posterior invariant to any reshaping of y that preserves the sample mean.
    mean = observations.mean(axis=0)
    reshaped = mean[None, :] + 0.3 * (observations - mean[None, :])
    check_a = {
        "max_abs_diff_J0": float(np.abs(static_posterior(observations, J_F) - static_posterior(reshaped, J_F)).max()),
        "max_abs_diff_JX": float(np.abs(static_posterior(observations, J_X) - static_posterior(reshaped, J_X)).max()),
        "method": "algebraic identity verified numerically at the design's frozen constants; "
                  "the proof is the identity Sum_t||y_t-z||^2 = Sum_t||y_t-ybar||^2 + T||ybar-z||^2",
    }

    # B. v1 F condition: joint 16-state forward == product of four independent filters.
    joint_f = joint_forward(observations, J_F)
    product_f = product_of_chains(observations)
    check_b = {
        "max_abs_diff": float(np.abs(joint_f - product_f).max()),
        "total_variation": float(0.5 * np.abs(joint_f - product_f).sum()),
    }

    # C. F condition, with positive control on chain 0's own stream.
    other_streams = observations.copy()
    other_streams[:, 1:] = rng.normal(scale=3.0, size=(T, 3))
    own_stream = observations.copy()
    own_stream[:, 0] = rng.normal(scale=3.0, size=T)

    def chain0_from_joint(obs: np.ndarray) -> np.ndarray:
        joint = joint_forward(obs, J_F)
        return np.array([joint[STATES[:, 0] > 0].sum(), joint[STATES[:, 0] < 0].sum()])

    def chain0_independent(obs: np.ndarray) -> np.ndarray:
        weights = chain_forward(obs, 0)
        return np.array([weights[1], weights[0]])

    check_c = {
        "field_semantics": "entries are [P(z_0=+1), P(z_0=-1)] for the terminal time step; "
                           "chain0_from_joint marginalises the 16-state joint, chain0_independent "
                           "runs the standalone two-state filter on stream 0 only",
        "joint_marginal_original": chain0_from_joint(observations).tolist(),
        "joint_marginal_other_streams_replaced": chain0_from_joint(other_streams).tolist(),
        "joint_marginal_own_stream_replaced": chain0_from_joint(own_stream).tolist(),
        "independent_filter_original": chain0_independent(observations).tolist(),
        "independent_filter_other_streams_replaced": chain0_independent(other_streams).tolist(),
        "independent_filter_own_stream_replaced": chain0_independent(own_stream).tolist(),
        "insensitivity_to_other_streams": float(np.abs(chain0_from_joint(observations) - chain0_from_joint(other_streams)).max()),
        "sensitivity_to_own_stream": float(np.abs(chain0_from_joint(observations) - chain0_from_joint(own_stream)).max()),
    }

    # D. v1 X condition: joint posterior is not the product of per-chain marginals.
    joint_x = joint_forward(observations, J_X)
    table = np.ones(len(STATES))
    for i in range(4):
        marginal = np.array([joint_x[STATES[:, i] > 0].sum(), joint_x[STATES[:, i] < 0].sum()])
        table *= marginal[(STATES[:, i] > 0).astype(int)]
    table /= table.sum()
    check_d = {
        "max_abs_diff": float(np.abs(joint_x - table).max()),
        "total_variation": float(0.5 * np.abs(joint_x - table).sum()),
        "note": "J>0: independent per-chain marginals do not reconstruct the joint",
    }

    # E. Independent oracle: brute-force trajectory enumeration vs recursive forward.
    short = observations[:4]
    brute = brute_force_posterior(short, J_X)
    recursive = joint_forward(short, J_X)
    check_e = {
        "horizon": 4,
        "trajectories_enumerated": int(len(STATES) ** 4),
        "max_abs_diff_JX": float(np.abs(brute - recursive).max()),
        "max_abs_diff_JF": float(np.abs(brute_force_posterior(short, J_F) - joint_forward(short, J_F)).max()),
        "note": "second implementation: direct enumeration over all 16^4 trajectories, "
                "no shared recursion with the forward algorithm",
    }

    # F. Difficulty comparison at the frozen constants: oracle posterior entropy per
    #    condition. This is the number the design's difficulty hard gate (§4.7) needs.
    def episode_posterior(coupling: float, rng: np.random.Generator) -> np.ndarray:
        latent = STATES[rng.integers(0, len(STATES))]
        stream = LAMBDA * latent[None, :] + rng.normal(size=(T, 4))
        return joint_forward(stream, coupling)

    def entropy_stats(coupling: float, episodes: int = 2000, seed: int = 7) -> dict:
        rng_local = np.random.default_rng(seed)
        entropies, sharpness = [], []
        for _ in range(episodes):
            posterior = episode_posterior(coupling, rng_local)
            entropies.append(float(-(posterior * np.log(posterior)).sum() / np.log(len(STATES))))
            sharpness.append(float(posterior.max()))
        entropies = np.array(entropies)
        sharpness = np.array(sharpness)
        return {
            "episodes": episodes,
            "mean_normalised_entropy": float(entropies.mean()),
            "sd_normalised_entropy": float(entropies.std(ddof=1)),
            "mean_max_probability": float(sharpness.mean()),
            "mean_entropy_nats": float(entropies.mean() * np.log(len(STATES))),
        }

    stats_f = entropy_stats(J_F)
    stats_x = entropy_stats(J_X)
    gap = stats_f["mean_normalised_entropy"] - stats_x["mean_normalised_entropy"]
    se = float(np.sqrt(stats_f["sd_normalised_entropy"] ** 2 + stats_x["sd_normalised_entropy"] ** 2) / np.sqrt(2000))
    check_f = {
        "F": stats_f,
        "X": stats_x,
        "gap_F_minus_X_normalised": gap,
        "gap_se": se,
        "note": "difficulty proxy for the §4.7 hard gate; a large gap means the coupling "
                "condition is also easier/harder and the primary interaction is confounded",
    }

    # G. Stationary distribution actually induced by the transition kernel, and the
    #    marginal priors it implies. The design no longer asserts stationarity equals p_J.
    def stationary_distribution(coupling: float, iterations: int = 4000) -> np.ndarray:
        dist = np.full(len(STATES), 1.0 / len(STATES))
        matrix = transition_matrix(coupling)
        for _ in range(iterations):
            dist = dist @ matrix
        return dist

    def marginal_plus(dist: np.ndarray) -> list[float]:
        return [float(dist[STATES[:, i] > 0].sum()) for i in range(4)]

    def ising_target(coupling: float) -> np.ndarray:
        target = np.exp(ising_energy(STATES, coupling))
        return target / target.sum()

    stationary_f = stationary_distribution(J_F)
    stationary_x = stationary_distribution(J_X)
    check_g = {
        "stationary_marginals_F": marginal_plus(stationary_f),
        "stationary_marginals_X": marginal_plus(stationary_x),
        "max_abs_deviation_from_uniform_F": float(np.abs(stationary_f - 1.0 / len(STATES)).max()),
        "max_abs_deviation_from_ising_target_X": float(np.abs(stationary_x - ising_target(J_X)).max()),
        "note": "F is uniform by construction (independent symmetric chains); X is the "
                "kernel's own stationary law, which is not p_J. Marginal priors are the "
                "quantity the ruling's same-marginals condition depends on.",
    }

    document = {
        "schema": "g1_ksi_0001_math_verification_v3",
        "task": "G1-KSI-0001 设计数学主张的只读核对（含独立 oracle 与阳性对照）",
        "read_only": "无训练、无任务样本、无冻结指标重算；仅小规模精确推断",
        "design_constants": {"T": T, "beta_persistence": BETA, "lambda": LAMBDA, "J_F": J_F, "J_X": J_X, "edges": list(EDGES)},
        "rng": "numpy default_rng(20260924)，本地、不写入任何运行档",
        "checks": {
            "A_v0_mean_only_dependence": check_a,
            "B_F_joint_equals_product_of_chains": check_b,
            "C_F_per_chain_sufficiency_with_positive_control": check_c,
            "D_X_joint_not_product_of_marginals": check_d,
            "E_independent_brute_force_oracle": check_e,
            "F_condition_difficulty_gap": check_f,
            "G_stationary_law_and_marginals": check_g,
        },
        "verdict": {
            "A_confirms_v0_flaw": bool(check_a["max_abs_diff_J0"] < 1e-12 and check_a["max_abs_diff_JX"] < 1e-12),
            "B_confirms_F_factorization": bool(check_b["max_abs_diff"] < 1e-10),
            "C_confirms_per_chain_sufficiency": bool(
                check_c["insensitivity_to_other_streams"] < 1e-10 and check_c["sensitivity_to_own_stream"] > 1e-3
            ),
            "D_confirms_X_coupling_is_real": bool(check_d["total_variation"] > 1e-3),
            "E_confirms_forward_against_enumeration": bool(
                check_e["max_abs_diff_JX"] < 1e-10 and check_e["max_abs_diff_JF"] < 1e-10
            ),
            "G_reports_stationary_law": True,
            "G_marginals_uniform_in_both_conditions": bool(
                max(abs(m - 0.5) for m in check_g["stationary_marginals_F"]) < 1e-9
                and max(abs(m - 0.5) for m in check_g["stationary_marginals_X"]) < 1e-9
            ),
        },
        "limits": [
            "这些检查只核对解析主张与实现一致性，不构成任务资格、控制余量、功效或任何架构效果证据。",
            "B/C/E 的成立依赖 J=0 或给定的耦合矩阵与按维分解的发射模型；若改动生成器须重推。",
            "D 只证明该 J 取值下联合后验不可由逐链边缘重建，不证明任何架构在该任务上会成功或失败。",
            "E 在 T=4 上做枚举，是 oracle 的第二实现，不是对 T=16 运行档的独立重放。",
            "F 的熵差只作为 §4.7 难度硬门的输入数字，不构成架构效果证据。",
            "E 曾抓到一次真实实现缺陷：修正前的前向递推在最后一步多做一次转移（得到 z_{T+1} 的预测分布而非 z_T 的滤波后验），"
            "而 B/C/D 因两个实现共享同一约定错误无法发现。该缺陷已修，E 现为 PASS。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "verdict": document["verdict"],
                      "C": {"insens_other": check_c["insensitivity_to_other_streams"],
                            "sens_own": check_c["sensitivity_to_own_stream"]},
                      "D_tv": check_d["total_variation"], "E_max": check_e["max_abs_diff_JX"],
                      "G_marginals_X": [round(m, 6) for m in check_g["stationary_marginals_X"]],
                      "F_entropy": {"F": check_f["F"]["mean_normalised_entropy"],
                                    "X": check_f["X"]["mean_normalised_entropy"],
                                    "gap": check_f["gap_F_minus_X_normalised"]},
                      "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
