"""CAL-0008 terminal-headroom screen (v2, post pre-run review).

Trains one dense GRU control per (configuration, seed, learning rate) cell and measures
the oracle-relative KL of its final-step posterior across the registered budget, to answer
whether an acceptable task configuration exists in which the matched dense control does NOT
saturate at the registered budget.

v2 implements the pre-run review's required changes:
  * relative + absolute headroom gates (a fixed absolute nat line is scale-dependent);
  * a learning floor relative to each configuration's own untrained KL;
  * a plateau/convergence test so that undertraining is INCONCLUSIVE, not HEADROOM;
  * learning-rate selection on a separate training-side validation stream;
  * a decision statistic that is fully specified (final-step posterior KL, averaged over a
    pre-registered held-out episode count, with episode-level Monte Carlo error reported);
  * an independent scaled-probability oracle at the production horizon;
  * a menu that separates the capacity axis from the horizon/evidence axis, plus a
    frustrated-coupling point.

Design: research/design/cal0008_terminal_headroom_predesign.md
Nothing here is architecture evidence; the screen only characterises the control.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------- registered constants
CONFIGS = {
    # sanity point: pipeline check only, excluded from the headroom answer
    "C1": {"M": 4, "T": 16, "lam": 0.7, "graph_seed": 4101, "frustrated": False, "role": "sanity"},
    # capacity axis: M grows, horizon and evidence strength held at C1's values
    "C2": {"M": 8, "T": 16, "lam": 0.7, "graph_seed": 8307, "frustrated": False, "role": "capacity"},
    # horizon/evidence axis: M held, T and lambda moved
    "C3": {"M": 8, "T": 32, "lam": 0.4, "graph_seed": 8307, "frustrated": False, "role": "horizon"},
    # frustration axis: mixed-sign couplings with odd cycles, same M/T/lambda as C3
    "C4": {"M": 8, "T": 32, "lam": 0.4, "graph_seed": 8307, "frustrated": True, "role": "frustration"},
}
ANSWER_ORDER = ("C2", "C3", "C4")   # C1 is never the answer
BETA = 0.8
COUPLING = 0.25
SEEDS = (0, 1, 2)
LRS = (1e-3, 3e-3)
UPDATES = 4000
BATCH = 128
EVAL_EPISODES = 512          # held-out stream, never used for selection
VALIDATION_EPISODES = 256    # training-side stream, used only to pick the learning rate
KL_GRID = (0, 250, 500, 1000, 2000, 4000)

EPSILON_ABS = 0.05      # nats: absolute residual floor
RHO_REL = 0.05          # terminal KL must exceed this fraction of the untrained KL
KAPPA_LEARNED = 0.90    # must have removed at least this fraction of the untrained KL
LR_TOL = 0.02           # nats: the two registered LRs must agree this closely, or both must clear kappa
TAU_PLATEAU = 0.01      # nats: |KL(u_i) - KL(u_{i+1})| bound over the final intervals
UNTRAINED_FLOOR = 0.10  # nats: below this the relative gates are ill-conditioned
UNTRAINED_CEILING_SLACK = 0.5  # nats: untrained KL must not exceed ln 2**M + this slack


def state_table(m: int) -> np.ndarray:
    return np.array(list(itertools.product((-1.0, 1.0), repeat=m)))


def signed_graph(m: int, seed: int, frustrated: bool) -> tuple[tuple[int, int, float], ...]:
    """Fixed sparse signed graph; frustrated variant adds negative chords (odd cycles)."""
    rng = np.random.default_rng(seed)
    nodes = list(range(m))
    rng.shuffle(nodes)
    edges: dict[tuple[int, int], float] = {}
    for i in range(m):
        a, b = sorted((nodes[i], nodes[(i + 1) % m]))
        edges[(a, b)] = 1.0
    if frustrated and m >= 4:
        for i in range(0, m, 2):
            a, b = sorted((nodes[i], nodes[(i + 2) % m]))
            if a != b:
                edges[(a, b)] = -1.0
    return tuple((a, b, sign) for (a, b), sign in sorted(edges.items()))


def energy(states: np.ndarray, edges) -> np.ndarray:
    return sum(sign * states[:, i] * states[:, j] for i, j, sign in edges)


def transition_matrix(states: np.ndarray, edges, beta: float = BETA, coupling: float = COUPLING) -> np.ndarray:
    logits = beta * (states @ states.T) + coupling * energy(states, edges)[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    matrix = np.exp(logits)
    return matrix / matrix.sum(axis=1, keepdims=True)


def emission_loglik(observations: np.ndarray, states: np.ndarray, lam: float) -> np.ndarray:
    residual = observations[:, None, :] - lam * states[None, :, :]
    return -0.5 * (residual ** 2).sum(axis=2)


def oracle_log_space(observations: np.ndarray, states: np.ndarray, edges, lam: float) -> np.ndarray:
    """Exact HMM forward recursion in log space (T emissions, T-1 transitions)."""
    logp = np.full(len(states), -math.log(len(states)))
    log_trans = np.log(transition_matrix(states, edges))
    emissions = emission_loglik(observations, states, lam)
    horizon = len(observations)
    for t in range(horizon):
        logp = logp + emissions[t]
        logp -= logp.max()
        if t < horizon - 1:
            logp = np.log(np.exp(logp) @ np.exp(log_trans))
            logp -= logp.max()
    weights = np.exp(logp)
    return weights / weights.sum()


def oracle_scaled_probability(observations: np.ndarray, states: np.ndarray, edges, lam: float) -> np.ndarray:
    """Independent implementation: linear-probability recursion with per-step rescaling.

    Different arithmetic path from oracle_log_space (no log-sum-exp anywhere), so agreement
    at the production horizon is evidence against a shared log-space defect.
    """
    trans = transition_matrix(states, edges)
    probs = np.full(len(states), 1.0 / len(states))
    horizon = len(observations)
    for t in range(horizon):
        probs = probs * np.exp(emission_loglik(observations, states, lam)[t] - emission_loglik(observations, states, lam)[t].max())
        probs /= probs.sum()
        if t < horizon - 1:
            probs = probs @ trans
            probs /= probs.sum()
    return probs / probs.sum()


def oracle_by_enumeration(observations: np.ndarray, states: np.ndarray, edges, lam: float) -> np.ndarray:
    """Third implementation for short horizons: enumerate every trajectory."""
    trans = transition_matrix(states, edges)
    log_trans = np.log(trans)
    emissions = emission_loglik(observations, states, lam)
    horizon = len(observations)
    totals = np.full(len(states), -np.inf)
    for trajectory in itertools.product(range(len(states)), repeat=horizon):
        weight = -math.log(len(states)) + emissions[0, trajectory[0]]
        for t in range(1, horizon):
            weight += log_trans[trajectory[t - 1], trajectory[t]] + emissions[t, trajectory[t]]
        totals[trajectory[-1]] = np.logaddexp(totals[trajectory[-1]], weight)
    totals -= totals.max()
    weights = np.exp(totals)
    return weights / weights.sum()


def oracle_batch(observations: np.ndarray, states: np.ndarray, trans: np.ndarray, lam: float,
                 chunk: int = 512) -> np.ndarray:
    """Batched exact forward posterior over the final latent state.

    observations: (N, T, M); returns (N, S). The transition matrix is passed in so it is
    built once per configuration instead of once per episode.
    """
    count, horizon, _ = observations.shape
    size = len(states)
    log_trans = np.log(trans)
    output = np.empty((count, size), dtype=np.float64)
    for start in range(0, count, chunk):
        block = observations[start:start + chunk]
        logp = np.full((len(block), size), -math.log(size))
        for t in range(horizon):
            residual = block[:, t, None, :] - lam * states[None, :, :]
            logp = logp - 0.5 * (residual ** 2).sum(axis=2)
            logp -= logp.max(axis=1, keepdims=True)
            if t < horizon - 1:
                logp = np.log(np.exp(logp) @ trans)
                logp -= logp.max(axis=1, keepdims=True)
        weights = np.exp(logp)
        output[start:start + chunk] = weights / weights.sum(axis=1, keepdims=True)
    return output


def sample_episodes(rng: np.random.Generator, states: np.ndarray, edges, lam: float, horizon: int,
                    count: int, trans: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised episode generation plus batched oracle posteriors."""
    if trans is None:
        trans = transition_matrix(states, edges)
    cumulative = trans.cumsum(axis=1)
    index = rng.integers(0, len(states), size=count)
    observations = np.empty((count, horizon, states.shape[1]), dtype=np.float32)
    for t in range(horizon):
        observations[:, t, :] = lam * states[index] + rng.normal(size=(count, states.shape[1]))
        if t < horizon - 1:
            draws = rng.random(size=count)
            index = (cumulative[index] > draws[:, None]).argmax(axis=1)
    posteriors = oracle_batch(observations.astype(np.float64), states, trans, lam)
    return observations, posteriors.astype(np.float32)


class DenseControl(nn.Module):
    """Dense single-stream GRU control: one stream, matched hidden state H=64."""

    def __init__(self, m: int, hidden: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=m, hidden_size=hidden, batch_first=True)
        self.readout = nn.Linear(hidden, 2 ** m)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(observations)
        return self.readout(output[:, -1, :])


def per_episode_kl(target: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """KL(target || softmax(logits)) in nats, per episode (final-step posterior)."""
    logq = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    p = np.clip(target, 1e-12, None)
    return (p * (np.log(p) - logq)).sum(axis=1)


def evaluate(model: nn.Module, observations: np.ndarray, posteriors: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(observations)).numpy()
    values = per_episode_kl(posteriors, logits)
    return {
        "mean_kl_nats": float(values.mean()),
        "mc_se_nats": float(values.std(ddof=1) / math.sqrt(len(values))),
        "episodes": int(len(values)),
    }


_DATA_CACHE: dict[tuple[str, int], tuple] = {}


def prepare_data(name: str, seed: int):
    key = (name, seed)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    spec = CONFIGS[name]
    m, horizon, lam = spec["M"], spec["T"], spec["lam"]
    states = state_table(m)
    edges = signed_graph(m, spec["graph_seed"], spec["frustrated"])
    trans = transition_matrix(states, edges)
    rng = np.random.default_rng(int(np.random.SeedSequence([20260924, spec["graph_seed"], seed]).generate_state(1)[0]))
    train = sample_episodes(rng, states, edges, lam, horizon, 4096, trans)
    validation = sample_episodes(rng, states, edges, lam, horizon, VALIDATION_EPISODES, trans)
    evaluation = sample_episodes(rng, states, edges, lam, horizon, EVAL_EPISODES, trans)
    # posterior entropy measured on the SAME evaluation stream as the untrained KL
    eval_post = evaluation[1].astype(np.float64)
    entropy = float(np.mean([-(row * np.log(np.clip(row, 1e-12, None))).sum() for row in eval_post]))
    _DATA_CACHE[key] = (states, edges, train, validation, evaluation, entropy)
    return _DATA_CACHE[key]


def run_cell(name: str, seed: int, lr: float, updates: int, batch: int) -> dict:
    spec = CONFIGS[name]
    m, horizon, lam = spec["M"], spec["T"], spec["lam"]
    states, edges, train, validation, evaluation, posterior_entropy = prepare_data(name, seed)
    train_obs, train_post = train
    val_obs, val_post = validation
    eval_obs, eval_post = evaluation

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = DenseControl(m)
    curve = {0: evaluate(model, eval_obs, eval_post)["mean_kl_nats"]}
    untrained = curve[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    obs_t, post_t = torch.from_numpy(train_obs), torch.from_numpy(train_post)
    generator = torch.Generator().manual_seed(seed + 991)
    model.train()
    start = time.time()
    for step in range(1, updates + 1):
        index = torch.randint(0, len(obs_t), (batch,), generator=generator)
        optimizer.zero_grad()
        loss = loss_fn(model(obs_t[index]), post_t[index])
        loss.backward()
        optimizer.step()
        if step in KL_GRID:
            curve[step] = evaluate(model, eval_obs, eval_post)["mean_kl_nats"]
    elapsed = time.time() - start

    held_out = evaluate(model, eval_obs, eval_post)
    val_result = evaluate(model, val_obs, val_post)
    return {
        "config": name, "role": spec["role"], "M": m, "T": horizon, "lambda": lam,
        "frustrated": spec["frustrated"], "seed": seed, "lr": lr, "updates": updates,
        "untrained_kl_nats": untrained,
        "terminal_kl_nats": held_out["mean_kl_nats"],
        "terminal_mc_se_nats": held_out["mc_se_nats"],
        "validation_kl_nats": val_result["mean_kl_nats"],
        "learned_nats": untrained - held_out["mean_kl_nats"],
        "learned_fraction": (untrained - held_out["mean_kl_nats"]) / untrained,
        "kl_curve": {str(k): v for k, v in sorted(curve.items())},
        "checkpoints": {str(u): curve.get(u, float("nan")) for u in (0, 250, 500, 1000, 2000, 4000)},
        "plateau_gaps": {
            "250_to_500": curve.get(250, float("nan")) - curve.get(500, float("nan")),
            "500_to_1000": curve.get(500, float("nan")) - curve.get(1000, float("nan")),
            "1000_to_2000": curve.get(1000, float("nan")) - curve.get(2000, float("nan")),
            "2000_to_4000": curve.get(2000, float("nan")) - curve.get(4000, held_out["mean_kl_nats"]),
        },
        "tail_flat": bool(all(abs(curve.get(a, float("nan")) - curve.get(b, float("nan"))) <= TAU_PLATEAU
                              for a, b in ((500, 1000), (1000, 2000), (2000, 4000)))),
        "no_step_rise": bool(all(curve.get(b, float("nan")) <= curve.get(a, float("nan")) + TAU_PLATEAU
                                 for a, b in ((250, 500), (500, 1000), (1000, 2000), (2000, 4000)))),
        "no_net_rise": bool(curve.get(4000, held_out["mean_kl_nats"]) <= curve.get(250, float("nan"))),
        "train_seconds": round(elapsed, 2),
        "posterior_entropy_nats": posterior_entropy,
        "edges": [[i, j, s] for i, j, s in edges],
    }


_ENTROPY_CACHE: dict[tuple[str, int], float] = {}


def oracle_entropy(name: str, seed: int) -> float:
    """Mean posterior entropy (nats) of the held-out stream for one configuration."""
    key = (name, seed)
    if key in _ENTROPY_CACHE:
        return _ENTROPY_CACHE[key]
    spec = CONFIGS[name]
    states = state_table(spec["M"])
    edges = signed_graph(spec["M"], spec["graph_seed"], spec["frustrated"])
    trans = transition_matrix(states, edges)
    rng = np.random.default_rng(int(np.random.SeedSequence([20260924, spec["graph_seed"], seed]).generate_state(1)[0]))
    _, posteriors = sample_episodes(rng, states, edges, spec["lam"], spec["T"], EVAL_EPISODES, trans)
    entropy = float(np.mean([-(p * np.log(np.clip(p, 1e-12, None))).sum() for p in posteriors]))
    _ENTROPY_CACHE[key] = entropy
    return entropy


def model_definition_checks() -> dict:
    """Independent checks of the registered model definition itself (transition, emission, graph)."""
    out = {}
    for name, spec in CONFIGS.items():
        m = spec["M"]
        states = state_table(m)
        edges = signed_graph(m, spec["graph_seed"], spec["frustrated"])
        trans = transition_matrix(states, edges)
        # (i) rows are distributions
        row_sum_residual = float(np.abs(trans.sum(axis=1) - 1.0).max())
        # (ii) J=0 kernel must equal the product of per-dimension two-state kernels
        flat = transition_matrix(states, tuple((i, j, 0.0) for i, j, _ in edges))
        pair = np.array([[-1.0], [1.0]])
        kernel = np.exp(BETA * (pair @ pair.T))
        kernel /= kernel.sum(axis=1, keepdims=True)
        product = np.ones((len(states), len(states)))
        for i in range(m):
            row_index = (states[:, i] > 0).astype(int)
            col_index = (states[:, i] > 0).astype(int)
            product *= kernel[row_index][:, col_index]
        factorisation_residual = float(np.abs(flat - product).max())
        # (iii) energy against a brute-force edge loop
        energy_vector = energy(states, edges)
        brute = np.array([sum(sign * state[i] * state[j] for i, j, sign in edges) for state in states])
        energy_residual = float(np.abs(energy_vector - brute).max())
        # (iv) emission against the expanded quadratic form
        rng = np.random.default_rng(spec["graph_seed"] + 5)
        stream = spec["lam"] * states[rng.integers(0, len(states))] + rng.normal(size=(4, m))
        direct = emission_loglik(stream, states, spec["lam"])
        expanded = np.array([[-0.5 * (np.sum(y ** 2) - 2 * spec["lam"] * float(y @ s)
                                      + spec["lam"] ** 2 * float(s @ s)) for s in states] for y in stream])
        emission_residual = float(np.abs(direct - expanded).max())
        # (v) state indexing rebuilt from the documented convention:
        #     row i, dimension b -> +1 iff bit (m-1-b) of i is set, else -1
        decoded = np.array([[1.0 if (i >> (m - 1 - b)) & 1 else -1.0 for b in range(m)]
                            for i in range(len(states))])
        indexing_residual = float(np.abs(decoded - states).max())
        # (vi) graph rebuilt from the documented construction rule, written independently
        rebuild_rng = np.random.default_rng(spec["graph_seed"])
        order = list(range(m))
        rebuild_rng.shuffle(order)
        rebuilt: dict[tuple[int, int], float] = {}
        for i in range(m):
            a, b = sorted((order[i], order[(i + 1) % m]))
            rebuilt[(a, b)] = 1.0
        if spec["frustrated"] and m >= 4:
            for i in range(0, m, 2):
                a, b = sorted((order[i], order[(i + 2) % m]))
                if a != b:
                    rebuilt[(a, b)] = -1.0
        rebuilt_edges = tuple((a, b, s) for (a, b), s in sorted(rebuilt.items()))
        graph_residual = float(max(abs(s1 - s2) for (_, _, s1), (_, _, s2) in zip(edges, rebuilt_edges))) \
            if len(edges) == len(rebuilt_edges) else float("inf")
        graph_edge_count_match = len(edges) == len(rebuilt_edges)
        # (vii) emission identity: states differing in one dimension must differ in
        #      log-likelihood by exactly 2 * lambda * y_b * s_b
        y = stream[0]
        base = states[0]
        ratios = []
        for b in range(m):
            other = base.copy()
            other[b] = -other[b]
            idx_other = int(np.where((states == other).all(axis=1))[0][0])
            direct_gap = emission_loglik(stream[:1], states, spec["lam"])[0, 0] - \
                emission_loglik(stream[:1], states, spec["lam"])[0, idx_other]
            ratios.append(abs(direct_gap - 2.0 * spec["lam"] * y[b] * base[b]))
        emission_identity_residual = float(max(ratios))
        out[name] = {
            "state_indexing_residual": indexing_residual,
            "graph_rebuild_edge_count_match": graph_edge_count_match,
            "graph_sign_residual": graph_residual,
            "emission_ratio_identity_residual": emission_identity_residual,
            "transition_row_sum_residual": row_sum_residual,
            "J0_factorisation_residual": factorisation_residual,
            "energy_vs_bruteforce_residual": energy_residual,
            "emission_vs_expanded_residual": emission_residual,
            "shared_construction_note": "the enumeration oracle reuses transition_matrix and "
                                        "emission_loglik, so it cannot detect a shared model-definition error; "
                                        "these four structural checks are the independent path for that",
        }
    return out


def oracle_checks() -> dict:
    """Includes per-timestep agreement, soft-target normalisation and a two-route KL check."""
    """Production-horizon independent oracle check plus short-horizon enumeration."""
    out = {}
    for name, spec in CONFIGS.items():
        m, horizon = spec["M"], spec["T"]
        states = state_table(m)
        edges = signed_graph(m, spec["graph_seed"], spec["frustrated"])
        rng = np.random.default_rng(spec["graph_seed"] + 17)
        stream = spec["lam"] * states[rng.integers(0, len(states))] + rng.normal(size=(horizon, m))
        log_space = oracle_log_space(stream, states, edges, spec["lam"])
        scaled = oracle_scaled_probability(stream, states, edges, spec["lam"])
        short = min(horizon, 3 if m <= 4 else 2)
        enumerated = oracle_by_enumeration(stream[:short], states, edges, spec["lam"])
        log_short = oracle_log_space(stream[:short], states, edges, spec["lam"])
        # per-timestep agreement: each implementation runs its own recursion and reports
        # its posterior at every step (no mixing of log-space and linear-space arithmetic).
        def sequence_log_space() -> np.ndarray:
            logp = np.full(len(states), -math.log(len(states)))
            log_trans = np.log(transition_matrix(states, edges))
            emissions = emission_loglik(stream, states, spec["lam"])
            seq = []
            for t in range(horizon):
                logp = logp + emissions[t]
                logp -= logp.max()
                weights = np.exp(logp)
                seq.append(weights / weights.sum())
                if t < horizon - 1:
                    logp = np.log(np.exp(logp) @ np.exp(log_trans))
                    logp -= logp.max()
            return np.array(seq)

        def sequence_scaled() -> np.ndarray:
            trans = transition_matrix(states, edges)
            probs = np.full(len(states), 1.0 / len(states))
            emissions = emission_loglik(stream, states, spec["lam"])
            seq = []
            for t in range(horizon):
                probs = probs * np.exp(emissions[t] - emissions[t].max())
                probs /= probs.sum()
                seq.append(probs.copy())
                if t < horizon - 1:
                    probs = probs @ trans
                    probs /= probs.sum()
            return np.array(seq)

        seq_log, seq_scaled = sequence_log_space(), sequence_scaled()
        per_step = [float(np.abs(seq_log[t] - seq_scaled[t]).max()) for t in range(horizon)]
        reference = np.exp(rng.normal(size=len(states)))
        reference /= reference.sum()
        kl_from_log = float((reference * (np.log(reference) - np.log(np.clip(log_space, 1e-300, None)))).sum())
        kl_from_scaled = float((reference * (np.log(reference) - np.log(np.clip(scaled, 1e-300, None)))).sum())
        out[name] = {
            "production_horizon": horizon,
            "per_timestep_max_abs_diff": per_step,
            "per_timestep_max": float(max(per_step)),
            "soft_target_normalisation_residual": float(abs(log_space.sum() - 1.0)),
            "kl_two_routes_abs_diff": float(abs(kl_from_log - kl_from_scaled)),
            "max_abs_diff_log_vs_scaled": float(np.abs(log_space - scaled).max()),
            "normalisation_log_space": float(log_space.sum()),
            "normalisation_scaled": float(scaled.sum()),
            "enumeration_horizon": short,
            "max_abs_diff_log_vs_enumeration": float(np.abs(log_short - enumerated).max()),
            "trajectories_enumerated": int(len(states) ** short),
            "posterior_entropy_nats": float(-(log_space * np.log(np.clip(log_space, 1e-300, None))).sum()),
        }
    return out


def aggregate(rows: list[dict]) -> dict:
    """Per-seed gates, LR-insensitivity, then the registered branch order."""
    verdicts = {}
    for name in CONFIGS:
        cells = [r for r in rows if r["config"] == name]
        if not cells:
            continue
        by_seed: dict[int, dict[float, dict]] = {}
        for r in cells:
            by_seed.setdefault(r["seed"], {})[r["lr"]] = r
        seeds = sorted(by_seed)
        best = {s: min(by_seed[s].values(), key=lambda r: r["validation_kl_nats"]) for s in seeds}

        terminal = np.array([best[s]["terminal_kl_nats"] for s in seeds])
        untrained = np.array([best[s]["untrained_kl_nats"] for s in seeds])
        half = 1.96 * terminal.std(ddof=1) / math.sqrt(len(terminal)) if len(terminal) > 1 else float("nan")
        lower, upper = terminal.mean() - half, terminal.mean() + half

        # scale guard: an untrained near-uniform readout cannot exceed ln(2**M) by much;
        # the entropy identity is reported as a diagnostic, not gated (its residual also
        # reflects the spread of the random readout, not a defect).
        ceiling = math.log(2 ** CONFIGS[name]["M"]) + UNTRAINED_CEILING_SLACK
        ideal_untrained = np.array([math.log(2 ** CONFIGS[name]["M"]) - best[s]["posterior_entropy_nats"]
                                    for s in seeds])
        untrained_scale_residual = float(np.abs(untrained - ideal_untrained).max())
        well_conditioned = bool(np.all(untrained >= UNTRAINED_FLOOR) and np.all(untrained <= ceiling))
        residual_fraction = {s: best[s]["terminal_kl_nats"] / best[s]["untrained_kl_nats"] for s in seeds}
        learned_fraction = {s: best[s]["learned_fraction"] for s in seeds}
        lr_residuals = {
            s: {str(lr): by_seed[s][lr]["terminal_kl_nats"] / by_seed[s][lr]["untrained_kl_nats"]
                for lr in sorted(by_seed[s])} for s in seeds
        }
        plateau_ok = {
            s: bool(best[s]["tail_flat"] and best[s]["no_step_rise"] and best[s]["no_net_rise"]) for s in seeds
        }
        gates = {
            "well_conditioned": well_conditioned,
            "absolute_residual_ci_lower": bool(lower > EPSILON_ABS),
            "relative_residual_all_seeds": bool(all(residual_fraction[s] > RHO_REL for s in seeds)),
            "learned_enough_all_seeds": bool(all(learned_fraction[s] >= KAPPA_LEARNED for s in seeds)),
            "lr_cross_agreement": bool(all(
                by_seed[s][other]["terminal_kl_nats"] > EPSILON_ABS
                and (abs(by_seed[s][best[s]["lr"]]["terminal_kl_nats"]
                         - by_seed[s][other]["terminal_kl_nats"]) <= LR_TOL
                     or by_seed[s][other]["learned_fraction"] >= KAPPA_LEARNED)
                for s in seeds for other in by_seed[s] if other != best[s]["lr"])),
            "plateau_all_seeds": bool(all(plateau_ok.values())),
        }
        if not well_conditioned:
            label, reason = "INCONCLUSIVE", "ILL_CONDITIONED_UNTRAINED_SCALE"
        elif not gates["learned_enough_all_seeds"]:
            label, reason = "LEARNING_FAILURE", "LEARNED_FRACTION_BELOW_KAPPA"
        elif upper <= EPSILON_ABS:
            label, reason = "SATURATED", "TERMINAL_KL_CI_UPPER_AT_OR_BELOW_EPSILON"
        elif all(gates.values()):
            label, reason = "HEADROOM", "ALL_GATES_PASSED"
        elif not gates["plateau_all_seeds"]:
            label, reason = "INCONCLUSIVE", "PLATEAU_GATE_FAILED"
        else:
            label, reason = "INCONCLUSIVE", "OTHER_GATE_FAILED"

        verdicts[name] = {
            "role": CONFIGS[name]["role"],
            "selected_lr_per_seed": {str(s): best[s]["lr"] for s in seeds},
            "terminal_kl_mean": float(terminal.mean()), "terminal_kl_ci95": [float(lower), float(upper)],
            "terminal_kl_minmax": [float(terminal.min()), float(terminal.max())],
            "untrained_kl_mean": float(untrained.mean()),
            "mean_episode_mc_se": float(np.mean([best[s]["terminal_mc_se_nats"] for s in seeds])),
            "residual_fraction_per_seed": {str(s): residual_fraction[s] for s in seeds},
            "learned_fraction_per_seed": {str(s): learned_fraction[s] for s in seeds},
            "residual_fraction_both_lrs": {str(s): lr_residuals[s] for s in seeds},
            "plateau_gaps_per_seed": {str(s): best[s]["plateau_gaps"] for s in seeds},
            "checkpoints_selected_lr": {str(s): best[s]["kl_curve"] for s in seeds},
            "terminal_kl_both_lrs": {str(s): {str(lr): by_seed[s][lr]["terminal_kl_nats"] for lr in sorted(by_seed[s])} for s in seeds},
            "plateau_ok_per_seed": {str(s): plateau_ok[s] for s in seeds},
            "untrained_scale_residual_diagnostic": untrained_scale_residual,
            "untrained_ceiling_nats": ceiling,
            "gates": gates, "verdict": label, "reason": reason,
        }
    # C1 budget-adequacy anchor: if the easiest same-family configuration is not driven to
    # saturation at this budget, the budget is not demonstrated adequate and no HEADROOM
    # verdict may be issued.
    budget_adequate = verdicts.get("C1", {}).get("verdict") == "SATURATED"
    if not budget_adequate:
        for name in ANSWER_ORDER:
            if name in verdicts:
                verdicts[name]["verdict"] = "INCONCLUSIVE"
                verdicts[name]["reason"] = "BUDGET_ADEQUACY_ANCHOR_FAILED"
                verdicts[name]["downgraded_by"] = "budget_adequacy_anchor_failed"
    answer = next((n for n in ANSWER_ORDER if verdicts.get(n, {}).get("verdict") == "HEADROOM"), None)
    if answer is None:
        tested = [verdicts[n]["verdict"] for n in ANSWER_ORDER if n in verdicts]
        answer = "CLOSE_CONFIGS" if tested and all(v in ("SATURATED", "LEARNING_FAILURE") for v in tested) else "NO_CONFIG_PASSED"
    return {"per_config": verdicts, "answer": answer, "budget_adequacy_anchor": budget_adequate,
            "c1_verdict": verdicts.get("C1", {}).get("verdict")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "oracle", "screen"), default="oracle")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    args = parser.parse_args()

    if args.mode == "oracle":
        result = {"mode": "oracle", "checks": oracle_checks(), "model_definition": model_definition_checks()}
    else:
        updates = 200 if args.mode == "pilot" else args.updates
        rows = []
        for name in [n for n in args.configs.split(",") if n]:
            for seed in SEEDS:
                for lr in LRS:
                    row = run_cell(name, seed, lr, updates, BATCH)
                    rows.append(row)
                    print(json.dumps({k: row[k] for k in ("config", "seed", "lr", "untrained_kl_nats",
                                                          "terminal_kl_nats", "learned_fraction",
                                                          "plateau_gaps", "train_seconds")},
                                     ensure_ascii=False), flush=True)
        if args.mode == "pilot":
            result = {"mode": "pilot", "updates": updates, "rows": rows,
                      "aggregate": "NOT_APPLICABLE_ENGINEERING_PILOT"}
        else:
            result = {"mode": "screen", "updates": updates, "rows": rows, "aggregate": aggregate(rows)}

    payload = {
        "schema": "cal0008_terminal_headroom_screen_v2",
        "design": "research/design/cal0008_terminal_headroom_predesign.md",
        "constants": {"beta": BETA, "coupling": COUPLING, "seeds": list(SEEDS), "lrs": list(LRS),
                      "batch": BATCH, "updates": UPDATES, "eval_episodes": EVAL_EPISODES,
                      "validation_episodes": VALIDATION_EPISODES, "kl_grid": list(KL_GRID),
                      "epsilon_abs_nats": EPSILON_ABS, "rho_rel": RHO_REL,
                      "kappa_learned": KAPPA_LEARNED, "tau_plateau_nats": TAU_PLATEAU,
                      "answer_order": list(ANSWER_ORDER)},
        "configs": CONFIGS,
        "result": result,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(json.dumps({"output": str(args.output),
                          "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()[:16]}, ensure_ascii=False))
    if args.mode == "oracle":
        print(json.dumps(result["checks"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
