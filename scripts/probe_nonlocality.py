"""Model-free non-locality probe for candidate task generators (design-support tool).

NOT a registered run and NOT a verdict. For a generator with an exact oracle it measures

    gap(k) = KL( p(z_T | y_1:T)  ||  p(z_T | y_{T-k+1:T}) )

i.e. how much the *distant* past changes the current latent posterior, where the truncated
posterior is the exact filter restarted from the stationary prior on the last k steps. A gap
that decays slowly in k means no bounded-memory model can reproduce the posterior; a gap
that vanishes by small k means the task is effectively local, which is why a 64-unit
recurrent control can compress it.

Modes
-----
  single    : one Ising-coupled Markov regime (the CAL-0008 family)
  mixture   : K discrete regimes, drawn once per episode, unknown to the filter
  theta     : a continuous coupling strength on a fine grid, drawn once per episode

Usage: python scripts/probe_nonlocality.py --mode single|mixture|theta
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

T = 32
M = 6
LAM = 0.7
BETA = 0.8
EPISODES = 200
SEED = 20260924
WINDOWS = (1, 2, 4, 8, 16, 32)


def states_table() -> np.ndarray:
    return np.array(list(itertools.product((-1.0, 1.0), repeat=M)))


def ring_edges() -> tuple[tuple[int, int], ...]:
    return tuple((i, (i + 1) % M) for i in range(M))


def energy(states: np.ndarray, edges, signs) -> np.ndarray:
    return sum(s * states[:, i] * states[:, j] for (i, j), s in zip(edges, signs))


def regimes_for(mode: str) -> list[dict]:
    edges = ring_edges()
    if mode == "single":
        return [{"name": "J0.25_ferro", "signs": tuple([1.0] * len(edges)), "J": 0.25}]
    if mode == "mixture":
        return [
            {"name": "J0.00_ferro", "signs": tuple([1.0] * len(edges)), "J": 0.0},
            {"name": "J0.25_ferro", "signs": tuple([1.0] * len(edges)), "J": 0.25},
            {"name": "J0.50_ferro", "signs": tuple([1.0] * len(edges)), "J": 0.50},
            {"name": "J0.25_frustrated", "signs": tuple([(-1.0) ** i for i in range(len(edges))]), "J": 0.25},
        ]
    if mode == "theta":
        return [{"name": f"theta_{j:.4f}", "signs": tuple([1.0] * len(edges)), "J": float(j)}
                for j in np.linspace(0.0, 0.60, 32)]
    raise ValueError(mode)


def transition(states: np.ndarray, edges, regime: dict) -> np.ndarray:
    logits = BETA * (states @ states.T) + regime["J"] * energy(states, edges, regime["signs"])[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    matrix = np.exp(logits)
    return matrix / matrix.sum(axis=1, keepdims=True)


def stationary(regime: dict, states: np.ndarray, edges) -> np.ndarray:
    dist = np.full(len(states), 1.0 / len(states))
    matrix = transition(states, edges, regime)
    for _ in range(2000):
        dist = dist @ matrix
    return dist


def exact_filter(observations: np.ndarray, states: np.ndarray, trans: np.ndarray,
                 prior: np.ndarray, lam: float = LAM) -> np.ndarray:
    logp = np.log(np.clip(prior, 1e-300, None))
    horizon = len(observations)
    for t in range(horizon):
        residual = observations[t][None, :] - lam * states
        logp = logp - 0.5 * (residual ** 2).sum(axis=1)
        logp -= logp.max()
        if t < horizon - 1:
            logp = np.log(np.exp(logp) @ trans)
            logp -= logp.max()
    weights = np.exp(logp)
    return weights / weights.sum()


def joint_filter(observations: np.ndarray, states: np.ndarray, trans_stack: list[np.ndarray],
                 prior: np.ndarray, lam: float = LAM) -> np.ndarray:
    """Exact filter over the augmented (regime, latent) state; returns the latent marginal."""
    size = len(states)
    logp = np.log(np.clip(prior, 1e-300, None)).reshape(len(trans_stack), size)
    horizon = len(observations)
    for t in range(horizon):
        residual = observations[t][None, :] - lam * states
        logp = logp - 0.5 * (residual ** 2).sum(axis=1)[None, :]
        logp -= logp.max()
        if t < horizon - 1:
            for k, trans in enumerate(trans_stack):
                logp[k] = np.log(np.exp(logp[k]) @ trans)
            logp -= logp.max()
    weights = np.exp(logp)
    weights /= weights.sum()
    return weights.sum(axis=0)


def episode(rng: np.random.Generator, states: np.ndarray, lam: float,
            trans: np.ndarray, cumulative: np.ndarray) -> np.ndarray:
    index = int(rng.integers(0, len(states)))
    stream = np.empty((T, M))
    for t in range(T):
        stream[t] = lam * states[index] + rng.normal(size=M)
        if t < T - 1:
            index = int(np.searchsorted(cumulative[index], rng.random()))
    return stream


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    return float((p * (np.log(p) - np.log(q))).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "mixture", "theta"), required=True)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    states = states_table()
    edges = ring_edges()
    regimes = regimes_for(args.mode)
    trans_stack = [transition(states, edges, r) for r in regimes]
    cum_stack = [t.cumsum(axis=1) for t in trans_stack]
    stat_stack = [stationary(r, states, edges) for r in regimes]
    joint_prior = np.concatenate([s / len(regimes) for s in stat_stack])
    gaps = {k: [] for k in WINDOWS}
    entropies = []

    for _ in range(args.episodes):
        which = 0 if args.mode == "single" else int(rng.integers(0, len(regimes)))
        stream = episode(rng, states, LAM, trans_stack[which], cum_stack[which])
        if args.mode == "single":
            full = exact_filter(stream, states, trans_stack[which], stat_stack[which])
        else:
            full = joint_filter(stream, states, trans_stack, joint_prior)
        entropies.append(float(-(np.clip(full, 1e-12, None) * np.log(np.clip(full, 1e-12, None))).sum()))
        for k in WINDOWS:
            if args.mode == "single":
                truncated = exact_filter(stream[T - k:], states, trans_stack[which], stat_stack[which])
            else:
                truncated = joint_filter(stream[T - k:], states, trans_stack, joint_prior)
            gaps[k].append(kl(full, truncated))

    result = {
        "schema": "nonlocality_probe_v1",
        "status": "engineering premise check only; not a registered run, not a verdict",
        "mode": args.mode,
        "constants": {"T": T, "M": M, "lambda": LAM, "beta": BETA, "episodes": args.episodes,
                      "n_regimes": len(regimes), "seed": SEED},
        "mean_posterior_entropy_nats": float(np.mean(entropies)),
        "truncation_gap_nats": {str(k): float(np.mean(gaps[k])) for k in WINDOWS},
        "reading": "gap(k) = KL(full-history posterior || posterior from the last k steps only). "
                   "Slow decay in k means the posterior depends on distant history, so no "
                   "bounded-memory model can reproduce it; decay to zero by small k means the "
                   "task is effectively local and a fixed-state control can compress it.",
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(f"mode={args.mode}  entropy={result['mean_posterior_entropy_nats']:.3f} nats")
    print(f"{'k':>4} {'gap(nats)':>10}")
    for k in WINDOWS:
        print(f"{k:>4} {result['truncation_gap_nats'][str(k)]:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
