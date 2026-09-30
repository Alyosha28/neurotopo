"""Premise probe for a structurally different task family (semi-Markov latent).

NOT a registered run and NOT a verdict: this is a read-only premise check for a design
proposal. It compares how well the exact posterior of two generators can be predicted
from a *fixed-dimensional summary* of the observation history:

  M-generator : the CAL-0008 Markov latent (2 latents, persistence + coupling)
  S-generator : a semi-Markov latent whose switch probability depends on the current
                dwell time (Weibull hazard), so the filter must track run lengths

If the S posterior needs far more summary dimension for the same residual, the design
premise ("this task is not low-dimensional-compressible") has support. A linear probe
under-estimates what a trained recurrent control can do, so a positive result here is
necessary-but-not-sufficient and must still be screened by a registered screen.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

T = 32
M = 2
LAM = 0.7
D_MAX = 32
D0 = 8.0
KAPPA = 2.0
BETA = 0.8
COUPLING = 0.25
EPISODES = 3000
TRAIN = 2000
SEED = 20260924
OUT = Path("research/design/structural_task_probe/semimarkov_premise_probe.json")


# ---------------------------------------------------------------- M-generator (Markov)
def markov_states() -> np.ndarray:
    return np.array(list(itertools.product((-1.0, 1.0), repeat=M)))


def markov_transition(states: np.ndarray) -> np.ndarray:
    energy = COUPLING * (states[:, 0] * states[:, 1])[None, :]
    logits = BETA * (states @ states.T) + energy
    logits -= logits.max(axis=1, keepdims=True)
    m = np.exp(logits)
    return m / m.sum(axis=1, keepdims=True)


def emission_loglik(obs: np.ndarray, states: np.ndarray, lam: float = LAM) -> np.ndarray:
    residual = obs[:, None, :] - lam * states[None, :, :]
    return -0.5 * (residual ** 2).sum(axis=2)


def markov_episodes(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    states = markov_states()
    trans = markov_transition(states)
    cum = trans.cumsum(axis=1)
    obs = np.empty((count, T, M))
    post = np.empty((count, len(states)))
    log_trans = np.log(trans)
    for e in range(count):
        idx = int(rng.integers(0, len(states)))
        stream = np.empty((T, M))
        logp = np.full(len(states), -math.log(len(states)))
        for t in range(T):
            stream[t] = LAM * states[idx] + rng.normal(size=M)
            logp = logp - 0.5 * ((stream[t][None, :] - LAM * states) ** 2).sum(axis=1)
            logp -= logp.max()
            if t < T - 1:
                logp = np.log(np.exp(logp) @ trans)
                logp -= logp.max()
                idx = int(np.searchsorted(cum[idx], rng.random()))
        w = np.exp(logp)
        obs[e] = stream
        post[e] = w / w.sum()
    return obs, post


# ------------------------------------------------------------ S-generator (semi-Markov)
def hazard(dwell: np.ndarray) -> np.ndarray:
    """Per-step switch probability as a function of current dwell (increasing hazard)."""
    return 1.0 - np.exp(-((dwell / D0) ** KAPPA))


def per_dim_kernel() -> tuple[np.ndarray, list[tuple[float, int]]]:
    """Kernel over (z, dwell) pairs for ONE latent dimension, plus the state list."""
    pairs = [(z, d) for z in (-1.0, 1.0) for d in range(1, D_MAX + 1)]
    index = {p: i for i, p in enumerate(pairs)}
    kernel = np.zeros((len(pairs), len(pairs)))
    for (z, d), i in index.items():
        h = float(hazard(np.array(d)))
        kernel[i, index[(z, min(d + 1, D_MAX))]] += 1.0 - h
        kernel[i, index[(-z, 1)]] += h
    return kernel, pairs


def semi_markov_episodes(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact filter over the augmented state (z1, d1, z2, d2); output marginal over (z1, z2)."""
    kernel, pairs = per_dim_kernel()
    size = len(pairs)                      # 2 * D_MAX = 64
    dwell_values = np.array([p[1] for p in pairs], dtype=float)
    hazard_vec = hazard(dwell_values)      # per (z, d) row
    obs = np.empty((count, T, M))
    post = np.empty((count, 4))
    # state (z, d) index bookkeeping for sampling
    idx_of = {p: i for i, p in enumerate(pairs)}
    for e in range(count):
        # start: uniform z, dwell 1
        d1, d2 = 1, 1
        z1, z2 = (1.0 if rng.random() < 0.5 else -1.0), (1.0 if rng.random() < 0.5 else -1.0)
        dist = np.zeros((size, size))
        dist[idx_of[(z1, d1)], idx_of[(z2, d2)]] = 1.0
        stream = np.empty((T, M))
        for t in range(T):
            stream[t] = LAM * np.array([z1, z2]) + rng.normal(size=M)
            # emission update: multiply along each z axis by the per-dim likelihood
            for dim, (zaxis, value) in enumerate(((0, stream[t][0]), (1, stream[t][1]))):
                like = np.array([math.exp(-0.5 * (value - LAM * z) ** 2) for z, _ in pairs])
                dist = dist * (like if dim == 0 else like[:, None])
            total = dist.sum()
            if total <= 0 or not np.isfinite(total):
                dist = np.full((size, size), 1.0 / dist.size)
            else:
                dist = dist / total
            if t < T - 1:
                dist = kernel.T @ dist @ kernel      # joint transition factorises over dims
                dist /= dist.sum()
                # sample the next latent for the observation stream
                flat = dist.ravel()
                pick = int(rng.choice(flat.size, p=flat / flat.sum()))
                i1, i2 = divmod(pick, size)
                z1, d1 = pairs[i1]
                z2, d2 = pairs[i2]
        # marginalise dwell -> posterior over (z1, z2)
        marginal = dist.reshape(2, D_MAX, 2, D_MAX).sum(axis=(1, 3))
        obs[e] = stream
        post[e] = marginal.ravel() / marginal.sum()
    return obs, post


# ------------------------------------------------------------------------ linear probe
def summarise(obs: np.ndarray, kind: str, k: int) -> np.ndarray:
    """Fixed-dimensional summaries of the observation history."""
    if kind == "window":
        return obs[:, -k:, :].reshape(len(obs), -1)
    if kind == "evidence":
        # per-state accumulated Gaussian log-likelihood for the 4 latent combinations
        states = markov_states()
        total = np.zeros((len(obs), len(states)))
        for t in range(obs.shape[1]):
            total += emission_loglik(obs[:, t], states)
        return total
    if kind == "dwell_features":
        # window + a crude run-length proxy: number of sign changes in the last k steps
        window = obs[:, -k:, :]
        changes = (np.diff(np.sign(window), axis=1) != 0).sum(axis=1).astype(float)
        return np.concatenate([window.reshape(len(obs), -1), changes], axis=1)
    raise ValueError(kind)


def ridge_kl(features: np.ndarray, target: np.ndarray, train: int, alpha: float = 1e-3) -> float:
    x, y = features, target
    x_tr, y_tr, x_te, y_te = x[:train], y[:train], x[train:], y[train:]
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-9
    x_tr = (x_tr - mu) / sd
    x_te = (x_te - mu) / sd
    x_tr = np.concatenate([x_tr, np.ones((len(x_tr), 1))], axis=1)
    x_te = np.concatenate([x_te, np.ones((len(x_te), 1))], axis=1)
    gram = x_tr.T @ x_tr + alpha * np.eye(x_tr.shape[1])
    weights = np.linalg.solve(gram, x_tr.T @ y_tr)
    logits = x_te @ weights
    logq = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    p = np.clip(y_te, 1e-12, None)
    return float((p * (np.log(p) - logq)).sum(axis=1).mean())


def entropy(posteriors: np.ndarray) -> float:
    p = np.clip(posteriors, 1e-12, None)
    return float((-(p * np.log(p)).sum(axis=1)).mean())


def main() -> int:
    rng = np.random.default_rng(SEED)
    obs_m, post_m = markov_episodes(rng, EPISODES)
    obs_s, post_s = semi_markov_episodes(rng, EPISODES)

    result = {
        "schema": "semimarkov_premise_probe_v1",
        "status": "engineering premise check only; not a registered run, not a verdict",
        "constants": {"T": T, "M": M, "lambda": LAM, "d_max": D_MAX, "hazard_d0": D0,
                      "hazard_kappa": KAPPA, "beta": BETA, "coupling": COUPLING,
                      "episodes": EPISODES, "train": TRAIN, "seed": SEED},
        "oracle_entropy_nats": {"markov": entropy(post_m), "semi_markov": entropy(post_s)},
        "probe": {},
    }
    for name, (obs, post) in (("markov", (obs_m, post_m)), ("semi_markov", (obs_s, post_s))):
        rows = []
        for k in (1, 2, 4, 8, 16, 32):
            f = summarise(obs, "window", k)
            rows.append({"summary": f"window_k{k}", "dim": int(f.shape[1]), "kl_nats": ridge_kl(f, post, TRAIN)})
        f = summarise(obs, "evidence", 0)
        rows.append({"summary": "accumulated_evidence", "dim": int(f.shape[1]), "kl_nats": ridge_kl(f, post, TRAIN)})
        for k in (4, 8, 16):
            f = summarise(obs, "dwell_features", k)
            rows.append({"summary": f"window_k{k}_plus_sign_changes", "dim": int(f.shape[1]),
                         "kl_nats": ridge_kl(f, post, TRAIN)})
        result["probe"][name] = rows

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"oracle_entropy": result["oracle_entropy_nats"]}, ensure_ascii=False))
    for name in ("markov", "semi_markov"):
        print(f"--- {name} ---")
        for row in result["probe"][name]:
            print(f"   {row['summary']:<34} dim={row['dim']:>3}  KL={row['kl_nats']:.4f} nats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
