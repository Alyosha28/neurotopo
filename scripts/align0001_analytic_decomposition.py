"""G1-ALIGN-0001：把解析分解代价拆成「对齐/置换 × F/X」2×2 表。

目的（决定性前置检查）：主终点 `I = (S-U)_F - (S-U)_X` 可能有一部分来自**表示代价**
（块对角信念态能多好地逼近真后验），而不是来自**对齐**本身。若表示代价成分就能解释
整个 `I`，则本设计测的不是对齐，必须停止。

做法：对同一批 episode 的精确联合后验 `p(theta, z | y_1:T)`，计算它在
「给定 theta 的条件独立块分解」族下的最优投影 KL，**分别**用
  - 对齐划分（块 = 任务的维度分组）
  - 置换划分（冻结随机划分，组大小相同）
在 F 与 X 两个生成器上各算一遍，得到 2x2 表与其交互。

无训练、无模型；工程前提检查，不是判词。F/X 生成器与 oracle 逻辑沿用
`/tmp/gpi_v1_analytic.py`（该脚本此前不在仓库内，本次一并收入以关闭来源缺口）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import sys

import numpy as np

M = 6
LAM = 0.7
BETA = 0.8
T = 64
GRID = 32
SEED = 20_260_924
PERM_SEED = 20_260_924

F_EDGES = ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3))  # two disjoint 3-rings
X_EDGES = ((0, 3), (3, 1), (1, 4), (4, 2), (2, 5), (5, 0))  # 6-cycle across groups

ALIGNED_2 = ((0, 1, 2), (3, 4, 5))
# K=4 的「对齐」必须**在任务组内切分**：组 A={0,1,2}、组 B={3,4,5}。
# 连续切分 ((0,1),(2,3),(4,),(5,)) 的 (2,3) 跨组，**不是对齐划分**（本脚本初版的错误）。
ALIGNED_4 = ((0, 1), (2,), (3, 4), (5,))
# 跨组划分**按设计 §2 冻结**（不得由脚本临时抽取，否则常数与注册臂不符）
PERMUTED_2 = ((1, 2, 5), (0, 3, 4))
PERMUTED_4 = ((0, 5), (2, 3), (4,), (1,))
TASK_GROUPS = ((0, 1, 2), (3, 4, 5))


def is_aligned(partition) -> bool:
    """每个块都完全落在某一个任务组内。"""
    return all(any(set(b) <= set(g) for g in TASK_GROUPS) for b in partition)


def crosses_groups(partition) -> bool:
    """至少一个块跨任务组。"""
    return any(not any(set(b) <= set(g) for g in TASK_GROUPS) for b in partition)

STATES = np.array(list(itertools.product((-1.0, 1.0), repeat=M)))
JS = np.linspace(0.0, 0.6, GRID)
STEP = float(JS[1] - JS[0])


def energy(states: np.ndarray, edges) -> np.ndarray:
    return sum(states[:, i] * states[:, j] for i, j in edges)


def trans_for(J: float, edges) -> np.ndarray:
    lg = BETA * (STATES @ STATES.T) + J * energy(STATES, edges)[None, :]
    lg -= lg.max(axis=1, keepdims=True)
    m = np.exp(lg)
    return m / m.sum(axis=1, keepdims=True)


def stationary(T_: np.ndarray) -> np.ndarray:
    d = np.full(len(STATES), 1 / len(STATES))
    for _ in range(1500):
        d = d @ T_
    return d


def build(edges):
    Ts = [trans_for(J, edges) for J in JS]
    return np.log(np.stack(Ts)), [t.cumsum(axis=1) for t in Ts], np.stack([stationary(t) for t in Ts]) / GRID


def joint_filter(obs: np.ndarray, logT: np.ndarray, prior: np.ndarray) -> np.ndarray:
    logp = np.log(np.clip(prior, 1e-300, None)).copy()
    for t in range(len(obs)):
        r = obs[t][None, :] - LAM * STATES
        logp = logp - 0.5 * (r**2).sum(axis=1)[None, :]
        logp -= logp.max()
        if t < len(obs) - 1:
            logp = np.log(np.einsum("ks,ksj->kj", np.exp(logp), np.exp(logT)))
            logp -= logp.max()
    w = np.exp(logp)
    return w / w.sum()


def factored_kl(p: np.ndarray, partition) -> float:
    """KL(p || 最优「给定 theta 的条件独立块分解」投影)。

    族为 q(theta, z_1..z_K) = prod_b p(theta, z_b) / p(theta)^(K-1)。
    对 K=2 退化为 `p(theta,zA) p(theta,zB) / p(theta)`（与 /tmp 原实现一致）。
    """
    K = len(partition)
    p3 = p.reshape((GRID,) + (2,) * M)
    marg = []
    for block in partition:
        axes = tuple(1 + i for i in range(M) if i not in block)
        marg.append(p3.sum(axis=axes))
    pth = p3.sum(axis=tuple(range(1, M + 1)))
    q = np.ones_like(p3)
    for mb, block in zip(marg, partition):
        shape = [GRID] + [1] * M
        for i in sorted(block):
            shape[1 + i] = 2
        q = q * mb.reshape(shape)
    denom = np.clip(pth, 1e-300, None) ** (K - 1)
    q = q / denom.reshape((GRID,) + (1,) * M)
    q = q.reshape(-1)
    q = np.clip(q, 1e-300, None)
    q = q / q.sum()
    pp = np.clip(p.reshape(-1), 1e-300, None)
    return float((pp * (np.log(pp) - np.log(q))).sum())


def sample(edges, logT, cum, rng, n, T_=T):
    out = np.empty((n, T_, M))
    for e in range(n):
        k = int(rng.integers(0, GRID))
        idx = int(rng.integers(0, len(STATES)))
        for t in range(T_):
            out[e, t] = LAM * STATES[idx] + rng.normal(size=M)
            if t < T_ - 1:
                idx = int(np.searchsorted(cum[k][idx], rng.random()))
    return out


def frozen_partitions(k: int):
    """按设计 §2 冻结的 (aligned, permuted) 划分，不做随机抽取。"""
    if k == 2:
        return ALIGNED_2, PERMUTED_2
    return ALIGNED_4, PERMUTED_4


def gap_profile(logT, prior, obs, ks=(4, 8, 16, 32), boot=200, rng=None):
    full = np.array([joint_filter(o, logT, prior) for o in obs])
    full_t = full.reshape(len(obs), GRID, -1).sum(axis=2)
    full_t /= full_t.sum(axis=1, keepdims=True)
    res = {}
    for k in ks:
        vals = []
        for i, o in enumerate(obs):
            tr = joint_filter(o[T - k :], logT, prior)
            tr_t = tr.reshape(GRID, -1).sum(axis=1)
            tr_t /= tr_t.sum()
            p = np.clip(full_t[i], 1e-300, None)
            q = np.clip(tr_t, 1e-300, None)
            vals.append(float((p * (np.log(p) - np.log(q))).sum()))
        vals = np.array(vals)
        bs = (
            [float(np.mean(rng.choice(vals, len(vals), replace=True))) for _ in range(boot)]
            if rng is not None
            else []
        )
        res[str(k)] = {
            "mean": round(float(vals.mean()), 4),
            "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
            if bs
            else None,
        }
    return res, full_t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--projection-episodes", type=int, default=80)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(SEED)
    boot_rng = np.random.default_rng(SEED + 7)

    partitions = {
        "aligned_K2": ALIGNED_2,
        "permuted_K2": PERMUTED_2,
        "aligned_K4": ALIGNED_4,
        "permuted_K4": PERMUTED_4,
    }

    results = {}
    for name, edges in (("F", F_EDGES), ("X", X_EDGES)):
        logT, cum, prior = build(edges)
        obs = sample(edges, logT, cum, rng, args.episodes)
        gaps, theta_post = gap_profile(logT, prior, obs, rng=np.random.default_rng(SEED + 1))
        H = float(np.mean([-(p * np.log(np.clip(p, 1e-300, None))).sum() for p in theta_post]))
        cell = {
            "edges": [list(e) for e in edges],
            "theta_posterior_entropy_nats": round(H, 4),
            "uniform_baseline_kl_nats": round(math.log(GRID) - H, 4),
            "gap": gaps,
            "factored_projection_kl_nats": {},
        }
        # 精确联合后验只算一次，再对四种划分各投影一次（避免 4x 重复滤波）
        posters = [joint_filter(o, logT, prior) for o in obs[: args.projection_episodes]]
        per_ep = {}
        for pname, part in partitions.items():
            vals = np.array([factored_kl(p, part) for p in posters])
            per_ep[pname] = vals
            cell["factored_projection_kl_nats"][pname] = round(float(vals.mean()), 4)
            bs = np.array([float(np.mean(boot_rng.choice(vals, len(vals), replace=True))) for _ in range(400)])
            cell.setdefault("factored_projection_ci95", {})[pname] = [
                round(float(np.percentile(bs, 2.5)), 4),
                round(float(np.percentile(bs, 97.5)), 4),
            ]
        # 配对：同一 episode 上两个划分的差（方差远小于各自方差）
        cell["paired_extra_cost_per_episode"] = {
            kk: (per_ep[f"permuted_{kk}"] - per_ep[f"aligned_{kk}"]).tolist()
            for kk in ("K2", "K4")
        }
        results[name] = cell

    # 表示代价成分的交互：置换相对对齐的额外投影代价，在 F 与 X 上的差（配对 bootstrap）
    decomposition = {}
    for kk in ("K2", "K4"):
        dF = np.array(results["F"]["paired_extra_cost_per_episode"][kk])
        dX = np.array(results["X"]["paired_extra_cost_per_episode"][kk])
        comp = float(dF.mean() - dX.mean())
        # 配对 bootstrap：对每个条件各自重抽 episode，再取交互
        bs = np.array(
            [
                float(
                    boot_rng.choice(dF, len(dF), replace=True).mean()
                    - boot_rng.choice(dX, len(dX), replace=True).mean()
                )
                for _ in range(2000)
            ]
        )
        decomposition[kk] = {
            "extra_cost_of_permuted_in_F_nats": round(float(dF.mean()), 4),
            "extra_cost_of_permuted_in_X_nats": round(float(dX.mean()), 4),
            "representational_component_of_I_nats": round(comp, 4),
            "mc_ci95_paired": [
                round(float(np.percentile(bs, 2.5)), 4),
                round(float(np.percentile(bs, 97.5)), 4),
            ],
            "mc_halfwidth_paired_nats": round(
                float((np.percentile(bs, 97.5) - np.percentile(bs, 2.5)) / 2), 4
            ),
            "per_condition_sd_F": round(float(dF.std(ddof=1)), 4),
            "per_condition_sd_X": round(float(dX.std(ddof=1)), 4),
        }

    report = {
        "schema": "gpi0001_align_decomposition_v1",
        "design": "research/design/g1_align0001_predesign_v0.md",
        "status": "engineering premise check; not a verdict, not a qualification, no Gate 1 effect",
        "generator": {
            "nodes": M,
            "F_edges": [list(e) for e in F_EDGES],
            "X_edges": [list(e) for e in X_EDGES],
            "degree": 2,
            "edges": 6,
            "theta_grid": GRID,
            "grid_step": round(STEP, 5),
            "beta": BETA,
            "lambda": LAM,
            "T": T,
            "episodes": args.episodes,
            "projection_episodes": args.projection_episodes,
            "seed": SEED,
            "perm_seed": PERM_SEED,
        },
        "partitions": {k: [list(b) for b in v] for k, v in partitions.items()},
        "partition_checks": {
            "task_groups": [list(g) for g in TASK_GROUPS],
            "aligned_K2_is_aligned": is_aligned(ALIGNED_2),
            "aligned_K4_is_aligned": is_aligned(ALIGNED_4),
            "permuted_K2_crosses_groups": crosses_groups(PERMUTED_2),
            "permuted_K4_crosses_groups": crosses_groups(PERMUTED_4),
            "crossing_dims_K2": sum(
                len(b) for b in PERMUTED_2 if not any(set(b) <= set(g) for g in TASK_GROUPS)
            ),
            "crossing_dims_K4": sum(
                len(b) for b in PERMUTED_4 if not any(set(b) <= set(g) for g in TASK_GROUPS)
            ),
        },
        "results": results,
        "decomposition": decomposition,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
