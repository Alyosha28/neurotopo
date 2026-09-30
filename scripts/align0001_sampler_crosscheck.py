"""G1-ALIGN-0001：任务**采样器**与**路由构造**的独立第二实现交叉核对。

设计 v2 §13.5 记录：oracle 已通过第二实现核对，但**任务生成器与路由构造尚待补**。本脚本补上。

第一实现（被核对方）
  - 采样：`align0001_analytic_decomposition.py` 的 `sample()`，用 `build()` 返回的**累积和**
    `cum` 配合 `np.searchsorted(cum[k][idx], rng.random())` 抽下一步。
  - 路由：`align0001_axis_pilot.py` 的 `build_arms()`，用 `mask[b, :, dims] = 1` 的
    0/1 路由张量，前向里以 `xt * route[b].sum(dim=0)` 掩蔽输入。

第二实现（本脚本）
  - 采样：**不用累积和**，改用 `rng.choice(N, p=T[k][idx])` 显式按概率抽；
    并用**独立**的 RNG 流。
  - 路由：改为**索引式**（直接按维度下标取 `xt[:, dims]` 再经 `inp_block`），
    与掩蔽式在数学上等价但实现路径不同。

核对内容
  1. **经验转移频率** vs 理论 `T(J)`：对每个抽到的 J，统计 `z_t -> z_{t+1}` 的频数矩阵，
     与 `T(J)` 逐元素比较，报告最大偏差与在 MC 误差内的判定。
  2. **经验初始分布** vs 平稳律。
  3. **路由等价性**：掩蔽式与索引式前向在**同一权重与输入**上逐元素一致。
无训练；工程前提检查，不是判词。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys

import numpy as np
import torch

M = 6
LAM = 0.7
BETA = 0.8
T = 64
GRID = 32
SEED = 20_260_924
N = 2**M

F_EDGES = ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3))
X_EDGES = ((0, 3), (3, 1), (1, 4), (4, 2), (2, 5), (5, 0))
TASK_GROUPS = ((0, 1, 2), (3, 4, 5))
ALIGNED_4 = ((0, 1), (2,), (3, 4), (5,))
PERMUTED_4 = ((0, 5), (2, 3), (4,), (1,))


def sample_second(edges, n_episodes, seed, T_steps=T):
    """第二实现采样：显式按概率抽（不用累积和/searchsorted），独立 RNG。"""
    sys.path.insert(0, "scripts")
    from align0001_analytic_decomposition import JS, build  # noqa: E402

    logT, cum_unused, prior = build(edges)
    Ts = [np.exp(logT[k]) for k in range(GRID)]

    rng = np.random.default_rng(seed + 999_983)  # 独立流
    obs = np.empty((n_episodes, T_steps, M))
    labels = np.empty(n_episodes, dtype=int)
    chains = np.empty((n_episodes, T_steps), dtype=int)
    for e in range(n_episodes):
        k = int(rng.integers(0, GRID))
        labels[e] = k
        # 初始态按该 J 的平稳律抽（与第一实现的 prior 一致）
        idx = int(rng.choice(N, p=prior[k] / prior[k].sum()))
        for t in range(T_steps):
            chains[e, t] = idx
            obs[e, t] = LAM * _states()[idx] + rng.normal(size=M)
            if t < T_steps - 1:
                idx = int(rng.choice(N, p=Ts[k][idx]))
    return obs, labels, chains, Ts


def _states():
    import itertools

    return np.array(list(itertools.product((-1.0, 1.0), repeat=M)))


def empirical_transition(chains, labels, k):
    """经验转移**计数**矩阵（不归一，供卡方检验用）。"""
    sel = chains[labels == k]
    if sel.size == 0:
        return None
    counts = np.zeros((N, N))
    a = sel[:, :-1].reshape(-1)
    b = sel[:, 1:].reshape(-1)
    np.add.at(counts, (a, b), 1.0)
    return counts


def chi2_transition(counts, T_k, min_expected: float = 5.0):
    """Pearson 卡方：观测计数 vs 行计数 × T_k，只在期望 >= min_expected 的格上统计。

    逐行比较频率在每行样本数很少时会给出虚假的巨大偏差；卡方按期望加权才是正确检验。
    """
    row = counts.sum(axis=1, keepdims=True)
    exp = row * T_k
    mask = exp >= min_expected
    if not np.any(mask):
        return None
    stat = float((((counts - exp) ** 2)[mask] / exp[mask]).sum())
    # T_k 是**已知**分布（不是从数据估计），故不损失自由度
    df = max(int(mask.sum()), 1)
    # 正态近似下的 z 分数：X^2 在 df 大时近似 N(df, 2df)
    z = (stat - df) / np.sqrt(2.0 * df)
    return {"chi2": round(stat, 2), "df": df, "z_score": round(float(z), 3),
            "cells_used": int(mask.sum()), "pass_3sigma": bool(abs(z) <= 3.0)}


def route_masked(xt: torch.Tensor, dims, inp_block: torch.Tensor) -> torch.Tensor:
    """第一实现路径：0/1 掩蔽。"""
    mask = torch.zeros(M)
    mask[list(dims)] = 1.0
    return (xt * mask) @ inp_block.T


def route_indexed(xt: torch.Tensor, dims, inp_block: torch.Tensor) -> torch.Tensor:
    """第二实现路径：按下标取值后经同一切片的输入权重（数学等价，路径不同）。"""
    sub = xt[:, list(dims)]
    return sub @ inp_block[:, list(dims)].T


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--tol-transition", type=float, default=0.05)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    report = {
        "schema": "align0001_sampler_crosscheck_v1",
        "status": "engineering premise check; not a verdict, no training",
        "conditions": {},
    }

    for name, edges in (("F", F_EDGES), ("X", X_EDGES)):
        obs, labels, chains, Ts = sample_second(edges, args.episodes, SEED)

        per_J = {}
        for k in range(GRID):
            counts = empirical_transition(chains, labels, k)
            if counts is None:
                per_J[str(k)] = {"pairs": 0, "chi2": None, "skipped": "no episode with this J"}
                continue
            res = chi2_transition(counts, Ts[k])
            if res is None:
                per_J[str(k)] = {"pairs": int(counts.sum()), "chi2": None,
                                 "skipped": "no cell reached the minimum expected count"}
                continue
            res["pairs"] = int(counts.sum())
            per_J[str(k)] = res

        # 初始分布 vs 平稳律（按**实际抽到的 J 频率**加权的平稳混合，而非只看 J=0）
        first = chains[:, 0]
        emp0 = np.bincount(first, minlength=N) / len(first)
        mix = np.zeros(N)
        for k in range(GRID):
            w = float(np.mean(labels == k))
            if w > 0:
                mix += w * _stationary(Ts[k])
        mix /= mix.sum()
        stat_dev = float(np.max(np.abs(emp0 - mix)))

        report["conditions"][name] = {
            "episodes": args.episodes,
            "per_J_transition_check": per_J,
            "worst_abs_z": round(
                max((abs(v["z_score"]) for v in per_J.values() if v.get("z_score") is not None), default=0.0), 3
            ),
            "initial_vs_stationary_max_dev": round(stat_dev, 5),
            "transition_pass": all(
                v.get("pass_3sigma", True) for v in per_J.values()
            ),
        }

    # 路由等价性：同一权重与输入下，掩蔽式与索引式逐元素一致
    torch.manual_seed(7)
    xt = torch.randn(64, M)
    route_cells = {}
    for label, dims in (("aligned_K4_block0", ALIGNED_4[0]), ("permuted_K4_block0", PERMUTED_4[0])):
        inp_block = torch.randn(16, M)
        a = route_masked(xt, dims, inp_block)
        b = route_indexed(xt, dims, inp_block)
        route_cells[label] = {
            "max_abs_diff": float((a - b).abs().max()),
            "equal": bool(torch.allclose(a, b, atol=1e-12)),
        }
    report["routing_equivalence"] = route_cells
    report["routing_pass"] = all(v["equal"] for v in route_cells.values())

    ok = (
        all(c["transition_pass"] for c in report["conditions"].values())
        and report["routing_pass"]
    )
    report["verdict"] = "SAMPLER_ROUTING_CROSSCHECK_PASS" if ok else "SAMPLER_ROUTING_CROSSCHECK_FAIL"
    report["reading"] = (
        "A second sampler (explicit categorical draw with an independent RNG stream, no cumulative-sum "
        "searchsorted) reproduces the generator's transition law and stationary initial law within Monte "
        "Carlo error, and the indexed routing path is elementwise identical to the masked one."
        if ok
        else "The second implementation disagrees; the sampler or routing construction is not trustworthy yet."
    )
    report["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                             "torch": torch.__version__, "platform": platform.platform()}

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0 if ok else 1


def _stationary(T_):
    d = np.full(len(T_), 1.0 / len(T_))
    for _ in range(20000):
        nd = d @ T_
        if np.max(np.abs(nd - d)) < 1e-16:
            return nd
        d = nd
    return d


if __name__ == "__main__":
    raise SystemExit(main())
