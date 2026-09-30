"""G1-ALIGN-0001：增广状态 (theta, z) 前向 oracle 的**独立第二实现**交叉核对。

设计 v1 §13.1 要求 oracle 需独立重建核对（索引/符号/发射），照 CAL-0008 的两层核对做法。
本脚本**不复用** `align0001_analytic_decomposition.py` 的滤波代码，改用**不同的实现路径**：

  第一实现（被核对方，`align0001_analytic_decomposition.py`）
    - 状态序：itertools.product((-1,1), repeat=M)，dim 0 为**最高**位
    - 转移：log 空间张量 `(GRID, 2^M, 2^M)`，一次 `einsum('ks,ksj->kj')` 同时传播全部 theta
    - 发射：`-0.5 * sum((y - lam*z)^2)`（残差平方）
    - 先验：对每个 J 求平稳律后除以 GRID

  第二实现（本脚本）
    - 状态序：dim 0 为**最低**位，用显式下标映射换回第一实现
    - 转移：**逐 (z,z') 对显式双层循环**构造，**逐 theta 显式循环**传播（无 einsum）
    - 发射：**展开式** `-0.5*(y.y - 2*lam*y.z + lam^2*||z||^2)`，不用残差平方
    - 平稳律：**幂迭代**与**左特征向量**两条独立路径互核
    - 递推：概率空间、每步归一化

两者都在 **(theta, z)** 联合空间上递推；若逐元素一致，则索引/符号/发射/归一化的实现风险显著降低。
无训练；工程前提检查，不是判词。
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys

import numpy as np

M = 6
LAM = 0.7
BETA = 0.8
GRID = 32
SEED = 20_260_924

F_EDGES = ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3))
X_EDGES = ((0, 3), (3, 1), (1, 4), (4, 2), (2, 5), (5, 0))

JS = np.linspace(0.0, 0.6, GRID)
N = 2**M


def bits_lsb_first(index: int) -> tuple[int, ...]:
    return tuple((index >> i) & 1 for i in range(M))


STATES_B = np.array([[1.0 if b else -1.0 for b in bits_lsb_first(i)] for i in range(N)])
PERM_B_TO_A = np.array(
    [sum(b << (M - 1 - d) for d, b in enumerate(bits_lsb_first(i))) for i in range(N)]
)
ZZ_B = (STATES_B**2).sum(axis=1)


def transition_b(J: float, edges) -> np.ndarray:
    """逐 (z, z') 对显式构造转移矩阵（B 序）。"""
    e = sum(STATES_B[:, i] * STATES_B[:, j] for i, j in edges)
    gram = STATES_B @ STATES_B.T
    out = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            out[i, j] = math.exp(BETA * float(gram[i, j]) + J * float(e[j]))
        out[i] /= out[i].sum()
    return out


def stationary_power(T_: np.ndarray, iters: int = 50000, tol: float = 1e-16) -> np.ndarray:
    d = np.full(len(T_), 1.0 / len(T_))
    for _ in range(iters):
        nd = d @ T_
        if np.max(np.abs(nd - d)) < tol:
            return nd
        d = nd
    return d


def stationary_eig(T_: np.ndarray) -> np.ndarray:
    """独立路径：T 的左特征向量（特征值 1）。"""
    w, v = np.linalg.eig(T_.T)
    k = int(np.argmin(np.abs(w - 1.0)))
    d = np.abs(np.real(v[:, k]))
    return d / d.sum()


def filter_b(obs: np.ndarray, T_list, prior_b: np.ndarray) -> np.ndarray:
    """第二实现：在 (theta, z) 联合空间上递推，逐 theta 显式循环，概率空间归一化。"""
    p = prior_b.copy()
    for t in range(len(obs)):
        y = obs[t]
        yy = float(y @ y)
        yz = STATES_B @ y
        loglik = -0.5 * (yy - 2.0 * LAM * yz + LAM**2 * ZZ_B)  # (N,)
        p = p * np.exp(loglik[None, :] - loglik.max())
        p /= p.sum()
        if t < len(obs) - 1:
            new = np.empty_like(p)
            for k in range(GRID):
                new[k] = p[k] @ T_list[k]
            p = new
            p /= p.sum()
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--tolerance", type=float, default=1e-9)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    sys.path.insert(0, "scripts")
    from align0001_analytic_decomposition import build, joint_filter, sample  # noqa: E402

    rng = np.random.default_rng(SEED)
    report = {
        "schema": "align0001_oracle_crosscheck_v1",
        "status": "engineering premise check; not a verdict, no training",
        "tolerance": args.tolerance,
        "conditions": {},
    }

    for name, edges in (("F", F_EDGES), ("X", X_EDGES)):
        logT, cum, prior_a = build(edges)  # prior_a: (GRID, N)，A 序，已含 1/GRID
        obs = sample(edges, logT, cum, rng, args.episodes)
        T_list = [transition_b(J, edges) for J in JS]

        # 第二实现的先验：独立求平稳律
        prior_b = np.stack([stationary_power(T_list[k]) for k in range(GRID)]) / GRID

        # 平稳律：幂迭代 vs 特征分解（两条独立路径，J=0）
        stat_gap_j0 = float(np.max(np.abs(stationary_power(T_list[0]) - stationary_eig(T_list[0]))))

        # 先验的 A/B 序核对
        prior_gap = float(np.max(np.abs(prior_b[:, PERM_B_TO_A] - prior_a)))

        deltas = []
        for o in obs:
            pb = filter_b(o, T_list, prior_b)[:, PERM_B_TO_A]
            pa = joint_filter(o, logT, prior_a)
            deltas.append(float(np.max(np.abs(pb - pa))))

        mx = max(deltas)
        report["conditions"][name] = {
            "episodes": args.episodes,
            "max_abs_delta_posterior": mx,
            "mean_abs_delta_posterior": float(np.mean(deltas)),
            "max_abs_delta_prior": prior_gap,
            "stationary_power_vs_eig_gap_J0": stat_gap_j0,
            "pass": bool(mx < args.tolerance and prior_gap < args.tolerance),
        }

    ok = all(c["pass"] for c in report["conditions"].values())
    report["verdict"] = "ORACLE_CROSSCHECK_PASS" if ok else "ORACLE_CROSSCHECK_FAIL"
    report["reading"] = (
        "A second, independent implementation -- different state ordering, explicit pairwise "
        "transition table, per-theta explicit propagation with no einsum, expanded emission, and "
        "an independently recomputed stationary prior -- reproduces the first below tolerance on "
        "every episode."
        if ok
        else "The two implementations disagree; the oracle is not trustworthy until explained."
    )
    report["environment"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
