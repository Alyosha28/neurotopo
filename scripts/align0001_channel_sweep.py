"""G1-ALIGN-0001 后续可行性扫描：容量通道随**任务图跨组边数**的变化。

决定0072 §5 的可救路径要求"重新界定效应量使容量通道显著小于 δ_min"。本脚本回答
**该任务族里是否存在这样的构造**：把 6 边任务图按**跨组边数 `c`** 参数化，对每个 `c`
计算 `Δ(c) = KL投影(跨组划分) − KL投影(对齐划分)`，则任一对条件 `(c₁, c₂)` 的
**容量通道成分**为 `Δ(c₁) − Δ(c₂)`。

`c = 0` 即 GPI-0001 的 F（两个组内三角），`c = 6` 取 GPI-0001 的 X（跨组 6-环）。
判据：若存在 `(c₁, c₂)` 使成分上界 `< δ_min/2 = 0.025` **且**两条件的 `gap(16)` 区间
下界都 `≥ 0.20`（非局部性门），则该构造可承载一个新设计；否则记录为**该任务族内不可行**。

无训练；工程前提检查，不是判词、不是资格、不构成任何效应结论。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys

import numpy as np

M = 6
GRID = 32
T = 64
SEED = 20_260_924
GROUPS = ((0, 1, 2), (3, 4, 5))
ALIGNED_K4 = ((0, 1), (2,), (3, 4), (5,))
CROSSED_K4 = ((0, 5), (2, 3), (4,), (1,))

WITHIN_POOL = ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3))
CROSS_POOL = ((0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5))
X_EDGES = ((0, 3), (3, 1), (1, 4), (4, 2), (2, 5), (5, 0))


def graph_for_c(c: int):
    """6 边图，恰有 c 条跨组边；c=0 为 F，c=6 为 X。确定性构造。"""
    if c == 0:
        return WITHIN_POOL
    if c == 6:
        return X_EDGES
    return tuple(WITHIN_POOL[: 6 - c]) + tuple(CROSS_POOL[:c])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--projection-episodes", type=int, default=40)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--delta-min", type=float, default=0.05)
    ap.add_argument("--gap-gate", type=float, default=0.20)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    sys.path.insert(0, "scripts")
    from align0001_analytic_decomposition import (  # noqa: E402
        build, factored_kl, gap_profile, joint_filter, sample,
    )

    rng = np.random.default_rng(SEED)
    boot_rng = np.random.default_rng(SEED + 11)

    cells = {}
    for c in range(7):
        edges = graph_for_c(c)
        logT, cum, prior = build(edges)
        obs = sample(edges, logT, cum, rng, args.episodes)
        gaps, _ = gap_profile(logT, prior, obs, ks=(16,), rng=np.random.default_rng(SEED + 1))
        posters = [joint_filter(o, logT, prior) for o in obs[: args.projection_episodes]]
        per = {}
        for pname, part in (("aligned", ALIGNED_K4), ("crossed", CROSSED_K4)):
            per[pname] = np.array([factored_kl(p, part) for p in posters])
        d = per["crossed"] - per["aligned"]          # 同一 episode 上配对
        bs = np.array([float(boot_rng.choice(d, len(d), replace=True).mean()) for _ in range(args.boot)])
        cells[c] = {
            "cross_edges": c,
            "edges": [list(e) for e in edges],
            "degree_sequence": sorted(
                sum(1 for e in edges if v in e) for v in range(M)
            ),
            "aligned_projection_kl": round(float(per["aligned"].mean()), 4),
            "crossed_projection_kl": round(float(per["crossed"].mean()), 4),
            "delta": round(float(d.mean()), 4),
            "delta_ci95": [round(float(np.percentile(bs, 2.5)), 4),
                           round(float(np.percentile(bs, 97.5)), 4)],
            "gap16_mean": gaps["16"]["mean"],
            "gap16_ci95": gaps["16"]["ci95"],
            "nonlocality_gate_pass": bool(
                gaps["16"]["ci95"] is not None and gaps["16"]["ci95"][0] >= args.gap_gate
            ),
        }

    # 枚举条件对 (c1 < c2)：容量通道成分 = Δ(c1) − Δ(c2)
    pairs = []
    for c1 in range(7):
        for c2 in range(c1 + 1, 7):
            comp = cells[c1]["delta"] - cells[c2]["delta"]
            # 保守上界：成分绝对值 + 两个 Δ 的区间半宽之和
            hw = lambda c: (cells[c]["delta_ci95"][1] - cells[c]["delta_ci95"][0]) / 2
            ub = abs(comp) + hw(c1) + hw(c2)
            pairs.append({
                "c1": c1, "c2": c2,
                "component": round(comp, 4),
                "conservative_upper_bound": round(ub, 4),
                "both_conditions_nonlocal": bool(
                    cells[c1]["nonlocality_gate_pass"] and cells[c2]["nonlocality_gate_pass"]
                ),
                "viable": bool(ub < args.delta_min / 2 and
                               cells[c1]["nonlocality_gate_pass"] and
                               cells[c2]["nonlocality_gate_pass"]),
            })

    viable = [p for p in pairs if p["viable"]]
    nonlocal_cs = [c for c in range(7) if cells[c]["nonlocality_gate_pass"]]

    report = {
        "schema": "align0001_channel_sweep_v1",
        "status": "engineering premise check; not a verdict, no training, no Gate 1 effect",
        "question": ("Does any pair of conditions in this task family have a capacity channel "
                     "small enough (< delta_min/2) while both conditions stay non-local?"),
        "constants": {"M": M, "T": T, "grid": GRID, "delta_min": args.delta_min,
                      "gap_gate": args.gap_gate, "episodes": args.episodes,
                      "projection_episodes": args.projection_episodes, "seed": SEED},
        "cells": {str(k): v for k, v in cells.items()},
        "pairs": pairs,
        "nonlocal_conditions": nonlocal_cs,
        "viable_pairs": viable,
        "verdict": "VIABLE_CONSTRUCTION_EXISTS" if viable else "NO_VIABLE_CONSTRUCTION_IN_THIS_FAMILY",
        "reading": (
            f"{len(viable)} condition pair(s) satisfy both the capacity-channel bound and the "
            "non-locality gate; a new design could be built on one of them."
            if viable else
            "No condition pair in this family has a capacity channel below delta_min/2 while "
            "keeping both conditions non-local. Within this task family the alignment effect "
            "cannot be separated from the capacity channel by choosing a different graph."
        ),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform()},
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
