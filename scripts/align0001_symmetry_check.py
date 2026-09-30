"""G1-ALIGN-0001：判定 S/U 两臂是否只是同一问题的重标号（对审查问题 I2 的决定性检查）。

独立前瞻设计审查提出：既然两臂共用同一个循环矩阵、路由非零数相同，则在 F 中
「两臂的优化问题等价到隐藏单元/输入索引置换」，臂间差异只能来自初始化与优化轨迹。

该断言可**精确**判定：两臂在条件 G 上等价，当且仅当存在 G 的**图自同构** σ，
把 S 的维度划分映到 U 的维度划分（因为任务只通过边集依赖维度，而两臂的循环矩阵
在各自划分下是同构的块对角结构）。

本脚本枚举全部 6! = 720 个置换，求出 F 与 X 的自同构群，再检查 S 划分与 U 划分
是否在同一轨道内。无训练、无采样；纯组合判定。

若两划分**不在**同一轨道 → I2 被驳倒，两臂是真正不同的问题。
若**在**同一轨道 → I2 成立，设计必须停止或改主终点。
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys

M = 6
F_EDGES = ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3))
X_EDGES = ((0, 3), (3, 1), (1, 4), (4, 2), (2, 5), (5, 0))

ALIGNED_2 = ((0, 1, 2), (3, 4, 5))
PERMUTED_2 = ((1, 2, 5), (0, 3, 4))  # §2 冻结的跨组划分 (125)(034)
# K=4 的「对齐」必须在任务组内切分（v0 的连续切分 (01)(23)(4)(5) 里 (2,3) 跨组，是错的）
ALIGNED_4 = ((0, 1), (2,), (3, 4), (5,))
PERMUTED_4 = ((0, 5), (2, 3), (4,), (1,))  # §2 冻结的跨组划分 (05)(23)(4)(1)
TASK_GROUPS = ((0, 1, 2), (3, 4, 5))


def is_aligned(partition) -> bool:
    return all(any(set(b) <= set(g) for g in TASK_GROUPS) for b in partition)


def crosses_groups(partition) -> bool:
    return any(not any(set(b) <= set(g) for g in TASK_GROUPS) for b in partition)


def edge_set(edges):
    return frozenset(frozenset(e) for e in edges)


def is_automorphism(perm, edges_fs) -> bool:
    """perm[i] = 顶点 i 的像。"""
    return frozenset(frozenset((perm[a], perm[b])) for a, b in (tuple(e) for e in edges_fs)) == edges_fs


def canon(partition, perm):
    """把划分按 perm 搬运，返回规范化（排序后的 frozenset of tuples）。"""
    return frozenset(tuple(sorted(perm[v] for v in block)) for block in partition)


def norm(partition):
    return frozenset(tuple(sorted(b)) for b in partition)


def analyse(name, edges, part_s, part_u):
    efs = edge_set(edges)
    autos = [p for p in itertools.permutations(range(M)) if is_automorphism(p, efs)]
    target = norm(part_u)
    source = norm(part_s)
    hits = [p for p in autos if canon(part_s, p) == target]
    orbit = {canon(part_s, p) for p in autos}
    return {
        "condition": name,
        "edges": [list(e) for e in edges],
        "automorphism_group_order": len(autos),
        "automorphisms": [list(p) for p in autos],
        "partition_S": [list(b) for b in source],
        "partition_U": [list(b) for b in target],
        "S_orbit_size": len(orbit),
        "S_and_U_in_same_orbit": bool(hits),
        "witness_automorphisms": [list(p) for p in hits[:8]],
        "verdict": "EQUIVALENT_UP_TO_RELABELLING" if hits else "GENUINELY_DIFFERENT",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    cells = []
    for k, ps, pu in (("K2", ALIGNED_2, PERMUTED_2), ("K4", ALIGNED_4, PERMUTED_4)):
        for name, edges in (("F", F_EDGES), ("X", X_EDGES)):
            cell = analyse(name, edges, ps, pu)
            cell["K"] = k
            cells.append(cell)

    any_equivalent = any(c["S_and_U_in_same_orbit"] for c in cells)

    report = {
        "schema": "gpi0001_align_symmetry_check_v1",
        "design": "research/design/g1_align0001_predesign_v1.md",
        "purpose": (
            "Decide review issue I2: are the S and U arms merely a relabelling of the same "
            "problem? Two arms are equivalent on condition G iff some automorphism of G's edge "
            "set maps the S dimension partition onto the U partition."
        ),
        "status": "combinatorial check; not a verdict, no training, no Gate 1 effect",
        "partitions_used": {
            "K2": {"S": [list(b) for b in ALIGNED_2], "U": [list(b) for b in PERMUTED_2]},
            "K4": {"S": [list(b) for b in ALIGNED_4], "U": [list(b) for b in PERMUTED_4]},
        },
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
        "cells": cells,
        "overall": {
            "I2_holds_anywhere": any_equivalent,
            "reading": (
                "I2 is REFUTED on every cell: no automorphism maps S onto U, so the two arms face "
                "genuinely different problems and an arm difference cannot be dismissed as "
                "relabelling. The claim that all axes being 1.000 forces the arms to be "
                "interchangeable is therefore false."
                if not any_equivalent
                else "I2 holds on at least one cell: on that condition the arms are equivalent up "
                "to relabelling and any arm difference there can only come from optimisation path."
            ),
        },
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
