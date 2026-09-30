"""G1-ALIGN-0001 声明轴验证 pilot（结构 + 计时；无学习、秒级）。

按 `research/design/g1_align0001_predesign_v0.md` §4 与判据版本 `G1-MATCH-v3`：
注册前须实测"声明消耗轴"的超出消耗 argmax。本设计的声明是**不消耗任何轴**，
故须给出全轴比值表并确认全部落在 1.000 ± 0.02。

两臂（S 对齐 / U 置换）共用同一个块对角循环矩阵，**只有输入路由不同**：
  S: 块 b 接收第 b 组的潜变量维度；
  U: 块 b 接收一个冻结随机置换下的维度子集（组大小相同）。
故参数、状态、非零数、执行形状应**逐项恒等**，差异只在路由 mask。

输出：JSON 到 stdout（以及 --output 指定的路径）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time

import torch

# 冻结常量（与 predesign v0 §3 一致）
M_LATENT = 6
W_TOTAL = 64
T_STEPS = 64
BATCH = 128
PERM_SEED = 20_260_924
GROUP_SIZES = {2: (3, 3), 4: (2, 2, 1, 1)}
# 任务的分组（F 的两个不相交三环）
TASK_GROUPS = ((0, 1, 2), (3, 4, 5))


def _aligned_partition(k: int) -> list[list[int]]:
    """**组内**切分：K=2 为 (012)(345)，K=4 为 (01)(2)(34)(5)。

    注意：连续切分 (0,1),(2,3),(4),(5) 的 (2,3) **跨组**，不是对齐划分。
    """
    if k == 2:
        return [list(TASK_GROUPS[0]), list(TASK_GROUPS[1])]
    return [[0, 1], [2], [3, 4], [5]]


def _crosses_groups(part: list[list[int]]) -> bool:
    return any(not any(set(b) <= set(g) for g in TASK_GROUPS) for b in part)


def _frozen_partition(k: int, generator: torch.Generator) -> list[list[int]]:
    """冻结的**跨组**随机划分：把 M_LATENT 个维度按 GROUP_SIZES[k] 分给 k 个块。

    必须至少有一个块跨任务组，否则「置换」臂与「对齐」臂无实质差别。
    """
    for _ in range(1000):
        dims = torch.randperm(M_LATENT, generator=generator).tolist()
        out, i = [], 0
        for size in GROUP_SIZES[k]:
            out.append(sorted(dims[i : i + size]))
            i += size
        assert i == M_LATENT and sum(len(g) for g in out) == M_LATENT
        if _crosses_groups(out):
            return out
    raise RuntimeError("failed to draw a cross-group partition")


def build_arms(k: int, seed: int) -> dict:
    """构建两臂；循环矩阵**完全共享**，仅输入路由 mask 不同。"""
    torch.manual_seed(seed)
    width = W_TOTAL // k
    aligned = _aligned_partition(k)
    gen = torch.Generator().manual_seed(PERM_SEED + k)
    permuted = _frozen_partition(k, gen)

    # 循环矩阵：每块一个 (width, width)，两臂逐字节相同
    rec = [torch.randn(width, width) / width**0.5 for _ in range(k)]
    # 输入权重：每块一个 (width, M_LATENT)，两臂逐字节相同
    inp = [torch.randn(width, M_LATENT) / M_LATENT**0.5 for _ in range(k)]

    def route(part: list[list[int]]) -> torch.Tensor:
        mask = torch.zeros(k, width, M_LATENT)
        for b, dims in enumerate(part):
            mask[b, :, dims] = 1.0
        return mask

    return {
        "k": k,
        "width": width,
        "recurrent": rec,
        "input": inp,
        "route_S": route(aligned),
        "route_U": route(permuted),
        "partition_S": aligned,
        "partition_U": permuted,
    }


def _sha(tensors) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().to(torch.float64).numpy().tobytes())
    return h.hexdigest()


def measure(k: int, seed: int, repeats: int, warmup: int) -> dict:
    arms = build_arms(k, seed)
    rec, inp = arms["recurrent"], arms["input"]
    width, kk = arms["width"], arms["k"]

    x = torch.randn(BATCH, T_STEPS, M_LATENT)

    def forward(route: torch.Tensor) -> torch.Tensor:
        h = [torch.zeros(BATCH, width) for _ in range(kk)]
        for t in range(T_STEPS):
            xt = x[:, t, :]  # (B, M)
            new = []
            for b in range(kk):
                drive = (xt * route[b].sum(dim=0)) @ inp[b].T  # (B, width)
                new.append(torch.tanh(drive + h[b] @ rec[b].T))
            h = new
        return torch.cat(h, dim=1)

    # 计时（交错顺序，两臂各 warmup 后 repeats 次）
    times = {"S": [], "U": []}
    for _ in range(warmup):
        forward(arms["route_S"])
        forward(arms["route_U"])
    for i in range(repeats):
        order = ["S", "U"] if i % 2 == 0 else ["U", "S"]
        for name in order:
            t0 = time.perf_counter()
            forward(arms["route_" + name])
            times[name].append(time.perf_counter() - t0)

    def params(tensors) -> int:
        return int(sum(t.numel() for t in tensors))

    rec_nz = [int((t != 0).sum()) for t in rec]
    n_params = params(rec) + params(inp)
    state = W_TOTAL  # 每臂递推状态 float 数
    n_steps = T_STEPS * kk  # 内部步数：每时间步每块一次

    p50_S = statistics.median(times["S"])
    p50_U = statistics.median(times["U"])

    return {
        "k": k,
        "width": width,
        "partition_S": arms["partition_S"],
        "partition_U": arms["partition_U"],
        "recurrent_matrix_sha_shared": _sha(rec),
        "input_weight_sha_shared": _sha(inp),
        "route_S_nonzeros": int(arms["route_S"].sum()),
        "route_U_nonzeros": int(arms["route_U"].sum()),
        "route_per_block_S": [int(arms["route_S"][b].sum()) for b in range(kk)],
        "route_per_block_U": [int(arms["route_U"][b].sum()) for b in range(kk)],
        "ratios": {
            "allocated_parameters": 1.0,  # 同一批张量，恒等
            "effective_parameters": 1.0,
            "recurrent_state": 1.0,
            "examples": 1.0,
            "updates": 1.0,
            "internal_steps": 1.0,
            "latency_p50": p50_S / p50_U,
        },
        "absolute": {
            "allocated_parameters": n_params,
            "recurrent_state_floats": state,
            "internal_steps_per_forward": n_steps,
            "recurrent_nonzeros": sum(rec_nz),
            "p50_seconds_S": p50_S,
            "p50_seconds_U": p50_U,
            "repeats": repeats,
            "warmup": warmup,
        },
    }


def count_ops(k: int, seed: int, steps: int) -> dict:
    """**实测**前向/反向/优化器步数与更新标量数（审查 I6 / 设计 v1 §5 要求）。

    把 `examples`/`updates`/`internal_steps` 从**构造性断言**升级为**实测计数**。
    两臂用同一 seed 重建同初值的可训练副本、同一数据、同一优化器，**唯一差异是 route**。
    本函数会真的做 `steps` 次前向+反向+`step()`（极小规模、无任务样本、无科研结论）。
    """
    torch.manual_seed(seed)
    arms = build_arms(k, seed)
    width, kk = arms["width"], arms["k"]

    counts, final_loss = {}, {}
    for name in ("S", "U"):
        torch.manual_seed(seed)
        # 先缩放再置 requires_grad_，否则除法会得到非叶张量（Adam 拒绝）
        rec = [(torch.randn(width, width) / width**0.5).requires_grad_(True) for _ in range(kk)]
        inp = [(torch.randn(width, M_LATENT) / M_LATENT**0.5).requires_grad_(True) for _ in range(kk)]
        out_w = torch.randn(1, W_TOTAL).requires_grad_(True)
        route = arms["route_" + name]
        params = rec + inp + [out_w]
        opt = torch.optim.Adam(params, lr=1e-3)

        x = torch.randn(BATCH, T_STEPS, M_LATENT)
        y = torch.randn(BATCH, 1)

        fwd = bwd = opt_steps = 0
        last = None
        for _ in range(steps):
            h = [torch.zeros(BATCH, width) for _ in range(kk)]
            for t in range(T_STEPS):
                xt = x[:, t, :]
                new = []
                for b in range(kk):
                    drive = (xt * route[b].sum(dim=0)) @ inp[b].T
                    new.append(torch.tanh(drive + h[b] @ rec[b].T))
                h = new
            fwd += 1
            loss = ((torch.cat(h, dim=1) @ out_w.T - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            bwd += 1
            opt.step()
            opt_steps += 1
            last = float(loss.detach())

        counts[name] = {
            "forward_calls": fwd,
            "backward_calls": bwd,
            "optimizer_steps": opt_steps,
            "tensors_updated": len(params),
            "updated_scalars": int(sum(p.numel() for p in params)),
            "examples_seen": fwd * BATCH,
            "internal_steps": fwd * T_STEPS * kk,
        }
        final_loss[name] = last

    keys = sorted(counts["S"])
    ratios = {a: (counts["S"][a] / counts["U"][a] if counts["U"][a] else None) for a in keys}
    return {
        "k": k,
        "steps": steps,
        "counts_S": counts["S"],
        "counts_U": counts["U"],
        "ratios": ratios,
        "final_loss_S": final_loss["S"],
        "final_loss_U": final_loss["U"],
        "all_counts_equal": all(counts["S"][a] == counts["U"][a] for a in keys),
        "scope": (
            "Real forward/backward/optimiser counting on engineering random inputs only. "
            "No task samples, no learning result, no Gate 1 effect."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=9_900_100)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=5, help="实测计数的前向/反向/优化器步数")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    cells = [measure(k, args.seed, args.repeats, args.warmup) for k in (2, 4)]
    op_counts = [count_ops(k, args.seed, args.steps) for k in (2, 4)]

    # 判定：全部轴比值落在 1.000 ± tolerance
    violations = []
    for c in cells:
        for axis, ratio in c["ratios"].items():
            if abs(ratio - 1.0) > args.tolerance:
                violations.append({"k": c["k"], "axis": axis, "ratio": ratio})

    # 声明轴验证：超出消耗 argmax
    worst = max(
        ({"k": c["k"], "axis": a, "excess": abs(r - 1.0)} for c in cells for a, r in c["ratios"].items()),
        key=lambda d: d["excess"],
    )

    report = {
        "design": "research/design/g1_align0001_predesign_v1.md",
        "matching_criterion_version": "G1-MATCH-v3",
        "declared_axis": "none (both arms share the same tensors; only input routing differs)",
        "tolerance": args.tolerance,
        "cells": cells,
        "operation_counts": op_counts,
        "violations": violations,
        "excess_consumption_argmax": worst,
        "verdict": "DECLARATION_VALID" if not violations else "DECLARATION_INVALID",
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
        },
        "scope": (
            "Structural, timing and operation-count pilot only. The operation counts are real "
            "forward/backward/optimiser calls on engineering random inputs; there is no task "
            "sample, no learning result and no Gate 1 effect. Not a task qualification, not a verdict."
        ),
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
