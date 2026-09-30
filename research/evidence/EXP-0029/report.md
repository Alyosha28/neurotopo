# EXP0029 v1：候选学习资格 12 seed 复现与失败模式诊断结果

2026-09-29。判词 **`PARTIALLY_SUPPORTED / LEARNING_QUALIFICATION_AT_BUDGET`**。候选仍为已冻结的 [`SC-REL-v3`](../design/arch0001_successor_compatible_core_v3.md)（A0，决定0102），协议 [`experiments/EXP-0029/v1.md`](../../experiments/EXP-0029/v1.md)（运行前冻结），配置 [`configs/exp0029_v1.json`](../../configs/exp0029_v1.json)，产物 `artifacts/EXP-0029/v1/`。本轮只测"该预算/该前测任务上的学习资格可靠性"，**不产生 A2 效应判词**，A1 仍因缺少非封顶合格任务而未通过（决定0103）。

## 结果

12 个新 seed（28201–28212）× 3 臂，各 1000 次固定课程更新，共 36000 研究更新、36 次拟合共 217.7 秒；评估用 EXP0020 的 8 个 cell、每 cell 1024 条、按 seed 独立取流。

| 臂 | 学习旗标通过 | IID_U 翻转 | LONG_U 全时点 | ROT_U 全时点 | LEAK04_U 全时点 | LEAK04_U 翻转 |
|---|---:|---:|---:|---:|---:|---:|
| `CAND`（SC-REL-v3，664 参数） | **12/12** | .9999 | .9919 | .9918 | .9012 | .9057 |
| `EXTRA_SUP`（基础 REL＋同分支监督，340 参数） | **12/12** | .9987 | .9767 | .9767 | .9151 | .9169 |
| `INV_COMP(w32,b1)`（强控制，406 参数） | **12/12** | 1.0000 | 1.0000 | 1.0000 | .9990 | .9990 |

逐 seed 明细见 `artifacts/EXP-0029/v1/analysis.json`（`summary.arms[*].rows`），失败模式分类器本轮未触发（无失败）。最难 seed 是 28209（`CAND` IID_U 翻转 .9989、`EXTRA_SUP` .9845），仍全部通过；其余 seed 基本为 1.0000。

## 主终点与前 3 个 seed 的分别报告

**主终点（预注册）＝本轮 12 个新 seed**：三臂均 12/12 通过，按判词表第 4 行（`CAND`∈{11,12}/12 且全部失败均为 `HOLD_COLLAPSE`；零失败时该条件空真成立）判 `PARTIALLY_SUPPORTED / LEARNING_QUALIFICATION_AT_BUDGET`。

**前 3 个 seed 按预注册诊断第 5 条单独报告、不进入主终点**：EXP0028 的 `CAND` 为 2/3，唯一失败 28103 为 `HOLD_COLLAPSE`（IID_H 1.0000、翻转 .5026、终损 .5056）、`INV_COMP` 3/3。两批合并的 14/15 只是**事后描述性读数**，没有预注册功效、不用于置信区间，也不作为判词依据；可以确定的只是"12 个新 seed 全部通过，另有一个较早协议样本中的低频 `HOLD_COLLAPSE` 失败被完整保留"。

## 预注册诊断

1. **损失检查点**：12 个 seed 的第 250 步窗口（25 步均值）最高 0.4063（`CAND` seed28209），第 1000 步窗口最高 0.1283；与 EXP0028 失败 seed 28103 的 0.5056 相比，属正常收敛区间内的慢启动，不是该失败的那种晚期不收敛。
2. **命令响应一致性**：12 seed 均值 `response_joint_accuracy` 在 IID_U 为 `CAND` .9999 / `EXTRA_SUP` .9976 / `INV` 1.0000；在 LEAK04_U 为 .8826 / .8832 / .9989。`current_ok_next_response_bad` 在 LEAK04_U 为 .0254 / .0521 / .0001。
3. **臂间配对**：本轮无失败，无法检验配对结构；EXP0028 的唯一失败 seed（28103）只跑了 `CAND` 与 `INV_COMP`，其中 `INV_COMP` 通过，即该次失败只出现在带修正器的臂上（该轮未跑 `EXTRA_SUP`，不能据此断言修正器是原因）。
4. **描述性 cell 观察（不构成效果判词）**：在 LEAK04_U 上 `CAND` .9012 < `EXTRA_SUP` .9151 < `INV_COMP` .9990；差异方向与"修正器在该预算下有害"相容，但该比较未作 A2 预注册、控制本身接近封顶，只能作为 A2 任务设计的输入，不能当作架构结论。

## 独立复核

`scripts/verify_exp0029_replay.py` 在 `/tmp` 干净源码副本独立进程重训 3 个归档运行：`CAND` seed28212、`INV_COMP(w32,b1)` seed28201、最难 seed `EXTRA_SUP` seed28209。三者最终参数哈希、训练数据哈希、8 cell logits 哈希全部相等，全部指标差 0.0（`verification.json`，23.1 秒）。复核不重选种子、不改配置、不覆盖原产物。

## 工程偏离与来源

运行期间记录三处仅位于汇总与编排路径的工程修正，`fit`／`evaluate`／`condition_metrics` 与冻结的 `src/neurotopo/arch0001_core.py`（哈希与 EXP0028 manifest 一致）未改动：模块哈希链为 `ca338347…`（首轮 run，manifest 记录）→ `3de25da0…`（第一次修正后，中间态）→ `7fb6bf81…`（当前，`provenance_check.json` 记录）。修正内容：(1) `cells_mean` 对 HOLD cell 的 `flip_accuracy=None` 做空值保护——首轮 `--phases run` 的 `TypeError` 发生在 36 个 run 记录全部写盘之后，校验 36 个记录全部存在且可解析，无缺记录/非有限值，故协议 `ENGINEERING` 行不成立，断点续跑读取也未被用于恢复任何记录；(2) `runs/*.json` 过滤 macOS `._*` 附属文件；(3) 断点续跑读取。重放偏离：协议写 2 个目标（一个通过、一个失败，若存在），本轮无失败 run，实际重放 3 个通过 run，第三个"最难 seed 28209"是事后选择；28103 失败运行的重放由 EXP0028 `verification.json` 覆盖（0 误差）。运行清单记录 commit `f5a6e2c`、`git_dirty=true`（活动 git 元数据在沙箱可写范围外，无法提交）。

## 边界与重开条件

本轮只支持"候选与控制在冻结 IID8 课程/预算下有可复现的学习资格（14/15 与 15/15）"，不改变 A1 判词（仍缺非封顶合格任务），不替代 A2，不追认旧 Gate 1，也不证明修正器有效。重开条件写入决定0104：A2 任务族仍须按决定0103 第 2 条另行注册（控制可学非封顶、排除简单显式解、有受界 oracle）；任何候选改动仍须命名 v4 并保留 v3 与本次失败记录；本轮已暴露的 8 个 cell 与 seed 不得用于反向挑选 A2 配置。
