# G1-BOUND-0001 / NeuroTopo 恢复检查点（2026-09-24）

本文件记录**当前确切状态与恢复所需的最小输入**，供后续会话接手。**不是判词、不是注册。**

## 1. 仓库状态

- `HEAD = 892274f`（其后若仅追加本文件，以 `git log` 为准）；工作树干净，仅一个与本项目无关的未跟踪目录 `artifacts/architecture-motion-20260924/`。
- `uv run --frozen pytest -q` → **218 passed**。
- Gate 1 行文本为 v3 判据；**状态仍 `NOT_PASSED`**。**无 EXP 注册、无训练授权**；非回归清单 12 条无一重开。
- 74 份决定文件、63 个反思记录目录。

## 2. 已完成的（可依赖）

| 项 | 落点 |
|---|---|
| G1-RS 任务寻找 + EXP-0014 关闭 | 决定0070；注册表 EXP-0014 行 `CLOSED_BEFORE_FORMAL` |
| `G1-MATCH-v3` 再入筛查（12 条，复活 0） | 决定0071 |
| G1-ALIGN-0001 设计 v0/v1/v2 + 三轮独立审查 | `research/reflections/G1-ALIGN-0001-DESIGN{,-V1,-V2}/`（`check` 均 `valid: true`） |
| **停止 G1-ALIGN-0001**（五条可机核停止条件全不满足） | 决定0072 |
| oracle / 采样器 / 路由的**独立第二实现核对** | `align0001_oracle_crosscheck.json`、`align0001_sampler_crosscheck.json` |
| **容量通道扫描 → `VIABLE_CONSTRUCTION_EXISTS`** | `align0001_channel_sweep.json`、`g1_align0001_channel_sweep_note.md` |
| **G1-BOUND-0001 设计 v0**（S1–S5 全过、分支穷尽） | `g1_bound0001_predesign_v0.md` |
| `gap(k)` 覆盖到 `k=T` 的门**定义性空洞**的更正 | `g1_bound0001_gap_gate_correction.md`、`align0001_gap_to_T.json` |
| 块交换性论证（K=4 逐块路由计数不等非混淆） | `g1_align0001_block_load_note.md` |
| 审查投递方式变更与独立性处置 | `research/reflections/G1-BOUND-0001-DESIGN/DELIVERY_NOTE.md` |

## 3. 状态更新（2026-09-24，审查已返回）

**独立审查已获得**：第三次投递（文件投递 + 问题收窄为两问）判 **`V0_REJECTED`**，见决定0073。
`research/reflections/G1-BOUND-0001-DESIGN/` 的 `check` 判 `valid: true`（3 项，`REASSESSMENT_REQUIRED`）；审查者 `FILES_READ` 仅列 packet 一个文件。

**结论：整个"路由对齐"方向停止**（决定0073）——G1-ALIGN-0001 与 G1-BOUND-0001 均不注册、不跑资格流、不训练；不再以换条件对/换措辞重开。

**因此 §5 的"恢复所需最小输入"对这两个设计已作废。** 若要继续推进项目，需要的是**新的研究问题**（不是本方向的收窄）：唯一被审查指明的救法是**架构级可达界**（架构级 oracle，或可证包含两臂架构的族），那已不是本设计的 `Δ`，须另立事前设计与独立审查。

**仍未完成**：预注册、跑通、判词——但它们**不再指向 G1-BOUND-0001**。

## 4. 历史阻塞记录（**已解除**）

**独立审查子代理不返回**——该条件曾跨 9 轮、3 次投递持续存在，目标因此两度标记受阻。**第三次投递（收窄为两问）成功返回**，阻塞已解除。时间线：

| 轮 | 代理 | 状态 |
|---|---|---|
| 7 | `43170ab4…`（内联投递） | 启动 |
| 8–10 | 同上 | 持续 running；第 10 轮发送催办要求立即输出 JSON |
| 11 | 同上 | 催办后仍无响应 → **判定卡死、已中断**，**未产出任何判词** |
| 11 | `2610f084…`（文件投递） | 重发，启动 |
| 12–13 | 同上 | 持续 running |
| 13 | `c73a709e…`（文件投递 + **问题收窄为两问**、`issues` 上限 3） | 启动 |
| 15 | 同上 | **返回 `V0_REJECTED`**（3 项），阻塞解除 |

**后果**：按仓库规则，设计**须经独立批评**才能进入预注册；故步骤 2、3 **无法开始**。

**不得**据首次代理称"本设计已获独立审查"（它未产出判词）。

## 5. 恢复所需的最小输入

1. **审查通道恢复**：重新投递 `research/reflections/G1-BOUND-0001-DESIGN/packet.json`（`packet_sha256 = 25120331d66b421a3c973ae5b3de6adcdc52aaddf32a022182e738940552c994`）给**全新上下文**审查者，要求其输出 `review.json`（schema 见 `src/neurotopo/reflection.py` 的 `validate_review`）并附 `FILES_READ`。
2. 审查落地后：写 `reaction.json`（19 项式逐条处置），跑
   `uv run --frozen python -m neurotopo.reflection check --packet … --review … --reaction …` 直至 `valid: true`。
3. 若审查判 `V0_NEEDS_NARROWING`：按 R 条出 `g1_bound0001_predesign_v1.md`；若 `V0_REJECTED`：按决定0072 §5 再评估或停止。
4. 通过后：写 `research/entries/EXP-XXXX.json` + `experiments/EXP-XXXX/v1.md`（**须含** `matching_criterion_version: G1-MATCH-v3`、声明消耗轴、冻结划分与 `C_ub = 0.0064`、`k_max = 16`），再走控制资格。

## 6. 复现命令

```bash
export UV_CACHE_DIR=/private/tmp/neurotopo-uv-cache
export UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-venv
cd .

uv run --frozen pytest -q
uv run --frozen python -m neurotopo.loop status
uv run --frozen python scripts/align0001_channel_sweep.py --output research/design/structural_task_probe/align0001_channel_sweep.json
uv run --frozen python scripts/align0001_symmetry_check.py --output research/design/structural_task_probe/align0001_symmetry_check.json
uv run --frozen python scripts/align0001_analytic_decomposition.py --episodes 200 --projection-episodes 60 --output research/design/structural_task_probe/align0001_analytic_decomposition.json
```

**注意**：本工作区是 NTFS 卷（macfuse），macOS 会不断在 `.git` 内生成 AppleDouble `._*` 侧车，导致 `error: non-monotonic index`。**每次 git 操作前**执行 `find .git -name '._*' -type f -delete`。

## 7. 会话上下文

本会话上下文已近上限。§3 的步骤 2、3（资格流与训练）所需工作量远超剩余余量；**即使审查恢复，本会话也可能无法完成"跑通 + 判词"**。接手时请以本文件 §1–§4 为准，不要依赖任何叙述。
