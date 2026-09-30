# 决定0104：EXP0029 v1 学习资格复现判词

2026-09-29。按运行前冻结的 [`experiments/EXP-0029/v1.md`](../experiments/EXP-0029/v1.md)，EXP0029 v1 判词为 **`PARTIALLY_SUPPORTED / LEARNING_QUALIFICATION_AT_BUDGET`**。证据见 [`research/evidence/EXP-0029/report.md`](../evidence/EXP-0029/report.md)、产物 `artifacts/EXP-0029/v1/`、独立重放 [`verification.json`](../evidence/EXP-0029/verification.json) 与来源记录 [`provenance_check.json`](../evidence/EXP-0029/provenance_check.json)。

## 判词依据

**主终点＝协议预注册的 12 个新 seed（28201–28212）**，每臂 1000 次固定课程更新，共 36000 研究更新：**三臂均 12/12 通过** EXP0020 学习旗标。按判词表第 4 行（`CAND`∈{11,12}/12 且全部失败均为 `HOLD_COLLAPSE`），失败集合为空时该条件取空真读法成立，判 `PARTIALLY_SUPPORTED / LEARNING_QUALIFICATION_AT_BUDGET`；第 5 行要求"失败存在"，不适用。评估均值：`INV_COMP` 全部 cell 近 1.0000；`CAND` IID_U 翻转 .9999、LEAK04_U .9057；`EXTRA_SUP` .9987／.9169。独立干净副本重放 3 个归档运行（含最难 seed 28209）参数/数据/预测哈希与全部指标差为 0。

**前 3 个 seed 按预注册诊断第 5 条单独报告，不进入主终点**：EXP0028 的 `CAND` 为 2/3 通过，唯一失败 28103 为 `HOLD_COLLAPSE`（IID_H 1.0000、翻转 .5026、终损 .5056），`INV_COMP` 3/3。把两批合并成 14/15 是**事后描述性读数**，不是本次判词的依据、没有预注册功效，也不用于置信区间估计；本轮结论只限"12 个新 seed 全部通过，且已知一个较早协议样本中的低频失败模式为 `HOLD_COLLAPSE`"。

## 状态

- **A1 学习资格条款：该预算/该前测任务上 `PARTIALLY_SUPPORTED`**（不是 A1 通过）。
- **A1 门整体仍未通过**：`ARCH-GATES-v1` 还要求任务有信息与非封顶余量，而 EXP0028 的预注册菜单无合格条件（决定0103）。
- A2–A6 仍 `NOT_EVALUATED`；旧 Gate 1 与全部旧判词不变；`ResearchLoop` `project_gate1` 未被触碰。
- 描述性观察（不构成效果判词）：LEAK04_U 上 `CAND` .9012 < `EXTRA_SUP` .9151 < `INV_COMP` .9990；该比较未预注册为效果终点、控制接近封顶，只能作为 A2 任务设计的输入。
- 工程记录（见报告与 `provenance_check.json`）：
  - 运行期只修改汇总与编排路径（`summarize` 的空值保护、`runs/*.json` 过滤 macOS `._*`、已完成 run 的断点续跑读取）；`fit`／`evaluate`／`condition_metrics` 与冻结的 `src/neurotopo/arch0001_core.py` 均未改动（core 哈希与 EXP0028 manifest 一致）。
  - 模块哈希链：`ca338347…`（manifest 记录，首轮 `--phases run` 所用）→ `3de25da0…`（第一次修正后，当时记录）→ `7fb6bf81…`（第二次修正后，即当前与 `provenance_check.json` 记录的版本）。三处修改都不在训练/评估代码路径上；此前决定正文引用的 `3de25da0…` 是中间态，现更正为完整链。
  - 首轮 `--phases run` 在 `summarize` 抛出 `TypeError`（HOLD cell 的 `flip_accuracy=None` 进入 `np.mean`），**发生在 36 个 run 记录全部写盘之后**；校验 36 个 `runs/*.json` 全部存在且可解析，无缺记录、无非有限值，故协议中 `ENGINEERING` 行（非有限值/异常/缺记录）不成立；断点续跑读取路径最终未被用于恢复任何记录（分析阶段直接读取既有记录）。
  - 运行与分析各阶段都在 `scripts/verify.py` 通过且 `suite.json` 源码哈希与运行时一致后才执行；哈希不匹配时运行器会拒绝执行（该保护在第一次分析尝试时实际触发过）。运行清单记录 commit `f5a6e2c`、`git_dirty=true`（沙箱不可写活动 git 元数据，未创建提交）。
  - 重放偏离：协议写"重放 2 个已归档运行（一个通过、一个失败，若存在）"；本轮无失败 run，实际重放 3 个通过 run（`CAND` seed28212、`INV_COMP` seed28201、`EXTRA_SUP` seed28209）；第三个目标"最难 seed 28209"是看到结果后选择的，属事后加强而非预注册；28103 失败运行的重放已由 EXP0028 的 `verification.json` 覆盖（0 误差）。

## 重开条件

1. A2 任务族仍须按决定0103 第 2 条另行注册：强控制可学且非封顶、排除"简单显式规则即可解"、给出解析或受界 oracle，菜单/选择规则/评估流在运行前冻结。这是当前 A1 的主要缺口。
2. 后续 A1 协议版本若把学习资格种子预算提高（例如以 15 seed 复现），必须保留 1/15 的失败记录与其 `HOLD_COLLAPSE` 归类，不得只报通过 seed。
3. 任何对候选方程、初始化、损失或推理路径的改动必须命名 `v4` 并保留 v3、EXP0028 与 EXP0029 的全部记录；改动须先给出可测预测与失败判据。
4. 本轮已暴露的 8 个 cell、15 个 seed 与描述性 cell 比较不得用于反向挑选 A2 配置或追认任何效果。
5. 后续版本的判词表须显式写出"零失败"一行，避免依赖空真读法；packet 的 action 摘要须逐项列明省略了哪些字段（本轮省略了每 seed 的次 cell 明细，完整值在 `artifacts/EXP-0029/v1/runs/*.json`）。
