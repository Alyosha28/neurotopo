# DIR-0005 事前设计 v0：Gate 1 候选组合审计与候选规格

2026-09-25。**状态：设计稿，先于审计执行落盘并单独提交**（使"先设计、后取数"的次序可审计）。本稿**不注册实验、不授权训练、不生成样本、不主张 Gate 1、不改变 `NOT_PASSED`、不重开非回归清单任何条目**。

## 1. 问题

在**当前生效规则**（Gate 1 文本 `Generic trajectory benefit after matching the declared consumed axis`；`G1-MATCH-v3`；入口策略 `gate1_candidate_first_v2` 含决定0080 的 `architecture_organization`；决定0062 对学习效率终点的 B1–B7 限制；决定0049/0070 对 G1-RS 任务寻找的关闭；决定0081 对 DIR-0004 的停止）与**累积负面结果**（GPI-0001、G1-ALIGN-0001、G1-BOUND-0001、DIR-0003、DIR-0004，以及 CAL-0008 的饱和实测与 TH-0001 的状态维下界）下：

> **是否存在一个 任务 × 机制 × 对照 三元组，能把一个"组织操作"以可归因、可核验的方式做成项目 Gate 1 判词？若不存在，缺的到底是规则项还是证据项？**

## 2. 输入范围（只读）

`research/stage_gates.md`；`research/decisions/0002–0081`；`research/loop/current_frontier.md`；`research/negative_results/*`；最近四次设计/方案审查（`G1-ALIGN-0001-DESIGN-V2`、`G1-BOUND-0001-DESIGN`、`DIR-0003-{PREDESIGN,REDESIGN}`、`DIR-0004-PREDESIGN`）；`src/neurotopo/entry_policy.py`；`research/prior_art/{evidence_matrix,novelty_boundary}.md` 与其分线记录。审计**不训练、不生成样本、不写注册**。

## 3. 冻结的判定项（K1–K9，逐项 PASS / FAIL / UNKNOWN）

每条均可追溯到既有决定或审查，不在审计中新增标准：

| 项 | 内容 | 来源 |
|---|---|---|
| K1 余量 | 匹配的稠密强单流控制在该任务上**实测非饱和**（由控制侧资格流给出，非断言） | CAL-0008、EXP-0009、EXP-0013 |
| K2 可归因 | 存在**无架构基线**（直接按已知结构求解/高精度编码），且主对照**不是稻草人** | G1-BOUND I1、DIR-0004 I9 |
| K3 可识别 | 两臂间的容量/表征通道**有有效界**且显著小于 `δ_min`（或远小于待检效应） | G1-ALIGN S1/S2/S4、DIR-0004 I4 |
| K4 终点 | 主终点是**水平量**；项目轨不得用时间量/学习效率量 | 0062 B(B7)、DIR-0003 C 区审计、DIR-0004 I6 |
| K5 匹配 | 只声明**一条**消耗轴并经 pilot 实测 argmax 验证；`SUPPORTED` 须 matched-or-profiled compute | `G1-MATCH-v3` |
| K6 通用性 | 注册方**自缚**一个冻结的通用性谓词（`Generic` 现行无可核操作化） | 0080 §3、DIR-0003 决定0074 §5 留档 |
| K7 新颖性 | 机制内核未被先例覆盖（**正文/定理级**，非仅元数据级） | AGENTS.md、DIR-0004 I8 |
| K8 闸门功效 | 资格阶梯各级有**预注册水平/容差/功效**，可被数字否定；不得"名义通过" | G1-BOUND S4、DIR-0004 I5 |
| K9 无泄漏 | 对照不得被弱化（保留输入/输出容量与 IID 可学性）；监督不得泄漏被操纵因子 | 决定0044、DIR-0004 I5 |

**标签规则（事前冻结）**：

- ≥1 个候选 K1–K9 **全 PASS** → `CANDIDATE_EXISTS`，下一步是该候选的事前设计与独立前瞻审查。
- 仅因 K4 或 K6 等**规则项**失败（证据项全 PASS）→ `RULE_CHANGE_FIRST`，必须点名**具体待裁定规则**及其解除条件。
- 因**证据项**（K1/K2/K3/K8）失败 → `NO_REGISTERABLE_CANDIDATE_NOW`，必须写出**最小重开条件**。
- 任何 `UNKNOWN` **不得**被读作 PASS；出现 UNKNOWN 即不得给 `CANDIDATE_EXISTS`。

## 4. 我预判并必须主动检验的最强反驳

- **R1（本审计是托词）**：审计只产出标签、不产出候选，等于用元分析回避"设计一个候选"这件真工作。**检验**：审计必须给出**候选规格**（K1–K9 的可操作形式）与至少一条**可执行的下一步**，而不是止于"没有候选"。
- **R2（困境论证过度一般化）**：若审计主张"有表征内容 ⇒ 有容量通道；通道为零 ⇒ 只剩优化差"，那可能只是决定0073 单一案例（路由对齐）的过度外推。**检验**：必须给出该命题的成立范围与**反例搜索**（是否存在"表征内容大而通道可证小"的操纵）。
- **R3（候选集不完整）**：结论依赖于我枚举的候选方向。**检验**：候选集须含至少一个**未经本仓库检验的机制类别**（例如算法对齐类、精确重整化类、时间尺度层级类），并各自给出最强拒绝。

## 5. 禁止的推论

不得把任一 FAIL 读成"该方向被证伪"；不得据本审计注册 EXP、启动训练或跑资格流；不得重开非回归清单条目；不得把标签当作 Gate 1 状态变化（Gate 1 仍 `NOT_PASSED`）。

## 6. 产出

`research/evidence/DIR-0005_candidate_spec_audit.md`（审计正文 ＋ 候选三元表 ＋ 标签）与 `research/evidence/dir0005_candidate_spec_audit.json`（机器可读的 K1–K9 逐项判定）。审计完成后按仓库规程冻结 packet 送 packet-only 独立前瞻审查，再出决定记录。
