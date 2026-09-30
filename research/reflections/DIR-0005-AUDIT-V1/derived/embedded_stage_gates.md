| Gate | Requirement | State |
|---|---|---|
| 0 | Locked environment, registry, runner, tests, resource accounts, prior art | PASSED; 177 tests + all-model overfit at EXP-0011 engineering freeze; EXP-0004 frozen-commit seed9101 all5 models exactly reproduced |
| 1 | Generic trajectory benefit after matching the declared consumed axis (compute matched-or-profiled for SUPPORTED; all other axes disclosed as measured ratios) | NOT_PASSED; EXP-0004 primary prediction FALSIFIED; independently reviewed |
| 2 | Adaptation exceeds noise/compute controls | NOT_STARTED |
| 3 | Functional hypothesis-specific causal specialization | NOT_STARTED |
| 4 | SAE without relevance collapse | NOT_STARTED |
| 5 | Biological topology exceeds strong matched nulls | NOT_STARTED |
| 6 | Compact synthetic principles recover effect | NOT_STARTED |
| 7 | Transfer to controlled natural sequence tasks | NOT_STARTED |
| 8 | Independent LM | NOT_STARTED |
| 9 | Scaling after replication | NOT_STARTED |

No adaptation, communication, DARC, biological graph or LM is authorized by evidence before its prerequisite. Failure to establish Gate 1 keeps the program here; do not add mechanisms to rescue a negative comparison.

DIR-0002 (`research/decisions/0054_dir0002_direction_portfolio.md`) audited the current direction portfolio read-only under the active entry policy and decisions 0043/0049/0053. Its corrected label is `RULE_CHANGE_FIRST`: no direction is registerable before two prospective rulings — whether a structure-mismatch (cross-coupled) condition counts as "another task family" under decision0049, and whether Gate 1 permits a learning-efficiency endpoint (updates-to-ε / curve area) as primary. It also records that TH-0001 blocks manufacturing a capacity separation by making the task harder at matched total state, and that the targeted prior-art search could not run (retrieval unavailable), leaving D3/D4 unassessable. No experiment was registered, no gate state changed; Gate 1 remains `NOT_PASSED`.

**Gate 1 匹配口径：判据版本 `G1-MATCH-v3`（决定0069，2026-09-24 生效；取代 `G1-MATCH-v2`/决定0068）。** **单层：无通用/严层之分，无自动降级，无数值上限。**

- **必须匹配的只有一条轴**：机制实际消耗的那条轴，注册时声明，并经 **pilot 实测的超出消耗 argmax 验证**。
- **`state` 匹配不再是必须的**（此为对 v2 的实质放宽）：`state` 与其余各轴（分配/有效参数、样本、更新、内部步数、时延）一律在 manifest 中按**实测比值披露**，匹配与否均须披露。
- **取消数值上限与超限自动降级**：v2 的默认 `1.15×` 上限与降级机制作废，代之以披露。
- **`compute`**：给 `SUPPORTED` 判词须 **matched-or-operator-profiled**（保留 v2 的收紧，不退回"仅披露"）。
- **时延**：受计费报告指标；**除非机制把时延声明为消耗轴，否则不单独阻断 Gate 1 判词**。
- **判词仅就所声明的轴条件化成立**；`state` 未匹配的比较**不得**被读作分离出机制的组织结构。
- 整包比较仍走带**资源未匹配标志**的 `PACKAGE_EFFECT` 轨，**禁止聚合或作为 Gate 1 证据引用**。
- 决定0067 §5 的**日落条款继续有效**。

**变更日志（决定0069）**：**本次为净放宽**，五项动作——(1) `state` 由"必须匹配"改为"披露"；(2) 取消各轴数值上限（默认 `1.15×`）；(3) 取消超限自动降级；(4) 两层并为单层，严层不可达时的"不可测声明"义务随之作废；(5) 时延不再单独阻断判词。**同时保留**"声明消耗轴须匹配（经 pilot argmax 验证）"与"`SUPPORTED` 须 matched-or-profiled compute"。Gate 1 要求文本已**同批**改为 `Generic trajectory benefit after matching the declared consumed axis`，以消除 0067/0068 期间"行文点名 `state` 而判据不要求 `state`"的**同一类不一致**。

**代价已记录（不得隐去）**：`state` 未匹配的比较**不能再被读作**分离出机制的组织结构；v3 下判词必须条件化；**v3 与 v2 科学上不等价，v3 更弱**。本决定**不是**"澄清"或"等价改写"。

**生效依据与审查状态（必须与判据本身分开读）**：本判据由**用户明确直接授权**生效（"别审查了，我允许你放宽"），**独立前瞻审查被豁免、未执行**——已冻结送审 packet（`research/reflections/DECISION-0069/packet.json`）但**无** `review.json`/`reaction.json`，**未运行** `reflection check`。故本判据是**经用户授权的规则变更**，而非**经独立审查确认的规则变更**；不得被引述为"已通过审查"。

**追溯性**：0068 之后**零注册**（`HEAD == 427f7f5`，其后零提交，`experiments/` 与 `research/entries/` 无变更）→ **无需追溯改判**。早于 v3 的、以 Gate 1 为目的的注册按 v1/v2 判据裁决或正式重新注册，**不得**按 v3 追溯。**非回归清单**（候选级 9 条 + 任务/组合级 3 条）**全部保持被否、无一重开**；放宽**不构成**重开理由，重开须走新的注册。每个注册须存储 `matching_criterion_version`。**决定0068 保留在库**并已标注为被本决定取代（其全文见 `research/decisions/0068_gate1_matching_clause.md`）。Gate 1 状态仍 `NOT_PASSED`，无 EXP 注册，既有判词全部不变。
