# DIR-0006 C8 审计的独立前瞻审查说明（送审前提交）

2026-09-25。**不是效果证据、不是注册、不是训练许可、不改变 Gate 1 `NOT_PASSED`、不主张 H1/H2 被证实或否证。**

## 背景（最少事实）

- **C8** 是 DIR-0005 审计里"以图拓扑本身为被操纵量"的候选类别；两次 DIR-0005 审查（v0 的 I7、v1 的 I7）都指出原枚举**遗漏了它**，而它最贴近本项目使命（topology × dynamics）。DIR-0005 v2 把 C8 整行标 `NOT_EVALUATED`，决定0082 记录 C8/C9 为"下一步优先取证方向"。
- **H1 现状**：`UNTESTED`。决定0004 关闭了"从 EXP-0004/0009 回溯宣称拓扑原则"的路径，并要求重开先做三件事（成熟 null 可行性、宽范围独立操纵验证、主匹配对照余量校准＋版本化因果端点）；决定0014 判 `NO_GO_NOW`：**不写、不运行新的互反/谱联合搜索器、不装 BCTpy**，且未来 H1 问题须事前固定目标差异方向/幅度、每图样本、搜索预算、全部保度量与匹配量的数值容差、失败处理；`research/design/h1_null_family_review_criteria.md` 已把上述固化为四条评审条件。
- **本审计（`DIR-0006_c8_registerability_audit.md`）的标签**：`C8_NEEDS_NULL_DESIGN_FIRST`。其判定项 N1–N7 沿用 DIR-0005 v2 §5 的七条；其最强反驳 P1（"这只是决定0014 的改名重演"）被正面回答为：**图证据侧自 0014 以来无新东西**，新的是入口规则（决定0080 放行 `architecture_organization`，此前项目轨对图拓扑类因子是结构死路）与方法（DIR-0005 的判据装置）。

## 问题（请只回答这五个）

1. **标签是否成立**：`C8_NEEDS_NULL_DESIGN_FIRST` 是否被 N1–N7 的判定支持？特别是：① 被证明不可分离（`Q = 0.75 − cross/1024` 恒等式）与只有存在性证据（互反 ±0.002–0.004）这两半，是否足以排除 `C8_CANDIDATE_SPECIFIABLE`，又是否不足以直接给 `C8_NO_GO_UNCHANGED`？
2. **P1 的回答是否诚实**：把"自 0014 以来无新图证据"如实写出、同时指出入口规则与方法确有变化，是否构成对"改名重演"这一指控的正面处置？有无被轻描淡写或反向夸大之处？
3. **N2/N3 的 FAIL 是否成立**：可用余量任务不存在（EXP-0004 的 67.04% 按原件不能单独检验该假设，须另立注册并满足十项要求）与通道存在**已知未受控分量**（第二大特征值随交换移动 −0.0036…+0.0070 未被匹配）——这两条 FAIL 是否证据充分？
4. **下一步是否唯一且合规**：把合法动作限定为"先做 null 图族设计评审，且在通过独立审查前不写不运行任何搜索器"，是否与决定0004/0014 及 `h1_null_family_review_criteria.md` 一致？有无**遗漏的合法路径**（例如不依赖新搜索器的其他取证方式）？
5. **是否越界或过度一般化**：审计是否在任何处把 FAIL/BLOCKED 读成"图拓扑原则无用"或"H1 被否证"？是否把存在性 witness 当作效应证据？是否隐含地重启了 0014 禁止的搜索？

## 输入范围

packet 只含三段：本说明（`design`）、`DIR-0006_c8_registerability_audit.md`（`action`）、一个 verification bundle（`action.outputs[0]`），后者逐字内嵌：`h1_h2_existing_evidence_audit.md`、`h1_reciprocity_feasibility.md`、`research/design/h1_null_family_review_criteria.md`、`research/prior_art/topology_null_feasibility.md`、决定0004、决定0014、决定0016、`research/evidence/gate1_benchmark_baseline_audit.md`、`research/hypotheses.md`（H1/H2 行）。

不得浏览仓库其他内容、不得上网、不得运行代码。**引文只从 `design.md` 或 `action.md` 逐字复制**（内嵌证据是转义串，校验器不匹配其中带引号片段）。请在 `overall_assessment` 末尾附 `FILES_READ`。

## 最强拒绝（须正面处理）

- **Q1（纸上审计无信息）**：不写搜索器的约束使本审计只能做纸面判断，结论"需要先设计 null 族"可能只是把问题推给下一步。若如此，请指出审计**本可产出而未产出**的可判定内容。
- **Q2（标签过于保守）**：既然 `Q = 0.75 − cross/1024` 已证明一条路不可分离、且 0014 已判 `NO_GO_NOW`，更诚实的标签是否是 `C8_NO_GO_UNCHANGED`（即：没有新证据，保持关闭），而不是把球踢给"先设计 null 族"？
- **Q3（把入口规则变化当成新证据）**：0080 只改了入口结构、不提供证据；若审计的"新东西"仅止于此，是否等于用规则变化为重启造势？

## 判词与决策规则（事前固定）

| 判词 | 后果 |
|---|---|
| `C8_AUDIT_REJECTED` | 撤回标签，保留失败记录；C8 回到 `NOT_EVALUATED`，不得据此推进 null 族设计 |
| `C8_AUDIT_NEEDS_NARROWING` | 出修订版并重新冻结送审（一次性；若再审仍判 NEEDS_NARROWING，按主代理记录并停止） |
| `C8_AUDIT_ACCEPTED` | 出决定记录；下一步**只允许** null 图族设计评审，且**在通过独立审查前不写、不运行任何搜索器**；**不**注册实验、**不**授权训练 |

无论哪种判词：Gate 1 仍 `NOT_PASSED`；无 EXP 注册；无训练授权；非回归清单 12 条无一重开；H1/H2 仍 `UNTESTED`。
