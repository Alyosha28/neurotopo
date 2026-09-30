# 决定0060：G1-KSI-0001 v1 独立评审与 v2 修正

接受独立 packet-only 评审 `research/reflections/G1-KSI-0001-V1/`（8 项问题**全部 accept**；其中 R6 的 `source` 标签按所引证据的实际锚点由 `both` 规范为 `action`，问题实质未变）。审查说明 `research/design/g1_ksi_v1_review_protocol.md` 于送审前提交。

## v1 判词

- **v0 缺陷诊断与生成器数学成立**：检查 A–E 通过——v0 后验只依赖样本均值（`max|Δ|≈2e-19`）；F 条件精确等于四条独立两状态时间滤波；F 中链 0 对他链流不敏感（`2e-16`）而对自身流敏感（`0.545`）；X 的联合后验与逐链边缘乘积 TV 距离 `0.95`；轨迹枚举第二实现与前向递推一致（`1.9e-15`）。
- **未达可进入数值协议的门槛**：`question_3 = not_ready`。两项规则请求均为**需收窄**（`must_be_narrowed`），不得读作已裁定。
- **检查 E 抓到一处真实实现缺陷**：修正前的前向递推在最后一步多做一次转移（得到 `z_{T+1}` 的预测分布而非 `z_T` 的滤波后验），自洽检查 B/C/D 因两实现共享同一约定错误无法发现。已修，v1 评审所依据的数值已更新为 v2 核对记录（`schema: g1_ksi_0001_math_verification_v2`）。这印证了"oracle 需第二实现"这一评审意见的必要性。

## v2 修正（逐条）

| 评审项 | 修正 |
|---|---|
| R1 资格门检测不出无作用面任务 | §4.8 新增配对 MDD ≤ `δ_min` 与分辨力检查（终点须能分辨先验已知不同的臂）；§4 末段明确资格门**不能**排除结构零交互 |
| R2 常数未承诺、`δ_min` 未定、AUC 无终端非劣 | §2.1 冻结 `T=16, β=0.8, λ=0.7, J_F=0, J_X=0.25, E=四环`；§3 新增终端损失非劣容差 `τ_term`；§5 要求效果轮前完成配对功效计算 |
| R3 对称坍缩只有诊断无补救 | §4.9 每块独立输入投影＋每块独立初始化；坍缩诊断升为**停止项**，FAIL 即停、不重调 |
| R4 难度混淆的唯一解法被降级 | §4.7 难度匹配升为**硬门**（超容差即停止）；删除并存的"难度不同"表述；§6 明确 `λ_X` 变体不得进入 primary |
| R5 检查 C 无阳性对照、字段语义不明 | 检查 C 增加自身流阳性对照并写明字段语义（`[P(z_0=+1), P(z_0=-1)]`） |
| R6 oracle 只有自洽检查 | 检查 E 新增轨迹枚举第二实现（T=4，65,536 条轨迹） |
| R7 `ln K` 标度被当作已测 | §2.3 删除标度律暗示，新增 K 不变对照 `I_flat = D̄_F − D̄_X` 并纳入决策规则 |
| R8 由 oracle 结构推出机制 | §2.3 改写为**预测**：oracle 事实只说明两条件最优计算的可分解性不同，不蕴含有限预算差异 |

## 仍未完成（不得跳过）

数值协议阶段须以数字冻结：`B, ε, δ_min, τ_term, τ_lo, τ_hi`、种子数、bootstrap、难度容差、坍缩阈值、MDD 算法；生成器与 oracle 尚未编码为可运行任务；`λ_X` 匹配变体未设计。两项规则裁定的**正式记录**仍待独立前瞻评审，本决定不代替裁定。

## 不改变

Gate 1 `NOT_PASSED`；EXP-0004/0009、CAL 系列、EXP-0011 v6、EXP-0012、EXP-0013（`INCONCLUSIVE / LEARNING_FAILURE`）的既有判词；决定0043/0044/0049/0053/0054 的约束。本决定不注册实验、不授权训练、不改变任何阶段门。

## 附：决定编号更正

同日 `research/decisions/0054_exp0013_0014_v2_revise.md` 与本决定之前已提交的 `0054_dir0002_direction_portfolio.md`（提交 `a084754`，16:08）编号冲突；前者于 16:13 加入。已将前者重编号为 `0061_exp0013_0014_v2_revise.md`（`git mv`，内容未改，仅在文件头加一行更正说明）。无文件按原文件名引用它。今后新增决定请先 `ls research/decisions/` 取下一个空号。

## 附二：并发会话遗留草稿的状态（本次一并归档）

2026-09-24 18:33–19:24 期间，另一个会话在同一路径上产出了以下**未提交**草稿，本次由接手方一并归档以便记录不丢失，并如实标注状态：

- `research/design/g1_k_structure_learning_efficiency_v0.md`：KSI v0。**其生成器存在 §1 记录的可解析缺陷，已被 v1/v2 取代**，保留作为被拒记录。
- `research/design/dir0002_rule_clarifications_review_protocol_v0.md`、`research/evidence/dir0002_rule_clarifications_action_v0.md`：两项规则裁定的审查说明与行动摘要。其 v0 评审（reviewer 标注 `GPT-6`）给出 2 项问题：**请求 B（学习效率终点类别）可接受但需收窄**；**请求 A（0049 适用性）未获通过**——缺具体矩阵对、机制论证与决策规则。v1/v2 已补矩阵与机制论证，但**正式裁定仍待独立前瞻评审**，本决定不代替裁定。
- `research/design/g1_ksi_v0_review_protocol.md`、`research/design/g1_m3_sham_intervention_method_pilot_v0.md`、`research/evidence/g1_ksi_v0_action_summary.md`：KSI v0 评审说明、D4 型匹配假干预方法试点草案与其行动摘要。**M3 试点为旧 M3 权重，无候选映射，不支持 H3 语义或 Gate 1 能力主张**（原摘要已如此限定）。
- `research/reflections/G1-KSI-0001-DRAFT*/`（含 v0、v2、v6 完整；v3–v5 仅 `packet.json`）与 `research/reflections/DIR-0002-RULE-CLARIFICATIONS-v0/`（完整）`v1–v4`（仅 `packet.json`）：**仅含 packet 的目录是未完成的反思路径，不构成评审记录**；它们不被引用为任何判词依据。
- `research/prior_art/dir0002_direction_prior_art.md` 的**补查段**由该会话追加，含三个外部来源链接。**接手方无法复核这些来源**（本会话检索通道不可用：`web_search` HTTP 402、`web_fetch` 全部非公网 IP），故该段在复核前**不得被当作已验证先例**；其"不构成新颖性结论"的自我限定有效且必须保留。
