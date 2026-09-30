# STR-0004：Gate 1 的“单轨迹对照”是否被要求证明功能上无内部分槽

2026-09-24，在修改任何Gate规则、重开G1-RS或启动任务训练前冻结。唯一问题：**用户章程§34/§40要求的“多条递归潜在轨迹 vs 单轨迹模型”是可实验的架构对比，还是必须先证明同容量强单流在功能上不能维持多个候选槽？** 本审计不阅读新任务效果、不重判EXP-0004/0009/0011、不以治理文字代替数值过门。

原始来源摘录（完整定位见 `research/design/gate1_user_charter_evidence.md`）：§34 Gate1“Show multi-trajectory benefit after matching state and compute. Otherwise stop.”；§40“Does maintaining multiple recurrent latent trajectories improve delayed-evidence hypothesis revision relative to single-trajectory models when parameter count, state capacity, training budget, and inference compute are controlled?”；§26强调多隐藏状态不够，要测持久、专化、冗余、坍塌和干预；§34 Gate3要求轨迹特异因果专化。对照现行 `research/stage_gates.md`、`docs/gate1_k_factor_qualification_template.md`、决定0033/0034/0035/0041/0042及各独立批评，明确哪些是用户要求，哪些是本地解释。

数学检查：若K条GRU宽h、总H=Kh、共同输入、完整读出，则H维标准GRU通过将循环权重设为block diagonal、输入/偏置按块排列，可逐步**精确模拟**K条互不通信的GRU。因此强单流有能力自发形成K槽是合理能力而非控制缺陷；任何要求其在所有数据/权重下功能上“无K槽”的条件可能与同容量强对照矛盾。反面也须查：若仅因block mask就自动叫“竞争假设轨迹”，旧M3在选中任务偶然胜随机图便可能滥过Gate1。拓扑约束改变组内/跨组边数，正差只可直接归于**显式独立递归组织的归纳偏置**，不得偷换成“候选语义/必要计算能力/新颖性”。

预写分支：

1. `ARCHITECTURAL_GATE1_TEST_ALLOWED`：若来源容许把K定义为**显式分立且持续更新的递归状态**，把单流定义为**单个未预设分槽的强GRU架构**，则控制可能内部分槽不使实验失效；但任何正Gate1判词只能是事前任务、固定预算、资源匹配下**显式K架构/独立递归组织**的有限收益。需有实际匹配稀疏及强稠密单流、实际主控制oracle相对余量、独立复核、轨迹持续/非坍塌操作护栏、输出/训练数据同等、先例归属与Gate3语义分离；只改变mask的结果必须同时报告拓扑效应，不能宣称纯粹候选语义。此分支也不自动放行已关闭的G1-RS或先例作者效果；新项目候选和数值协议仍需独立审查。
2. `FUNCTIONAL_K1_EXCLUSION_REQUIRED`：若原文或强因果理由确实要求控制在功能上不能编码多候选，指出可执行的同总状态/相同输出能力/不弱基线构造与验证标准；仅称“强连通图不能保证单轨迹”不够，因为同样问题适用于普通GRU。保持现行候选NO-GO，说明Gate1仍如何可证伪。
3. `INCONCLUSIVE`：若来源/数学不能区分，列出最小缺证，不改门、不追认任何候选。

最强反驳：放宽功能K1排除可能把**模块化/正则化**误标为多轨迹机制；旧M3同输入、无候选身份/评分，曾在有余量任务表现更差，不能仅改规则追认。审计必须逐字区分Gate1有限架构效果、Gate3候选语义、H1拓扑归因和先例新颖性；对正、负、资格失败给可执行判词，要求独立审查为何强控制不是故意弱化。若提出修订，只能前瞻版本化入口/阶段门，不降资源、效应、复核阈值，不改变旧实验和FR-0002 `FIELD_ONLY`。
