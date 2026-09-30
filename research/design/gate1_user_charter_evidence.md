# 用户原始研究章程中与 Gate 1 解释直接相关的条款

2026-09-24。从本会话用户提供的 NeuroTopo 原始章程摘录；这是**用户指令来源**，不是论文证据或后写项目决定。只保留 GOV-0001 必需的范围，供后续独立审查核对。

1. §34：`Gate 1 — Generic Multi-Trajectory Effect`；“Show multi-trajectory benefit after matching state and compute. Otherwise stop.”
2. §40 第一正式研究问题：“Does maintaining multiple recurrent latent trajectories improve delayed-evidence hypothesis revision relative to single-trajectory models when parameter count, state capacity, training budget, and inference compute are controlled?” 同节要求只有解决此问题后才引入 adaptive inhibition。
3. §20 Phase A：先在**没有生物拓扑**时确定通用计算机制；模型顺序为 M0 GRU、M1 更宽 GRU、M2 随机稀疏循环、M3 多轨迹、M4 加适应、M5 加通信、M6 DARC 原型。
4. §21 Phase B：在引入新生物拓扑前，“reproduce or minimally validate representative mechanisms” from TRNN、RIMs、conn2res、NCP 和 CTM where practical；目标是确保框架正确表达相关基线，而非完整重现论文。
5. §3.5 RIMs：多轨迹设计须把 RIMs 作为 mandatory baseline/prior art；NeuroTopo 需用因果干预检验轨迹是否代表竞争假设、不同解释或解题路径，而非仅是独立模块。
6. §26：多个隐藏状态不足以证明功能多轨迹；要测轨迹相似/持久/解码/专化/冗余/坍塌，并做轨迹删除；与 RIM 风格模块化比较。
7. §34 Gate 3：要求轨迹特异的因果功能专化，否则不得声称“竞争假设”。
8. §2、§9：不能从多 recurrent modules、内部时间或生物 wiring 单独主张新颖性；加入新机制前要问是否已有实现、差异能否测量与消融。集成链条的新颖性本身也不能预设。
9. §18 标准研究迭代：一个精确问题 → 相关先例 → 最强拒绝理由 → 事前实验规格（假设、对照、数据、指标、种子、资源匹配与证伪条件）→ 最小实现与验证 → 按登记运行 → 不确定性/成本分析 → 对抗审查 → 四分类决定 → 仓库更新。此项为原章程结构的中文概括，不是逐字引文。

本摘录支持对不同层级要求分别解释，**不**预断“先例模型可以直接通过 Gate 1”或“Gate 1 必须先有独有机制”。这些是 GOV-0001 待裁定的命题。
