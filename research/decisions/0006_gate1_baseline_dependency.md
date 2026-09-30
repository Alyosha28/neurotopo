# 决定：修正 Gate 1 的先例依赖顺序，保留原判据

2026-09-24。依据 `research/design/gate1_baseline_dependency_audit.md` 和独立预决策批评 `docs/gate1_dependency_independent_review.md`，决定把新 H3 候选设计前的纸面机制辨识、实际主对照资格和相关先例行为预检显式排在正式架构比较之前。`research/program.md` 已更新对应顺序。此前把“必须胜 PF-GRU/RIMs”写成 Gate 1 硬条件的草案表述被撤回；这是本轮审查发现并修复的策略错误。

Gate 1 仍按现行版本判断**状态与计算匹配后的通用多轨迹收益**。相对显式滤波、PF-GRU/RIMs 等先例的特有增量是另一个须预注册的贡献主张；任何先例复制的正效应都不等于 NeuroTopo 候选通过 Gate 1。若某候选需用 PF/RIM 才能判断新增机制，相关实现的定义性行为、粒子/路由成本与主对照 headroom 均须在正式运行前验证。是否纳入某一强基线应由具体主张决定并公开理由，不能事后挑选。

当前 H3 候选实现保持 `research/decisions/0005_h3_identifiability_gate.md` 的**条件性 NO-GO**。最先要解决的是零训练的可消融差异预测；在它出现前不移植完整 PF/RIM 训练系统，不追加 M3 任务难度搜索。EXP-0004 的限定 FALSIFIED、EXP-0009 的 INCONCLUSIVE、H1/H2 UNTESTED 与 Gate 1 NOT_PASSED 均未改变。此决策不授权适应、通信、DARC、生物图或 LM。
