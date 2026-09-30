# EXP-0010 v1：项目优先级 NO-GO，未进入 loop 或训练

依据事前草案 `d0c2da2`、独立预运行批评 `docs/EXP-0010_pre_run_review.md` 和四分支决策价值表 `research/evidence/EXP-0010_pre_run_decision_value.md`，决定 **`CLOSED_BEFORE_RUN_LOW_GATE1_VALUE`**。`research/entries/EXP-0010.json` 结构校验曾返回可进入 `independent_benchmark` QUESTION，但 `scientific_approval=false`；未调用 `ResearchLoop.begin`，未实现新模型，未读任何 EXP-0010 资格或正式效果，未产生结果。EXP-0010 的编号/草案保留以防后来静默重用或追认。

这是一项**研究排序**而非领域因素的 FALSIFIED：同 K2 温度对照无论正负都不能回答 Gate 1 的多轨迹相对强单流主问题。原 `FACTOR_SPEC_READY_FOR_REGISTRATION` 只承认可写成独立协议；本决定基于现在明确的 Gate 1 优先目标，终止实施。独立审查还发现 runner 缺两阶段资格闸门、效应措辞/CI不一致、固定题目流推断范围与3600/1800秒时限不匹配；因此即便另有领域优先级，也不能直接运行 v1。

保留 EXP-0004 `FALSIFIED`、EXP-0009 `INCONCLUSIVE` 与 Gate 1 `NOT_PASSED`。下一项可实施的项目方向必须在新事前设计中把 K/候选因素、单流主对照、状态/参数/执行计算、先例归属和真实任务余量联在一起；这不允许降低阶段门，也不许可适应、通信、DARC、生物拓扑或 LM。
