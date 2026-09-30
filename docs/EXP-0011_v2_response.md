# EXP-0011 v2 对预运行批评的回应

2026-09-24，在矩阵脚本/任务数值/模型训练前写入。v1仍存于`experiments/EXP-0011/v1.md`；v2见同目录。主代理接受全部五类阻断，未把审查当效果证据。

1. v2要求新增`pfstyle_field`工作流，资格失败以分析子状态和主`INCONCLUSIVE`正常走完整loop，不训练/创建正式模型或正式评估流；实施后须用故意失败的工程fixture验证这一条。
2. v2把未训练实例+隔离seed9_900_001的工程profile排在候选研究训练前，冻结状态/参数/算子清单、warmup5/计时30、两个长度的p50容差及`RESOURCE_UNRESOLVED/MISMATCH`停止分支；静态数仍明确为估算而非实测。
3. v2为S64_score指名现有`recurrent_mask(64,4,"random")`，为S68_sparse固定15度环和相同保度非对角交换，要求逐seed行列度、自环、连通、成功交换及哈希；工程失败不重抽。
4. v2不把额外clean-checkout重训塞进98,000更新：正式五seed从头训练已预留30,000作独立复核，另只做零更新冻结权重/输入重放。总3600秒改为preflight+worker+重放的累计阶段耗时，worker限时扣去preflight，任何总超限为`INCOMPLETE_BUDGET`。
5. v2补loss/KL方向和归约、正式主三臂学习护栏、两主差共用seed bootstrap索引与复核规则。诊断臂失败完整报告但不自动否定主包效果。v2仍未生成矩阵数组；需再获独立`READY_FOR_MATRIX_FREEZE`，生成一次数组后v3固定SHA/配置，才能进入QUESTION和实施。
