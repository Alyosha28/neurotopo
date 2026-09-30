# STR-0003：完整语义不必先验，但结构K因素资格未被这次审计裁定

按事前协议 `research/design/str0003_gate1_construct_scope_protocol.md`、证据 `research/evidence/STR-0003_gate1_construct_scope.md` 和独立预决策批评 `docs/STR-0003_predecision_review.md`，综合标签为 **`INCONCLUSIVE`**：用户§34/§40的Gate1任务效应不要求先完成Gate3的全部竞争假设语义干预；但“无需完整语义”**不能直接推出**任何block mask/K向量比较都隔离了多轨迹因素、可计项目Gate1。原草稿拟标更宽的 `GATE1_EFFECT_GATE3_SEMANTICS_SEPARATE`，独立批评指出遗漏了仅识别连接组织/稀疏性/读出限制的中间情形；主代理接受并收窄。初稿未提交为正式决定，未触碰旧结果。

未来每个候选须在效果前逐项证明：K条状态有何独立循环历史和更新、真实主控制缺少哪项操作、共同输出能力和读出如何保持、动态状态/有效及分配参数/训练及推理执行计算如何匹配；至少登记持久、冗余/坍塌等操作检查。若唯一改变是连接掩码，主张就只能是连接组织效果；强单流可以内部模拟多个解释是强控制能力，但不替代上述因子核查。轨迹对应具体不同潜在解释的**因果语义**仍属Gate3，先例新颖性独立；参考K项目资格依决定0019/0023继续未决。

这不是调低门槛、让旧M3过门或把用户催促当证据。EXP-0004 `FALSIFIED`、EXP-0009 `INCONCLUSIVE`、CAL-0006 `CEILING`、G1-MB/G1-DD训练前NO-GO和Gate1 `NOT_PASSED`全部保留。没有新架构训练；Gate2及后续阶段不启动。本轮只修正未来设计审查的**表述边界**，不修改代码入口或阶段门。
