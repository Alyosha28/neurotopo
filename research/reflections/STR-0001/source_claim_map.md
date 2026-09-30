# STR-0001 反思后的原文核对

2026-09-24。独立 packet 审查只能读 design/action，因此不能自行验证所引规则。主代理另核对下列原文；本核对不是对 packet 输入的事后扩充。

| 来源 | 可核原文要点 | 对结论的约束 |
|---|---|---|
| `research/design/gate1_user_charter_evidence.md:5-12` | §34 为 `Generic Multi-Trajectory Effect`；§40 要求匹配参数、状态、训练和推理预算的延迟证据修订；§3.5/26、Gate 3 另要求语义与干预；§2/9 不许从多模块单独声称新颖。 | 没有明确规定“先例骨干内新的 K 因素实验”能或不能过项目 Gate 1。 |
| `research/decisions/0006_gate1_baseline_dependency.md:5-7` | 明言“任何先例复制的正效应都不等于 NeuroTopo 候选通过 Gate 1”；当前 H3 候选仍须零训练可消融差异。 | 排除纯复制；没有定义“新的因素干预”是否仍为纯复制。 |
| `research/decisions/0012_reference_effect_vs_gate1.md:5-8` | 将严格本地参考结果记作领域证据，项目 Gate 1 排除“纯先例复制”，且要求未来改变计分先版本化。 | 保持旧规则；不能从章程语义直接把参考结果追认为项目过门。 |
| `research/decisions/0013_gate1_causal_estimand_scope.md:3-7` | PF-GRU 整套包效果不同于 K 因果效应，需 K 控制、等预算对照或资源—效果曲线。 | 说明若未来审准参考因素实验，仍不能只比 PF-GRU 与 GRU 的总包效果。 |

上述文本支持 `INCONCLUSIVE` 的**资格歧义**，不支持“现行规则一定自我循环导致停滞”或“先例因素实验一定可过项目 Gate 1”。
