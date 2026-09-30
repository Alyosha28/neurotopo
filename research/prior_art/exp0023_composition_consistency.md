# EXP0023组合一致性：先例、启发与边界

2026-09-28，研究实现/结果前核查。候选是输出层HOLD及FLIP²一致性辅助训练，匹配相同三次额外transition的标签监督控制；不以“组合/稳定”命名宣称新原理。

- DeepSeek mHC，https://arxiv.org/html/2512.24880v1 ，本会话已读§3–5：识别残差映射增益，约束对应矩阵并验证能力。这里借鉴“可测中间量＋定向修复”，未实现双随机残差或复现语言模型。
- DeepSeek-V3，https://arxiv.org/html/2412.19437v2 ，已读§4.5：MTP训练目标及控制；batch级辅助损失对照能获得与loss-free近似表现，提示需要区分具体机制和额外训练。没有照搬MTP或宣称本轮零额外训练成本。
- Miller/Hardt, Stable Recurrent Models，https://arxiv.org/abs/1805.10369 ，本轮读作者摘要与版本信息：稳定RNN的理论和经验研究已有先例；稳定性不自动证明更强长期记忆，本轮不把全局收缩当免费修复。
- Wang/Niepert, State-Regularized Recurrent Neural Networks，https://arxiv.org/abs/1901.08817 ，本轮读摘要：可学习有限状态转移正则、自动机与记忆任务已有研究。本轮不引入离散原型库，也不声称首次RNN状态正则。
- Shikhman, Semigroup Consistency as a Diagnostic for Learned Physics Simulators，https://arxiv.org/abs/2605.26324 ，本轮读摘要：作者报告组合误差与rollout退化的关联，同时正则效果混合。它是连续演化物理任务，不能直接迁移为本XOR操作组合的因果证据；但其“诊断有效不等于正则有效”是明确竞争解释。

本轮没有复用作者代码；commit/license/tests/maintenance为NOT_INSPECTED，未复现其数值。闭合Gram表达由现有本地REL/INV公式推导。独立旧模型诊断与六新训练seed分开，主预测采用尺度归一化D_rel，修复对照共享额外操作和标签关系；若只得到普通训练收益，不宣称独有机制或Gate1。
