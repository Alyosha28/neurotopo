# 决定：弱候选修订行为已有直接先例，固定槽机制仅部分重叠

2026-09-24。PA-0001 的预写问题、定向一手证据和独立审查见 `research/design/h3_fixed_slot_prior_art_audit.md`、`research/prior_art/fixed_slot_hypothesis_retention.md`、`docs/pa0001_independent_review.md`。

判定：对“多个解释并存，后续观测改变当前主导解释”这一**行为**，已有 `DIRECT_OVERLAP`；对“同一个低权重、固定身份的可训练 recurrent 槽一直保留并在反证后恢复”的**完整机制**，本次只有 `PARTIAL_OVERLAP`，没有证据证明架构等价、独有性或效应。Reid 的追踪树会剪枝，PF-LSTM 的整体恢复示例不追踪同一粒子身份，Belief Net 的 HMM 状态可转移。PF-RNN 原论文虽有 PF-LSTM `NoResample` 消融，但不能充作 PF-GRU 的匹配数值对照。

科学行动是收窄新颖性边界与未来对照合同，而非实现 H3-SLOT：任何新版本要先注册能区分精确/受界滤波、PF-GRU 重采样及**另行验证的**无重采样版本、等资源单流的任务输出预测，并交代自身身份/持久化消融。当前 SCR-0001 的 `NO_SPEC` 和 H3 原始规格的条件性 NO-GO 不变；没有新训练，Gate 1 仍 NOT_PASSED。此决定不把定向检索误报为穷尽综述。
