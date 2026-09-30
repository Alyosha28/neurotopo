# 决定：不运行拟议的固定 d=2 严格匹配对照

CAL-0005 的事前设计提交为 `c838f3b`，原行动、独立 packet 审查及逐条修正见 `research/design/cal0005_cadence_contrast_feasibility.md`、`research/evidence/CAL-0005_cadence_contrast_feasibility.md`、`research/reflections/CAL-0005/`。这是**实验设计可行性 NO-GO**，不是训练结果或四分类科学效应判定。

现行生成器每例 `T(d)=10+8d`。在相同的1500更新、batch128、192,000条训练样本且最大训练 d=2 时，非退化混合 d0/1/2 的平均 d 必小于2；它不可能同时与固定 d2 的输入 token 和随长度变化的 dense 前向线性代理计算相等。原 CAL-0004 混合日程为3,456,000 token、97,271,808,000代理 FLOPs；固定 d2 为4,992,000 token、140,328,960,000代理 FLOPs，分别多44.44%和44.26%。减少 batch/更新、补空白循环步或允许 d>2 都改变另一项设计约束。因此标签为 `NO_GO_FOR_CLEAN_D2_CONTRAST`，原拟议两臂训练不启动。

这个结论只排除所列约束下的干净固定 d2 对照，不否定 CAL-0004 的整组训练分布工程效果，也不声称节律、最高 d 暴露或训练顺序中任何一个已被独立识别。原 action 中代理 FLOPs 增幅笔误、额外读取两个直接源码文件的范围偏离，以及越过本审计的项目障碍排序，均在反思反应中明示和修正。EXP-0009 与 Gate 1 `NOT_PASSED` 保持原判。
