# EXP0017 固定控制资格失败

完整结果见`research/evidence/EXP-0017_order_observation_results.md`，决定0087。

主控制D64在CO/CU/NO的3个seed全部未通过IID<.10学习门；NU的3个seed通过学习，但OOD三seed平均.01967336低于.02，seed区间跨阈值。无cell同时合格，整体INCONCLUSIVE / LEARNING_FAILURE。

24模型完整、18000研究更新，无候选训练。保留复核器初失败及补充重放、累计复核耗时未完整记录和不完整诊断输出未留存的偏离。固定2×2效果计划关闭；不把本结果外推为非交换更新/DeltaProduct无效。
