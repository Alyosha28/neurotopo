# FR-0001 的 RIM 参考边界

2026-09-24。FR-0001 若继续，是**独立领域机制诊断**，不申报 NeuroTopo 项目 Gate 1、新颖架构或竞争假设轨迹。[RIMs 原论文](https://arxiv.org/abs/1909.10893)提出多个循环模块，以注意力选择相关模块更新，并允许稀疏模块间通信。[作者代码固定提交 `aa61773`](https://github.com/anirudh9119/RIMs/tree/aa61773a949c1e3f499d4c05dd77fcd2adbffa19) 的 `event_based/blocks_core.py` 用输入注意力分数、top-k mask，只保留活跃模块的新 hidden，并可用另一 attention 作模块通信；`BlockGRU.py` 包装分块 GRU。这里仅只读论文和代码，没有运行/复制作者实现；既有许可证/维护/测试审计见 `research/prior_art/evidence_matrix.md`。

拟议的两模块**连续软选择**不是 RIMs 实现复现：两个模块每 token 都计算候选 GRU 更新，然后用互补门控混合旧/新状态；不做 top-k 硬停算或模块间 attention。因此它只能测试“在相同两模块骨干和预算内，更新门控更尖锐是否有利于旧延迟绑定任务”，不能据结果声称 RIMs 完整机制或节省执行 FLOPs。旧 [EXP-0004](../evidence/gate1_benchmark_baseline_audit.md) 任务有主控制余量但可由缓存加晚期 selector 解决；若借其测出记忆收益，只能称该任务上的软选择更新效应，不证明假设专化或 SAE。理论上温度也改变梯度与优化，须作为整体干预报告而非纯粹神经选择性机制。
