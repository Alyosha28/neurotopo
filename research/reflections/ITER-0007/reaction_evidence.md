# ITER-0007 后反思的核验与行动

2026-09-24。独立 `gpt-6-sol` / high 以 `fork_turns=none` 仅收到已登记设计、完整分析和完成状态的 packet。`review.json` 三项证据引句通过 `neurotopo.reflection` 的逐字来源和 packet 哈希校验。主代理核查如下；反思意见本身不是额外训练种子或科学投票。

**R1 — 接受，属于我的基线校准策略失败。** CAL-0004 只用稠密 GRU 验证 varied 日程能修复 d=4 地板。EXP-0009 却以**实际主对照 C0** 为分母要求 >0.05 nat 改善；C0 正式 d=4 KL 只有0.012463，按 KL 非负性根本没有0.05余量。运行前的资源预检没有检查该效应量可行性，独立预运行批评也未收到 CAL-0004 数值摘要，因此未暴露这个风险。行动：在 `benchmark-design` skill 与 `docs/experiment_protocol.md` 增加主匹配对照的独立 oracle-relative 校准；`research/decisions/0003_exp0009_headroom_and_h3.md` 停止沿当前 M3 代理调难度寻找正结果。没有把 d=8 提升为主终点或重训后改阈值。

**R2 — 接受标签表达的局限，不改冻结判词。** 预注册判定顺序要求 headroom 先通过，因此本轮为 INCONCLUSIVE。十种子配对区间上界+0.000120、以及 C0 平均 KL0.012463 的数学上界，仍限制了**此固定测试流**可展示的 >0.05 效应。`decision_table_audit.md` 将资格轴、固定测试流上的效应界限和跨基准结构推论并列呈现；`research/evidence/exp0009_posterior_transfer.md` 保留原数值与头部失败。不能据区间上界越过先决条件改成广义 FALSIFIED。

**R3 — 接受复现/外推范围。** 独立科学审查逐题重算全部30模型×3条件的预测指标并复核来源、掩码和资源；干净 checkout 的端到端训练复现只覆盖一对 seed9401 的三模型。配对 CI 的十个重复单位是训练/图种子，并条件于一个固定最终测试流。`docs/EXP-0009_reproduction.md` 与 `research/iterations/ITER-0007/review.md` 均逐项声明这些边界。由于当前登记任务已因对照天花板失去判别力，重跑其余九个种子或事后更换测试流不改变当前判定，暂不把算力投入同一设计的追加确认。

三项处理均保留 EXP-0009 的 INCONCLUSIVE 与 Gate 1 的 NOT_PASSED；主代理完成的是方法与新颖性边界修正，而非结果修饰。
