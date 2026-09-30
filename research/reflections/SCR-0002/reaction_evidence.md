# SCR-0002 独立 packet 反思的主模型反应

2026-09-24。独立 `gpt-6-sol` / high 在 `fork_turns=none` 只收到已冻结的 design/action packet，提出三条可定位问题；三项均接受。原 packet、决定0024及 `NO_DISTINCT_SPEC` 不改，后续范围更正另写决定0025。

**SCR0002-R1 — 接受并给出反例。** 原行动的 log 比率恒等式仅追踪槽/粒子**索引 k**。`identity_counterexample.md` 用两个槽和一个逐槽独立的 A↔B 更新证明：索引一直存在且权重后来反超时，槽内容可能已代表另一解释；固定 `e_k` 若没有跨时预测一致性也只是标签。因此当前只能称“索引权重恢复”，不能称“同一假设持久恢复”。决定0025把未来候选身份的逐时可观测解码/置换/去身份干预列为进入条件；没有把这个反例升级为模型效果结果。

**SCR0002-R2 — 接受并拆开参考。** `reference_disambiguation.md` 分别命名作者原 `AUTH-PFGRU`、未来仅删除重采样调用的 `PFGRU-R0`、以及我们另加身份/上下文/零噪声的抽象 `PFSTYLE-ID0`。论文 Table 3 只有 PF-LSTM `NoResample` 消融；R0/ID0 均未在本项目实现或复现。受限交集只涉及 ID0 这类自建模板，不能充当 AUTH-PFGRU 包含 H3 或 R0 实测的证据。决定0025禁止再用一个“无重采样 PF-GRU”简称掩盖三个不同成本/算子对象。

**SCR0002-R3 — 接受并核旧产物。** `local_calibration_scope.md` 对 `gate1_belief_v1.json`、`gate1_source_belief_v1.json` 的原始键、提交、H64训练预算与三种子 posterior KL 作逐项定位：CAL-0001 为0.0005421/0.0002879/0.0003387，CAL-0002 为0.0022677/0.0018214/0.0020323 nat，均只相对各自解析 posterior 且低于原0.01天花板护栏。这不是未来 PF/固定槽实际主对照余量，也不能推广到动态任务。SCR-0002 的 NO-GO 仍直接来自候选公式与独有预测缺失。

完成上述三项后，当前 H3-SLOT 仍不得用新任务参数搜索补上规格。项目 Gate 1 保持 `NOT_PASSED`；本轮未训练任何模型、未拷贝或运行外部 PF/RIM 代码，也没有声称 H3 一般失败。
