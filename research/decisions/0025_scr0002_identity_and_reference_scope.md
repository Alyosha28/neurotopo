# SCR-0002 反思更正：分开索引、语义与三种 PF 参考

独立 packet-only 反思发现三个范围缺口，主代理接受；见 `research/reflections/SCR-0002/`。本文件是决定0024的**后续解释更正**，不修改其冻结设计/action/packet、`NO_DISTINCT_SPEC` 或 Gate 1。

一，权重恒等式证明的是未删索引的 log 比率可因后证据上升，**不**证明该索引里的 recurrent 内容始终代表同一假设。一个逐槽独立的 A↔B 转移即可在索引不变时改变语义。未来若要主张解释维护，必须事前定义跨时可观测身份解码和非硬编码干预；否则只叫索引权重恢复。

二，`AUTH-PFGRU`（作者每步重采样代码）、`PFGRU-R0`（未来仅移除重采样的待建对照）与 `PFSTYLE-ID0`（我们另加身份/上下文/零噪声的受限交集模板）是三个不同对象。后两者都未由原论文给出 PF-GRU 数值，也不能让交集构造承担作者架构等价的结论。逐项算身份、评分、BN、proposal、状态和读出成本。

三，旧 CAL-0001/0002 的近 oracle 数字只证明各自固定静态小后验任务在H64、1500步下达到低 posterior KL；详值和原始路径见 `research/reflections/SCR-0002/local_calibration_scope.md`。它们是辅助失败解释，不是这次 `NO_DISTINCT_SPEC` 的唯一依据，更不能泛化到未指定动态任务。未开始新 benchmark、PF-GRU 移植或架构训练。
