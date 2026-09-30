# EXP0024v1独立结果复核

冻结commit94f31a9；实际运行APFS ~/.local/share/neurotopo/experiment-runs/EXP-0024/v1。新独立上下文读取协议/源码/全部原始结果，编写可归档的独立复演脚本（保存在ITER0019，执行源来自干净冻结checkout），只调用一次完整12模型校准/192cell/静态表复演，<=300秒，无optimizer更新/新原型选择。
检查source/config/parent/data/artifact哈希、监督校准均值与PSD根、模型权重不变、t0不重整、pre-refresh评分、自己预测选原型、GAUGE误差与对照资格、主病例判词及其余seed损伤。独立重算XOR/BCE/准确率/first-error/描述统计；模型forward可共用，明确这一边界。失败保留，不静默重试。交付review.json、reproduction.json、audit_execution.json与复演脚本。最强替代解释是监督二状态codebook，成功不算新算法或新seed总体证据。
