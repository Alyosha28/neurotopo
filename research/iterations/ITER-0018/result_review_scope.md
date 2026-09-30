# EXP0023v2独立结果审查范围

目标：核对六新seed的P/M/R及四臂控制、预算、旧诊断边界；主反驳是额外监督/置信度尺度/直接优化指标的循环论证，不能把方法收益当自动机制证明。
冻结commit24f5558，协议experiments/EXP-0023/v2.md。正式产物写本机APFS ~/.local/share/neurotopo/experiment-runs/EXP-0023/v2，项目归档副本内容需hash一致。独立上下文从新干净checkout，以PYTHONPATH=src和/private/tmp/neurotopo-venv/bin/python，单次<=300秒运行composition_repair_reproduce；核对15parent诊断、24最终模型、120监测checkpoint、192评估cell及24新诊断、独立XOR/BCE/秩相关/配对CI/floor/判词。不重训、替换seed或变更阈值；失败保留，不静默重试。
保存review.json、reproduction.json、reproduction_seed_metrics.json、audit_execution.json到ITER0018。复演共享forward/core，独立归约而非独立训练实现。报告每个失败seed及INV、EXTRA强控制；无Gate1/生物/全局机制结论。
