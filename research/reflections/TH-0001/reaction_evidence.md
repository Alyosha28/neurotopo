# TH-0001 任务后独立反思与主代理行动

2026-09-24。独立 `gpt-6-sol` / high / `fork_turns=none` 初始任务只给 `research/reflections/TH-0001/packet.json` 路径及审查格式，要求只读该 packet；代理报告只读此文件。packet 含事前提交的设计、数学/源码审计与决定0017，不含先前独立数学批评或后来代码补证。三项原文证据通过 `validate_review`；上下文隔离不能只凭 schema 证明。

**R1 — 接受策略范围风险。** TH-0001 只否定“等总 H 下增加 K 会**凭状态维数**让多轨迹自然胜出”，并未经验测试模型×K 交互。`research/open_questions.md` 现明确：若有独立算法/任务论证、匹配状态/参数/训练与实际推理计算、oracle-relative 主对照余量和预写交互效应，未来 K 扫描仍可注册；不得把本数学审计当成禁止所有 K 问题的结论。决定0017 的原判词不改。

**R2 — 接受 packet 的实现可核验性缺口并补证。** 独立数学批评在 packet 前已审 `RecurrentModel.forward`：每步反馈的是 H64，但在早期 query 另存8维 `pre` 读出，两个 EXP-0009 主模型相同。主代理于当前 `resource_account` 单列此缓存，没有改 `hidden_state_floats`，并在 `tests/test_core.py`/`test_source_belief_study.py` 核对4类/8类输出。`scripts/audit_gate1_state_accounts.py` 的固定输入审计另实测 EXP-0009 两臂 `pre.shape=[128,8]`、缓存8 float32/32 bytes，CAL-0001 缓存4/16，源码哈希与审计 JSON 一致。`memory_limit` 明示基准 float32 与峰值内存未测。旧冻结 EXP/CAL JSON 没有追写；当前源码不是旧 artifact 的字节级复现。

**R3 — 接受构造式对应缺少 packet 内测试。** 同一补证脚本独立以浮点乘积似然而非生成器的 log 累加实现，固定两组128例、四种 pre/final 后验的最大绝对差为 `2.9506e−8`，并核对输入中的有效证据/干扰/query 标签与来源平衡。此有限样本 sanity 不是穷尽枚举；对所有序列成立的结论来自逐项源码方程和独立数学批评，而非128例本身。补证只加强实现一致性，不产生新模型效果。

执行 `/private/tmp/neurotopo-venv/bin/python scripts/verify.py` 得121项测试通过及五模型 tiny-overfit 全通过；新 `artifacts/validation/suite.json` 源指纹对应当前 `resources.py`。验证产物中的秒数变化只是运行时间波动，预测哈希与损失数字保持原值。TH-0001 的 `CAPACITY_ARGUMENT_UNSUPPORTED`、现有 EXP/CAL 判定与 Gate 1 `NOT_PASSED` 均未改变。
