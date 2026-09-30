# 决定：停止当前 H3-SLOT/M3 到提前 PF-GRU preflight 的分支

2026-09-24。只读当前阶段门、依赖决定、SCR-0001/PA-0001/MON-0001 的冻结证据后，经 `docs/pf_preflight_direction_independent_review.md` 独立批评，决定 **STOP_CURRENT_BRANCH**。具体停止的只是“当前 H3-SLOT 或旧 M3 没有可辨问题时，先实现 PF-GRU 定义性行为测试，希望据此救回 Gate 1”这一策略；不把它写成 PF-GRU 工程测试被 Stage Gate 绝对禁止、PF-RNN 无科学价值或所有多轨迹机制失败。

根据 `research/decisions/0006_gate1_baseline_dependency.md`，相关先例的行为资格属于已写出候选差异、实际主对照经独立余量校准后的 preflight。现在 H3-SLOT 仍 `NO_SPEC`，没有任务生成器、数值主效应与候选特异消融；EXP-0004 对旧 M3 的已登记收益为限定 FALSIFIED，EXP-0009 d4 因 C0 headroom 不足 INCONCLUSIVE，MON-0001 的旧 d1 监控也没有一致早期优势。先例核查又确认带权粒子和重采样已有直接实现。纯 PF-GRU 工程 smoke test 即使成功，也不会改变这些科学状态；失败也不能辨认机制问题与移植问题。

因此本轮**不注册、不实现、不训练 PF-GRU**，也不复制 GPL 作者代码、不把作者报告数值作为本项目效果。保持 EXP-0004、EXP-0009、SCR-0001、PA-0001 与 MON-0001 各自原范围；H3 一般命题 UNTESTED、当前规格 NO-GO、Gate 1 NOT_PASSED。若有独立、具体的软件工程阻塞需要 PF-GRU 测试，必须先版本化改变依赖顺序并登记该工程测试的输入、行为护栏、成本和停止判据；它仍不自动成为 Gate 1 效应。此决策是避免低信息实现的资源分配，不是新实验的四分类判词。
