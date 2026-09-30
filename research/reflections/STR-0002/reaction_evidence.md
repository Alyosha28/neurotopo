# STR-0002 packet-only 反思后的主模型反应

2026-09-24。独立 `gpt-6-sol` / high 在 `fork_turns=none` 只读 design/action packet，提出五个问题。五项均接受为**范围或运行入口**缺口；正式 `CANDIDATE_FIRST_FOR_GATE1` 只限未来项目 Gate 1 动机的资格训练，旧 CAL/EXP 与 Gate 1 `NOT_PASSED` 不改。原 packet、独立审查和事前 STR-0002 设计不改写。

**R1 — 领域筛查可改变自身决定，接受。** `research/reflections/STR-0002/decision_value_costs.md` 分开了“独立 benchmark 的任务资格决定”和“项目 Gate 1 架构效果决定”：CAL-0006 的成败本来有前瞻、有限的领域用途；缺因素时只是不能据此注册项目效果。决定0023 与新 `independent_benchmark` track 保留该路径，未把任务筛查一律判低信息。

**R2 — 参考 K 因子项目归属未决，接受。** `src/neurotopo/entry_policy.py` 对 project_gate1 缺因素返回 `MISSING_FACTOR`；具名 `reference_k` 试图直接计项目 Gate 1 时返回 `PROJECT_ELIGIBILITY_UNRESOLVED`，可在 `field_reference` 轨继续而 `gate1_claim=false`。合同同时要求操作、失败消融、方向预测、实际主控制以及分别列出的状态/参数/执行计算计划。是否公平、是否相对 PF/RIM 有独立意义仍需 `PRIOR_ART/CRITIQUE`，不能靠非空文本判定；项目归属只能在相关架构效果结果前另立版本化决定。四例真实 QUESTION dry-run 的输入哈希、拒绝码和事件数见 `artifacts/validation/str0002_entry_dryrun.json`。

**R3 — 条款存在不足以证明入口运行，接受。** 运行入口策略 `gate1_candidate_first_v1` 由 `ResearchLoop.enable_entry_policy` 对当前 live 账本显式启用；以后的 `begin` 缺合同会在 QUESTION 前拒绝，有合同则保存同一校验字节的内容寻址快照，结构通过也仅进入 QUESTION。四例 dry-run：缺候选项目和项目参考 K 分别拒绝且事件数不增；独立 benchmark 与领域参考 K 分别只增一个 QUESTION 事件。`artifacts/validation/str0002_live_migration_rehearsal.json` 在**当前**66事件账本副本中验证只追加 policy 事件、不改 ITER7/DECIDED/反思/方向；独立代码审查核验后才在 live 追加 event-000067。`artifacts/validation/str0002_live_activation.json` 再核对原66事件 SHA、当前67事件、新策略、一次无合同 begin exit2/无新事件。全量141项测试通过。入口校验只保证申报结构、内容快照和 claim 分轨，不证明科学语义、不能防范故意填充 boilerplate；后续 CRITIQUE 仍必需。

**R4 — 行动排序成本未申明，接受。** `decision_value_costs.md` 用零训练候选合同、已冻结 EXP-0004 只读再分析及 CAL-0006 实际921,600按臂样本/7200 arm-updates/12.67秒 staging 成本作受界比较；人力未量化，不伪造概率性 EIG。这个排序只对当前**项目 Gate 1**缺候选/控制成立，候选合同本身可能 NO_SPEC，独立 benchmark 仍可有自己的可改变决定。

**R5 — 候选优先可能变候选偏好任务搜索，接受。** `docs/experiment_protocol.md` 现在要求候选驱动筛查前固定有限任务菜单、最大尝试次数、保留全部失败/边界配置及未参与选择的最终任务/种子；仅选中配置有利的效应须降级。结构合同强制申报 `task_search.max_configurations/retain_all_attempts/heldout_rule`，但实质冻结与信息泄漏仍由独立审查把关。不会用新入口追认 EXP-0004/0009 或改 CAL-0006 阈值。

运行入口实现与恢复边界另见 `docs/STR-0002_entry_code_review.md`。源脚本在静态审查后曾被单元测试发现 `.resolve()` 跟随末端 symlink 的快照漏洞，已修正并由测试证明拒绝；这是“反思后反应”必须覆盖**实际运行**而不能只读政策文字的直接例子。
