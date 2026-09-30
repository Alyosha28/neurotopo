# GOV-0002 packet 反思的分阶段入口复核

2026-09-24，在构造样例或执行流程验证前保存。此处是对独立反思 R1–R5 的**事后治理诊断**，不是项目 EXP、候选机制批准或旧 Gate 1 判词修订。所有样例明确标 `SYNTHETIC_DRY_RUN_DO_NOT_TRAIN`，只在当前 ITER-0007 live ledger 的**临时副本**调用 QUESTION 入口，不对原 live 开始新迭代。

样例 A：具名但仅供测试的 `project_gate1` H3 因素合同，字段齐全、无实际主控制 headroom 测量。预期结构入口只允许 `QUESTION`；必须再经先例/独立设计批评，随后才能另立有界 CAL 校准主控制，正式效果 EXP 注册只能在真实主控制资格后。若入口把无 headroom 当已完成正式资格，或仅因为缺 headroom 而阻止设计 QUESTION，判流程错误；该样例不成为真实候选。

样例 B：以 EXP-0004 的同一任务**另问**记忆/检索、明确不声称 H3 假设语义的 `independent_benchmark` 合同，列具名操纵/匹配控制、有限训练预算、任务菜单与成功/失败/边界处理。预期只到独立 `QUESTION`；旧 EXP-0004 的 FALSIFIED 不变。若被项目 H3 候选门误挡，判流程错误。

样例 C/D：具名 `reference_k` 分别申报 `field_reference` 和 `project_gate1`，二者预算、三分支和因素同样完整。预期 C 仅进领域 `QUESTION`，D 返 `PROJECT_ELIGIBILITY_UNRESOLVED` 且不追加事件；任何项目计分仍须另立版本化归属决定。用已有 STR-0002 四例记录作交叉检查，不把样例设计当作者 PF-GRU 的实现或结果。

另列当前仓库**非 JSON** 的具名 M3/H3-SLOT/PF/RIM/H1 材料清单及五项必要字段的缺口；对 `acbe2e9` 前后 `stage_gates.md`、旧 EXP/CAL/决定文件做内容差异，保留源哈希。请独立审查者只凭 `research/loop/current_frontier.md` 和一个样例给出阶段、缺失材料、允许的控制资格工作与是否可正式注册的判断，测试 frontier 的可操作性。结果失败则修正文书/入口，不用它许可模型训练或把当前 NO_REGISTERABLE 标签扩大为科学不可能。
