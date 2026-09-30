# GOV-0002 阶段门与旧判词未被重写的差异审计

以事前设计提交 `065f3c7` 与 GOV-0002 行动提交 `acbe2e9` 比较：`git diff --unified=1 065f3c7 acbe2e9 -- research/stage_gates.md AGENTS.md docs/auto_research_workflow.md research/loop/current_frontier.md` 显示阶段门**仅新增一段** GOV-0002 当前组合说明，AGENTS/工作流各新增导航句，`current_frontier.md` 是新文件。Gate0–Gate9 表、EXP-0004/0009、CAL-0006 和此前各项阶段解释在该提交间没有被删改；这不代表读者会自动正确理解新段，故另由独立 frontier 审查验证。

`git diff --name-only 065f3c7 acbe2e9 -- experiments research/negative_results` 无输出；GOV-0002 没有重写任何已注册 EXP 协议或负结果文件。原 live `iteration_log.jsonl` 仍67事件、ITER-0007 DECIDED/已反思，GOV-0002 没有开启 EXP。组合标签和 frontier 仍需按 `research/evidence/GOV-0002_gate1_portfolio_entry.md` 的现存证据范围解释，不能用 SHA/差异检查当科学内容审查。
