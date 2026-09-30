# GOV-0002 frontier 的独立可操作性读审

独立 `gpt-6-sol` / high，以全新上下文**仅**收到 `research/loop/current_frontier.md` 与 `research/reflections/GOV-0002/examples/project_design_only.json`；没有读其他仓库内容、运行或修改。对该标记 `SYNTHETIC_DRY_RUN_DO_NOT_TRAIN` 的例子，判：

1. **最多 QUESTION/设计审查**。frontier 明言结构入口只到 QUESTION；样例自身也是 question-only。它的因素预测文字不能证明候选可行或新颖。
2. **目前不能启动有限主控制资格训练**。样例 `max_updates=0`，实际主控制只写将来需要校准；缺具体任务菜单、独立最终流、oracle 余量方案和独立科学设计批评。
3. **不能正式注册/运行项目 Gate 1 架构效果 EXP**。还缺真实版本化更新/身份/权重/读出、相对显式滤波/PF-GRU/RIM 的可证伪数值预测和失败消融、跨时身份验证、状态/缓存/有效与分配参数/实际执行计算匹配及合格实际主控制；若换参考 K 项目轨还须另行事前裁定资格/Gate2 条件。

该审查明确的可操作性边界：仅凭这两个文件**无法**核验 JSON schema 是否实际接受样例，亦无法得知独立批评的合格数值标准或完整资格文件格式。主代理因此另外执行 `artifacts/validation/gov0002_followthrough_dryrun.json` 的四例**复制 live 账本**入口测试；它证明结构 QUESTION 分流，但不升级成科学批准。两者结合，确认 frontier 不把中间设计状态误判为正式 EXP 可运行，却未替一个真正候选完成未来资格审查。

frontier 加入任务后第4项阶段分层后，同一独立上下文仍只读更新后的 frontier 与原合成样例复审：明确区分 `QUESTION → 独立批评与有界主控制资格 → 正式效果 EXP`，并将样例固定在 QUESTION 结构演示。它继续指出，两文件本身不能代替实际 schema 校验、独立批评记录、oracle 余量数值和资源/消融审计；四例 dry-run 及独立 GOV-0002 grounded 复核分别处理了可执行入口和原历史判词核查。这是文本可读性验证，不是 SCIENTIFIC_READY。
