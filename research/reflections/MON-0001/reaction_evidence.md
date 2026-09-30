# MON-0001 任务后反思与主代理行动

2026-09-24。独立 `gpt-6-sol` / high / `fork_turns=none` 的初始任务只给 packet 路径及固定审查格式，要求只读该文件；代理报告只读 `research/reflections/MON-0001/packet.json`。该 packet 含事前设计、原始分析 JSON、补充输入核验和方向决定，不含先前独立科学审查正文。四项原文证据均通过 `validate_review`。上下文隔离依赖实际调用与自述，文件 schema 无法单独证明。设计文件在本会话读取指标前写入，分析 JSON 保存其 SHA；但二者在查看结果后才一起提交，因此仓库最终快照不能独立证明写入顺序。`docs/post_task_reflection.md` 已要求今后的结果敏感只读分析优先先提交设计并披露此类时间证据边界。

**R1 — 接受单标签遮蔽风险。** `qualification_axes.json` 按固定三个 step 分别列出 C0≥0.01 的操作性余量、均值方向、bootstrap 下界和≥8/10正向 seed。300步有余量但方向护栏失败；900步无余量且方向护栏失败；1500步只是描述性点，虽方向护栏为真却不能追认早期效应。保留预写 `UNINFORMATIVE_CEILING` 标签，同时单列“早期两点无一致优势”。

**R2 — 接受推论范围。** 不继续给**现有 M3/当前任务**调难度和预算，是基于原模型不表示候选语义、当前监控无一致方向、先前 EXP-0004 负效应和 EXP-0009 主终点失去余量的**资源配置决定**；不是 d4 或其他任务的样本效率否证。`research/decisions/0009_exp0009_monitor_strategy.md` 的决策只关闭“凭这些旧监控追认 M3 早期优势”的路径，未改变原实验。

**R3 — 接受条件化不确定性边界，暂不追加同一 M3 训练。** 十种子 bootstrap 条件于单一固定 d1 验证流，中间预测数组未保存，无法逐例重算或估计新流变化。`docs/mon0001_independent_review.md` 已明确这一限制；`monitor_input_audit.json` 再生1,024例流并核对哈希，只能验证输入身份和绝对预算，不能补充预测/测试流不确定性。若将来提出广义样本效率主张，必须新注册独立流与种子；当前没有此类主张。

**R4 — 接受成本口径。** 原 JSON 中的训练秒数和97,271,808,000前向线性 FLOP proxy是**完整1,500步**数值，不能当作第300/900步累计成本。早期差值仅按相同更新/共享数据进度比较。`docs/experiment_protocol.md` 现要求任何正式样本/计算效率曲线在每个 checkpoint 单独记录累计样本、tokens、实测时间与清楚限定的执行计算；本轮不追认实际计算效率效果。

补证时主代理首次把 `result.json.resources` 错当成平坦结构而触发 `KeyError: allocated_parameters`；检查真实 schema 后使用 `resources.d1` 重算，失败尝试未写入证据文件。成功的 `monitor_input_audit.json` 同时验证配置/预检 hash、再生监控流、20份结果的绝对预算/参数/状态和十对初值，全部为真。这是一次读取策略假设错误及修复，不是模型运行失败。MON-0001 和 EXP-0009 的科学范围均未扩大，Gate 1 仍 NOT_PASSED。
