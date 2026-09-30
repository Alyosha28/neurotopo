# DIR-0001 任务后反思与主代理行动

2026-09-24。独立 `gpt-6-sol` / high / `fork_turns=none` 初始任务只给 `packet.json` 路径及固定审查格式，要求只读该文件；代理报告仅读取 `research/reflections/DIR-0001/packet.json`。首稿 I4 的 `source=action` 却引用了一句 design 原文，`validate_review` 拒绝；主代理让同一反思者只改为 `source=both`，其余科学内容不变。最终五项证据通过校验。隔离依据为调用设置/代理报告，schema 不能单独证明。

**I1 — 接受自我矛盾。** 原方向协议让完整数值搜索规格成为“是否值得写规格”的 GO 前置条件，属于门序设计错误。`research/design/h1_method_reentry_table.md` 拆成 A：判断是否值得写方法规格，B：完成数值协议及独立审查后才允许运行有界图搜索，C：能力/生物实验仍受 Gate 1；A 不要求规格已经完整。原冻结 DIR 协议不重写，NO_GO_NOW 仍由更决定性的条件②支撑。

**I2 — 接受候选状态口径。** 原设计容许“未来 null 候选去留”，行动却要求立即可运行能力实验，收缩过头。表中固定 `LOCAL_WITNESS_ONLY`、`TARGET_EXCLUDED_UNDER_STRONGER_CONSTRAINTS`、`BOUND_NONDECISIVE`、`RANGE_FEASIBLE_NOT_NULL` 和 `SEARCH_BUDGET_LOW_FEASIBILITY` 的窄含义；有限搜索失败只否定某预算/算法的可用性，不证明图空间空。BND-0001 的粗计数上界最终为 `BOUND_NONDECISIVE`，不触发完整搜索或 Gate 1 变化。

**I3 — 接受方法与训练边界。** `research/decisions/0015_h1_method_scope_correction.md` 明确：DIR-0001 只停止当下未定数值的完整互反/谱搜索；无新图的预写约束审计可有独立方法学价值，不等于生物图或新图训练获准。当前 H3 障碍不是禁止所有 H1 方法工作的普遍理由。

**I4 — 接受重审触发缺口。** 两道门的表写明可观察触发、候选状态变化和独立审查；本轮正式研究账本为已反思的 `ITER-0007 DECIDED`，正式 EXP 注册至 EXP-0009，未见新登记可运行的 Gate 1 迭代。读取 OS 进程清单的 `ps` 在 sandbox 报 `operation not permitted`，因此只据账本判断**注册队列**，不声称机器上无进程。当前 BND 上界未排除目标，故没有新图搜索触发。

**I5 — 接受较小行动并实际完成。** BND-0001 在读取结果前单独提交固定方案 `8a65f06`，只重建十张原图并计算计数必要上界，没有生成图、训练模型或读任务分数。十图上界均未排除 +0.10 目标，`artifacts/analysis/h1_reciprocity_count_bound_v1.json` 记录 `BOUND_NONDECISIVE`；这条低成本路径按预写停止规则结束，不当作图族可达证据。其独立复核和任务后反思另行归档。

这轮反思修正了主代理把“完整搜索 NO_GO”扩写成“任何 H1 方法准备 NO_GO”的策略错误；没有改变 H1/H2 UNTESTED、Gate 1 NOT_PASSED 或旧 EXP 判定。
