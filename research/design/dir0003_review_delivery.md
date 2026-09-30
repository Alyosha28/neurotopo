# DIR-0003 送审说明（action）

2026-09-24。本文件是 DIR-0003 本轮**行动记录**，与 `research/design/dir0003_new_question_predesign_v0.md`（design）及 `research/design/dir0003_capacity_precheck.json`（verification 输出）共同构成送审 packet。**包内只有这三个文件**；审查者请只依据包内内容作答。

## 本轮实际做了什么

1. **补齐检索欠账**：仓库记录（`research/prior_art/dir0002_direction_prior_art.md`）载明定向先例检索 `NOT_EXECUTED`（`web_search` HTTP 402、`web_fetch` 全部非公网 IP）。本轮确认 **bash `curl` 可直连公网**，据此完成七区真实取回并留 search log：`research/prior_art/dir0003_A..G`。
2. **跨区综合**：`research/evidence/DIR-0003_field_methods_synthesis.md`——把六个识别通道（P1 可模拟性／P2 优化-容量不可分／P3 任务可供性／P4 对照天花板／P5 身份硬编码／P6 读出容量）映射到相邻领域的可转移装置，并据此否决两个备选问题形态（NQ-1 组织-信息等价、NQ-2 表示界兑现）后选定 NQ-3。
3. **新问题事前设计**：`research/design/dir0003_new_question_predesign_v0.md`（轨道 `independent_benchmark`，`gate1_claim = false`）。
4. **可事前冻结的解析预检**：`scripts/dir0003_capacity_precheck.py` → `research/design/dir0003_capacity_precheck.json`（`sha256 = 6d9992043d61b5aa01c66d0194b508b0ab0ec65efc5882c71da3e450d27072fc`）。核冻结、凸读出、无训练。

## 请审查者回答的问题（Q1–Q6）

- **Q1**：把**总记忆容量**匹配到 1.0000000000009 是否真的中和了容量通道？还是**谱半径不匹配**（宽带领 0.9987 对匹配窄带 0.8575）构成未处置的残余容量通道，足以单独解释预测的交叉？
- **Q2**：主终点"交叉位置 + 符号模式"是否真是**形状预测**（不可被常数因子/调参/种子吸收），还是不过是把水平比较换成两点比较？位置容差 `±2` lag 是事前合理冻结还是事后可移动的自由度？
- **Q3**：本设计是否是任何已停止方向的**改名**？请按六项逐项对照（干预、对照臂、任务族、冻结划分、预测符号、终点函数），尤其与 GPI-0001（掩码式组织）与 G1-ALIGN/G1-BOUND（路由对齐）比较。
- **Q4**：本设计自列的最强拒绝是"这只是把记忆容量测了两遍"。§10 的 **S6（任务输入分布必须与容量探针不同）+ AR(1) 任务族**是否足以把这个拒绝变成**可被数据判定**的风险？还是"容量预测任务"仍是同义反复？
- **Q5**：冻结数字是否自洽？特别是：① 用**岭化**容量泛函（`γ_rel = 0.01`）而非常规无岭容量是否恰当；② "总容量匹配"是在**探针分布（i.i.d.）**下构造的，而任务用 **AR(1)** 输入——匹配在任务分布下是否还成立？若否，判词应当如何受限？
- **Q6**：五分支是否真的互斥穷尽？`INCONCLUSIVE_UNDERPOWERED` 的三个机械触发条件（长延迟侧 `MDD > 0.014`、`T_null` 上 `|R²| > 0.01`、Gram 条件数 `> 1e8`）是否可达且不会吞掉其他分支？还有没有**未注册的自由度**（评估流数量、`k` 档选取、容差、`δ_task`、`Δ_eq` 的来源）？

## 明确未授权事项（请在判词中保留）

不注册实验、不训练模型、不生成任务样本、不改变 Gate 1 `NOT_PASSED`、不重开非回归清单 12 条中的任何一条、不主张新颖性、不主张 H1/H2 证据、不主张能力或生物学结论。本设计自认定位在因果链的 **Representation** 环节，**不能**替代项目轨 Gate 1 需求。
