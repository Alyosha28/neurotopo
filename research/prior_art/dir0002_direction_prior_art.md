# DIR-0002 定向先例核查：首次检索未执行，后续补查见文末

2026-09-24。本文件与 `research/evidence/DIR-0002_direction_options_survey.md` 的研究判定**分开留档**。

## 首次执行状态：`PRIOR_ART_CHECK_NOT_EXECUTED`

按 `research/design/dir0002_direction_options_predesign.md` 的计划，本应针对 D1–D4 做受限先例检索。**实际未能执行**：

- `web_search`：三次尝试全部失败，`HTTP 402 Insufficient Balance`（endpoint `api.deepseek.com/anthropic/v1/messages`）。
- `web_fetch`：所试每个主机（arxiv.org、export.arxiv.org、example.com、google.com、api.semanticscholar.org、openreview.net、r.jina.ai）均返回 *"resolves to a non-public IP address"*，DNS/出口被黑洞。
- 委托的独立检索子代理报告同样的两类失败，**零来源被取回**。

因此：**本次没有任何外部文献被检索、阅读或核对。** 未验证的模型回忆**不得**作为引用、重叠判词或新颖性依据，本文件也不记录其标识符。

## 可用的已审内部先例边界（本次实际依据）

仓库内已有经审计的先例材料，本次只读并据其判定：

- `research/prior_art/novelty_boundary.md`：已核重叠轴表、"可谨慎检验的窄问题"三条、"不安全的宽泛主张"清单。
- `research/prior_art/evidence_matrix.md`：TRNN、RIMs、PF-RNN、EviTrack、conn2res、果蝇连接组 reservoir、flyvis、CTM、ncps/NCP、BPU、层级模块 reservoir（Milisav 等）、Reid MHT、Belief Net、Namikawa–Tani 专家、MH-iSAM2、Unified PF-LSTM 等；均标注"未复现"。

按这些已审材料的直接后果：

| 方向 | 已审先例覆盖 | 对本调查的约束 |
|---|---|---|
| D1 结构条件化组织 | **机制已覆盖**：RIMs 的多个近独立 recurrent 机制；层级模块 reservoir 已报告"多模块→时间尺度池→记忆能力"；`novelty_boundary.md` 已把"模块化循环网络能提升记忆/时间建模"列入**不安全主张**。 | D1 只能以**条件化/交互**形式提出（K 剂量 × 任务结构，带符号预测与斜率对比），且不得声称"模块化有效"或新颖性。其**交互估计对象**在已审集合中未见记载，但这只是"本次审计未见"，不是新颖性结论。 |
| D2 学习效率终点 | **机制已覆盖**（PF-RNN 等）；**方法学文献不在已审集合内**。 | D2 的机制无新意；其终点选择属方法学。按未验证的回忆，预算匹配的逐臂调参与曲线/AUC 终点是既有最佳实践，因此 D2 应表述为**采用既有方法学**，而非发现。该回忆**未核**，须在检索恢复后补核。 |
| D3 交换性参数绑定 | **未覆盖**：等变/绑定槽、任务相关性×共享收益的文献不在已审集合内。 | 重叠风险未知且可能高（多任务学习中"共享收益取决于任务相关性"属经典结论）。在补核前不得进入设计批评。 |
| D4 匹配假干预 | **未覆盖**：可解释性文献（匹配随机消融对照、预测误差模式、激活修补）不在已审集合内。 | 组件级重叠风险未知。`novelty_boundary.md` 的"模块分组不等于独立假设语义"与本方向互补，但不构成先例核查。 |

## 待补核查清单（检索恢复后逐条执行）

1. D1：模块/近独立循环机制文献中是否已有**预注册的 K 剂量 × 任务结构交互**，含对 K-斜率的带符号预测与资源匹配稠密对照；以及"交叉耦合潜变量使模块化失效"是否已被当作受控条件报告。
2. D2：预算匹配逐臂调参、updates-to-ε/曲线面积作为主终点的既有方法学工作；以及是否已有工作明确把"控制臂最终达到精确后验"作为**入选条件**而非取消资格。
3. D3：等变/绑定槽更新与"任务相关性→共享收益"交互的既有定量标度（尤其等参数总量下 tying 使每槽参数减少 K 倍时的正则化解释）。
4. D4：推理时单元/模块干预＋试次级预注册误差签名＋等幅度随机位置 sham 的组合是否已作为协议发表。
5. 上述四项均须记录固定提交/版本、许可与是否复现，遵循 `evidence_matrix.md` 的证据等级约定。

## 2026-09-24 后续定向检索：部分完成，不构成新颖性结论

先前 `NOT_EXECUTED` 记录准确描述了当时的检索失败。本次在用户提出 K×结构学习效率与匹配假干预后，使用公开网页搜索做了有限补查；以下为作者报告的论文结论／方法，不是本地复现，也没有检查作者代码、固定提交、代码许可或测试。

### 检索范围与直接来源

- 查询：模块化 RNN、结构模块化与功能专化、受控任务结构及模块干预。直接来源：[Dynamics of specialization in neural modules under resource constraints](https://www.nature.com/articles/s41467-024-55188-9)，Nature Communications (2024)。网页正文摘要与实验设计显示作者在受控 RNN 任务中比较模块规模、模块间连接和输入路径；作者报告结构模块化本身不保证功能专化，并以 module probing、module ablation 和 hidden-state correlations 测量模块功能。该结果覆盖“模块结构／功能分工及模块消融”这一宽方法族，**未核实**是否已有与本项目完全相同的 `K × coupling-structure` 学习曲线斜率交互或等幅随机位置 sham。
- 查询：activation patching 的指标、干预和方法学稳健性。直接来源：[Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042)，arXiv:2309.16042 (2023)。作者报告不同指标与 corruption/intervention 选择会产生不同定位结论。它支持在分析前冻结干预算子、误差签名与指标；模型类别和任务不同，**不能**据此判定 NeuroTopo 的匹配 sham 方案已被完整覆盖。
- 查询：group recurrent networks、组间相关性与实测运行时。直接来源：[Efficient Sequence Learning with Group Recurrent Networks](https://www.microsoft.com/en-us/research/uploads/prod/2018/11/N18-1073.pdf)，Microsoft Research 托管的论文 PDF。论文描述独立组递归及 representation rearrangement，并报告朴素逐组循环的运行时会受额外开销和并行度影响。它构成分组递归的一般先例，也说明相同矩阵形状／kernel 是减小时延差异的设计办法，不能替代实测。

### 对本方向的有限影响

- D1 不得声称“模块化 RNN 提升学习／记忆”或“使用模块干预”本身新颖。可审问题收窄为：在预先冻结的两类潜变量耦合结构中，资源匹配的 K 剂量效应是否出现有符号的学习效率斜率交互。本次搜索未找到该确切交互的直接来源，但搜索范围有限，**不构成未有人做过的证据**。
- D4 的一般内部激活干预、模块消融与功能测量已有先例。具体的“校准流锁定块／测试流复现误差签名＋等扰动范数随机位置 sham”组合仍未完成针对性覆盖；它只能作为应用到旧 M3 的方法学试点，不可声称方法创新或候选语义证据。
- 未检查任何代码仓库、提交、license 或维护状态；没有复现任何论文结果。D1/D4 的完整 prior-art 审查仍未完成。若未来要作新颖性／增量贡献主张，需按 `prior-art-audit` 规则完成来源身份、源码、许可及与最近先例的逐项对照。

## 2026-09-24 追加：D1／D4定向来源与代码核查（有限范围）

按本轮新设计问题，使用题名／别称／精确机制词搜索：`module specialization covariance resource constraints RNN ablation`、`matched-random component ablation per-example damage`、`random-position activation intervention`。以下是一次**有边界的补查**，不把搜索未命中当作新颖性证明。审阅日期为2026-09-24；未运行外部代码，也未复现作者结果。

### D1：模块递归 × 环境结构／学习效率

- **直接论文先例**：[Dynamics of specialization in neural modules under resource constraints](https://www.nature.com/articles/s41467-024-55188-9)，Nature Communications。作者报告在受控 recurrent architecture 中共同改变模块规模、模块间连接和输入／输出路径，并改变两个任务变量的 covariance；模块结构本身不保证功能专化，研究也含 module ablation。该结果与“任务结构依赖的模块组织效应”直接重叠，因此 D1 不得声称“模块化有利于专化”或“任务相关性 × 模块结构交互”本身是新机制。
- **作者源码检查**：[specialization-dynamics](https://github.com/GabrielBena/specialization-dynamics/tree/520dd88183a6fcf51ae6995956829ea10c2d906c)，commit `520dd88183a6fcf51ae6995956829ea10c2d906c`（2024-02-26）。检查 [`dynspec/datasets.py`](https://github.com/GabrielBena/specialization-dynamics/blob/520dd88183a6fcf51ae6995956829ea10c2d906c/dynspec/datasets.py) 的 `DoubleDataset.cov_ratio`（第206–285行：改变两个目标相同的概率）、[`dynspec/experiment.py`](https://github.com/GabrielBena/specialization-dynamics/blob/520dd88183a6fcf51ae6995956829ea10c2d906c/dynspec/experiment.py) 的 module-ablation 汇总（第343–375行）与 [`dynspec/retraining.py`](https://github.com/GabrielBena/specialization-dynamics/blob/520dd88183a6fcf51ae6995956829ea10c2d906c/dynspec/retraining.py) 的 masked-readout ablation（第190–231行）。这确认源码确有任务相关性操纵与模块贡献评估；未运行源码。该快照未见 `LICENSE` 文件或 pyproject license 字段，也未见专用 `tests/` 目录，故不建议复用代码。
- **与当前草案的差异边界**：论文已研究环境变量相关性、结构／资源约束与功能专化。此次有限搜索未在该来源中找到固定四潜变量、解析后验监督、`K={2,4}` 的有限学习曲线斜率、随机稀疏实际主控制及交叉耦合交互这一完整组合。此差异只界定可审 estimand，**不是新颖性证明**；gate 资格仍由本地阶段规则决定。

### D4：内部干预、任务误差签名与 matched-random sham

- **直接方法先例**：[Closure-Validated Circuit Discovery in Attention Heads: Co-activation Proposes, Ablation Disposes](https://arxiv.org/abs/2606.09607)，2026预印本及[作者配套源码](https://github.com/skydancerosel/coactivation-closure/tree/7e48183c0ee6f97c0d720337e1b50cd37880049e)。本次检查 README 与固定提交的 [`closure/closure_test_pythia_cluster4.py`](https://github.com/skydancerosel/coactivation-closure/blob/7e48183c0ee6f97c0d720337e1b50cd37880049e/closure/closure_test_pythia_cluster4.py)：作者逐样本测组件集消融的 loss／accuracy／目标 logit 变化，并比较同规模随机组件组；代码对候选头切片执行置零，再从匹配层集合抽取5组随机头。作者仓库标 MIT、7 commits；快照未见专用测试目录，未运行代码。它与“结构化组件干预＋随机位置/组件对照＋逐样本误差测量”有**直接方法重叠**，因此本地 sham 设计只能称已知干预方法在 recurrent posterior 模型上的受限应用。
- **具体差异**：该先例操作 Transformer attention-head output projection，随机控制匹配组件数量／层集合，未匹配同一个状态扰动向量的逐样本 L2 范数；本草案拟操作冻结 M3 recurrent hidden state，在新流上比较 H/R posterior error signature。这个具体对象差异不使一般干预法新颖，也不形成 H3 候选语义。
- **测量选择警示**：[Towards Best Practices of Activation Patching in Language Models](https://arxiv.org/abs/2309.16042)，Zhang & Nanda，ICLR 2024。作者报告 corruption/intervention 与 metric 选择可改变定位结果。本次核对 arXiv 论文页和 OpenReview 论文文本；arXiv 元数据页没有给出作者代码链接，未找到可固定源码仓库、许可或测试，未复现。

### 受限结论与证据等级

- D1 的“任务结构调节模块效应”已有直接先例；可审的是对后验学习效率这一新 estimand 的受限本地扩展，且不主张该交互新颖。
- D4 的组件消融、matched-random 控制和逐样本损伤签名均有直接先例；旧 M3 试点是方法应用，不是方法贡献，也不是候选语义证据。
- 证据等级为**论文作者报告＋指定源码文件检查**，无本地复现。`specialization-dynamics` 与 `coactivation-closure` 源码只在 `/private/tmp/` 审阅，未复制、依赖或执行。
- D1/D4 的确切组合仍非全领域穷尽检索；因此不作“无人做过”结论。D3 参数绑定／等变先例仍未审。Gate 1 两项规则问题仍由决定0054所要求的前瞻审议处理。

## 禁止推论与当前覆盖边界

本文件不构成新颖性结论，不构成"无人做过"，也不得被用来说明任何方向已被先例排除。开头 `PRIOR_ART_CHECK_NOT_EXECUTED` 与原 §5 禁止条款描述第一次检索失败时的状态；后续 D1/D4 补查更新了当前可进入的审查范围，但没有覆盖全领域。

- D1 可进行**有限 estimand 审查**：必须显式对照已有的任务变量 covariance × 模块结构／资源研究；不得声称模块专化或结构依赖效应本身新颖。任何更窄的交互增量仍需具体来源对照。
- D4 可进行**有限方法设计审查**：matched-random component ablation 与逐样本损伤签名已有直接先例。本地方案只能定位为已知方法在 recurrent posterior checkpoints 上的应用；等幅向量随机移位的精确组合尚未穷尽检索，也不能据搜索未命中主张新颖。
- D3 的参数绑定／等变与任务相关性先例仍未审，故仍不进入设计批评。D2 的学习效率测量属于已知方法学问题，本轮没有完成专项方法先例审查。
- D1/D4 的完整机制贡献主张在补足与最近来源的精确比较前均不成立；当前草案也不授权注册、实现或训练。
