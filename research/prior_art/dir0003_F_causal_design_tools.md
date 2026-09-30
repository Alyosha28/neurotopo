# dir0003_F 机制归因的实验设计与因果推断工具箱

**审计日期**：2026-09-24（UTC）
**审计类型**：文献方法审计（design patterns），非综述、不产出新颖性结论。
**证据等级声明**：**无本地复现**。本文件不含任何一次训练、评估或代码运行。所有内容均为**外部文献取回**，逐条标注核验层次：`pdf_section`（读到原文正文段落）／`abstract`（读到摘要原文）／`metadata`（仅从索引取回标题+年+场所+标识符，未读摘要或正文）。**凡未在本轮实际取回者，一律不写入。**
**检索边界（引擎 × 查询词）**：完整机器可读清单见同目录 `dir0003_F_search_log.json`。引擎实际使用：`crossref`、`europepmc`、`huggingface-papers`、`arxiv`（部分查询 0 命中 / 后段 HTTP 429）、`openalex`（**1 次即额度耗尽，0 取回，已按上级指示停用**）。起始查询词：`negative control outcome unmeasured confounding`、`equivalence testing two one-sided tests TOST`、`factorial design mediation analysis`、`randomization of the mediator`、`principal stratification`、`placebo test design`、`instrumental variable mechanism`、`design-based identification`、`registered report pre-registration decision rules`、`significant absence of evidence`、`non-inferiority margin`。扩充查询词见 search_log。
**取回规模**：**33 个互不相同的来源**；其中 `pdf_section` 5 条（S02、S10、S27、S28、S29）、`abstract` 8 条（S01、S04、S07、S08、S09、S11、S12、S14）、`metadata` 20 条。
**申报限制（重要）**：`metadata` 层来源我**没有读过摘要或正文**，其内容推断仅基于标题与场所，可靠性低于前两层。搜索未命中 ≠ 无人做过。§4 逐条列出未核验项。

---

## §1 来源表

| # | 逐字标题 | 年 | 场所 | URL / 标识符 | 核验层次 |
|---|---|---|---|---|---|
| S01 | Negative controls: a tool for detecting confounding and bias in observational studies. | 2010 | Epidemiology | doi:10.1097/ede.0b013e3181d61eeb | abstract |
| S02 | Negative controls and how to use them: a guidance paper. | 2026 | International Journal of Epidemiology | doi:10.1093/ije/dyag163 · PMC13525841 | **pdf_section** |
| S03 | Using negative controls to adjust for unmeasured confounding bias in time series studies | 2023 | Nature Reviews Methods Primers | doi:10.1038/s43586-023-00259-2 | metadata |
| S04 | Using Negative Control Populations to Assess Unmeasured Confounding and Direct Effects | 2024 | Epidemiology | doi:10.1097/EDE.0000000000001724 | abstract |
| S05 | An Introduction to Proximal Causal Learning | 2020 | medRxiv（预印本） | doi:10.1101/2020.09.21.20198762 | metadata |
| S06 | Regression-Based Proximal Causal Inference. | 2024 | American Journal of Epidemiology | doi:10.1093/aje/kwae370 | metadata |
| S07 | A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. | 1987 | Journal of Pharmacokinetics and Biopharmaceutics | doi:10.1007/bf01068419 | abstract |
| S08 | Equivalence Tests: A Practical Primer for t-Tests, Correlations, and Meta-Analyses | 2017 | Social Psychological and Personality Science | doi:10.1177/1948550617697177 | abstract |
| S09 | An Equivalence Approach to Balance and Placebo Tests | 2018 | American Journal of Political Science | doi:10.1111/ajps.12387 | abstract |
| S10 | Replication of null results: Absence of evidence or evidence of absence? | 2024 | eLife | doi:10.7554/elife.92311 · PMC11090505 | **pdf_section** |
| S11 | Principal stratification in causal inference. | 2002 | Biometrics | doi:10.1111/j.0006-341x.2002.00021.x | abstract |
| S12 | Design approaches to experimental mediation. | 2016 | Journal of Experimental Social Psychology | doi:10.1016/j.jesp.2015.09.012 | abstract |
| S13 | Mediation Analysis Is Harder Than It Looks | 2011 | Cambridge Handbook of Experimental Political Science | doi:10.1017/cbo9780511921452.035 | metadata |
| S14 | Increasing Transparency Through a Multiverse Analysis | 2016 | Perspectives on Psychological Science | doi:10.1177/1745691616658637 | abstract |
| S15 | Specification curve analysis | 2020 | Nature Human Behaviour | doi:10.1038/s41562-020-0912-z | metadata |
| S16 | The preregistration revolution. | 2018 | PNAS | doi:10.1073/pnas.1708274114 | metadata |
| S17 | The past, present and future of Registered Reports | 2021 | Nature Human Behaviour | doi:10.1038/s41562-021-01193-7 | metadata |
| S18 | Design-Based Uncertainty for Quasi-Experiments. | 2025 | Journal of the American Statistical Association | doi:10.1080/01621459.2025.2526700 | metadata |
| S19 | Prespecified falsification end points: can they validate true observational associations? | 2013 | JAMA | doi:10.1001/jama.2012.96867 | metadata |
| S20 | Noninferiority trials. | 2000 | Current Controlled Trials in Cardiovascular Medicine | doi:10.1186/cvm-1-1-019 · PMC59590 | metadata |
| S21 | Noninferiority and equivalence designs: issues and implications for mental health research. | 2008 | Journal of Traumatic Stress | doi:10.1002/jts.20367 | metadata |
| S22 | The use of the E-value for sensitivity analysis. | 2023 | Journal of Clinical Epidemiology | doi:10.1016/j.jclinepi.2023.09.014 | metadata |
| S23 | Quantifying the impact of unmeasured confounding in observational studies with the E value. | 2023 | BMJ Medicine | doi:10.1136/bmjmed-2022-000366 | metadata |
| S24 | Power to Detect What? Considerations for Planning and Evaluating Sample Size. | 2024 | Personality and Social Psychology Review | doi:10.1177/10888683241228328 | metadata |
| S25 | Using negative controls to identify causal effects with invalid instrumental variables. | 2025 | Biometrika | doi:10.1093/biomet/asae064 | metadata |
| S26 | Identification and estimation of causal peer effects using double negative controls for unmeasured network confounding | 2024 | JRSS-B | doi:10.1093/jrsssb/qkad132 | metadata |
| S27 | Designing and Interpreting Probes with Control Tasks | 2019 | EMNLP-IJCNLP 2019（pp. 2733–2743） | doi:10.18653/v1/d19-1275 · aclanthology.org/D19-1275.pdf | **pdf_section** |
| S28 | Rethinking the Value of Network Pruning | 2018 | arXiv 预印本 | arXiv:1810.05270 | **pdf_section** |
| S29 | With Little Power Comes Great Responsibility | 2020 | arXiv 预印本 | arXiv:2010.06595 | **pdf_section** |
| S30 | A Metric Learning Reality Check | 2020 | ECCV 2020 / Lecture Notes in Computer Science | doi:10.1007/978-3-030-58595-2_41 | metadata |
| S31 | Are Emergent Abilities of Large Language Models a Mirage? | 2023 | Advances in Neural Information Processing Systems 36 | doi:10.52202/075280-2425 | metadata |
| S32 | Sanity Checks for Saliency Maps | 2018 | arXiv 预印本 | arXiv:1810.03292 | metadata |
| S33 | How Registered Reports can improve trust in machine learning research. | 2026 | Nature Human Behaviour | doi:10.1038/s41562-026-02586-2 | metadata |

`pdf_section` 证据存放于会话临时目录（`/tmp/faudit/*.txt`），非仓库文件，不随仓库持久化。

---

## §2 方法设计装置

### §2.0 总览

| 装置 | 名称 | 原领域识别问题 | 对 NeuroTopo 的一句判定 |
|---|---|---|---|
| F-01 | 负对照结局 NCO | 未测量混杂不可见 | 可转移 |
| F-02 | 负对照人群 / 机制惰性亚组 | 处理在亚组内无效 | 可转移（改造方向） |
| F-03 | 等价检验 TOST / 非劣效 | "无效应"不可作为结论 | **优先可转移** |
| F-04 | 设计因子化混杂（双随机化） | 中介不可操纵 | 可转移（成本乘性增长） |
| F-05 | 主层分层 principal stratification | 后处理变量分层 | 需改造 |
| F-06 | 事前冻结判决函数 + 穷尽分支 | 判据分支非穷尽 | **优先可转移** |
| F-07 | 预指定证伪终点 / 安慰剂等价检验 | 平衡性检验被误用 | 可转移 |
| F-08 | 部分识别界 + E-value 型敏感性 | 混杂与主效应同量级 | **优先可转移** |
| F-09 | 设计基准不确定性 / 随机化推断 | 超总体模型假设 | 可转移 |
| F-10 | 近端因果推断 | 混杂可被"校正"而非仅检测 | 需改造（当前资源不足） |
| F-11 | 对照任务与选择性 selectivity | 探针容量伪装成任务能力 | **优先可转移** |
| F-12 | 从零重训 / 重初始化基线 | 结构收益 vs 优化路径收益 | **优先可转移** |
| F-13 | 事前功效与最小可检测效应 MDE | 不显著被当作无效应 | 可转移 |
| F-14 | 度量/协议敏感性（规格曲线） | 结论由度量选择决定 | 可转移 |
| F-15 | 级联随机化 sanity check | 归因未反映模型 | 需改造 |

### §2.1 五个首要装置的协议级细节

**F-03 等价检验（TOST）——把"无效应"变成可拒绝的假设**
- *识别问题*：原领域（生物等效性、心理学重复）中，"未检出差异"被误当作"差异不存在"。S07 的摘要原文：该检验被提出正是为了替换"检验无差异假设 + 要求功效"的 Power Approach。S10 正文：非显著结果"may also occur if an effect is actually present"，且"the lower the power of a study, the greater the ambiguity of a null result"。
- *协议做法*：事前声明最小感兴趣效应 `Δ`（SESOI）；检验 `H0: |β| ≥ Δ` 对 `H1: |β| < Δ`，用两个单侧检验，取两者 p 的**最大值**作为 TOST p；等价于要求 `(1−2α)` 置信区间**完全落入** `(−Δ, +Δ)`（S10 正文明确给出该等价关系并归因于 Schuirmann 1987）。
- *额外测量/资源*：只需已有的区间估计与一个事前声明的 `Δ`；另需事前功效以 `Δ` 为假设效应计算。
- *为何更抗混杂*：两臂标量差只能产出"不显著"，因而对 `Δ` 与 `(e_U−e_S)` 的混合无能为力；TOST 把"容量通道贡献小于 Δ"变成**一个可被数据拒绝的假设**，从而允许书写肯定性结论。
- *失效条件*：`Δ` 若事后按观测效应反推则整套失效；样本量不足时 TOST 永远不显著（无法区分"等价"与"功效不足"）；S10 亦指出小样本下"复制成功"反而更容易达成，故必须同时报告功效与 MDE。

**F-06 事前冻结判决函数 + 穷尽分支（多重宇宙 / 规格曲线 / 注册报告）**
- *识别问题*：原领域面临"研究者自由度"与"选择性报告"，导致结论不可证伪。S14 摘要：数据分析决策"simultaneously defensible, arbitrary, and motivated"；S15 提出把**全部合法规格**的结果一并展示。
- *协议做法*：(a) 在看到结果前枚举完整的合规格空间（种子 × 检查点 × 终点 × 匹配口径版本 × 样本量）；(b) 报告整个分布（规格曲线 / 多重宇宙），而非被选中的那一格；(c) 预注册一个**判决函数**，把结果向量映射到有限且**互斥穷尽**的判词集合，其中 `INCONCLUSIVE` 必须是**由事前条件触发的正式分支**（如：区间跨过 `δ_min`、容量上界 ≥ `δ_min/2`、功效低于目标）；(d) 保证每个分支可达（S17/S16/S33 提供注册报告的制度化先例，S33 专对机器学习）。
- *额外测量/资源*：主要成本是**计算量乘性增长**与写作成本；无新增测量通道。
- *为何更抗混杂*：它不是估计方法，而是**消除混杂解释的搜索空间**——两臂标量差的真正危险在于分析者可以在看到数字后选择口径；冻结判决函数使口径不能随数字移动。
- *失效条件*：枚举不完整（"规格曲线之外的规格"仍是自由度）；`Δ`、容量上界等常数若允许事后调整则退化为普通 NHST；注册报告本身也可能被规避（S17 讨论）。

**F-08 部分识别界 + 敏感性下界（E-value 型）**
- *识别问题*：未测量混杂与主效应同量级时，点估计不可信；原领域用"混杂需多强才能推翻结论"来量化。
- *协议做法*：给出混杂通道的**最大可能贡献**上界 `C_ub`（部分识别），并要求 `C_ub < 机制效应阈值`，而不是要求匹配；同时报告"若观测效应完全由混杂通道解释，该通道需达到多大强度"。
- *额外测量/资源*：需要一个**独立于机制**的容量通道测量（在 NeuroTopo 中即 oracle 投影代价/最优可达 KL），以及其区间。
- *为何更抗混杂*：两臂标量差只给出一个混合数；部分识别界把 `Δ` 的可达上界显式算出，使"容量通道能否解释 β"成为算术而非争论。G1-BOUND-0001 的设计 v0 已使用这一逻辑（`Δ ≤ 0.0064` 且 `β` 的 CI 下界 `> 0.0064` 蕴含 `e_U − e_S > 0`），但 S22/S23 提示 E-value 的原始定义面向风险比，连续型 KL 尺度上**需要自行论证校准**。
- *失效条件*：`C_ub` 不是有效界（G1-ALIGN-0001 停止条件 S4 正是"门槛须是被检验量的有效界，且需逐臂 `KL_trained − KL_projection` 证明覆盖"，packet 内**无此测量**）；界过宽则退化为无信息。

**F-11 对照任务与选择性（selectivity）**
- *识别问题*：机器学习内部——探针/读出层报告的"表征包含某信息"可能只是探针容量的产物。S27 正文原文："as long as a representation is a lossless encoding, a sufficiently expressive probe with enough training data can learn any task on top of it."
- *协议做法*：构造**控制任务**——"associate word types with random outputs"，输入分布与训练预算与主任务相同，但映射不可学习；用主任务准确率**减去**控制任务准确率得到 selectivity；模型/探针族比较一律在 selectivity 上做，而非绝对准确率。
- *额外测量/资源*：每个读出层需额外训练一个同预算控制任务；S27 报告控制任务仅需 10 维隐状态与少量样本即可标定。
- *为何更抗混杂*：两臂标量差无法分辨"机制真的编码了结构"与"读出层容量足够大"；控制任务给出一个**真值已知为零**的读出，把容量通道直接从每个数字里扣掉。
- *失效条件*：控制任务若与主任务难度不匹配，selectivity 的差会引入新偏倚；S27 正文亦承认"marginal differences in part-of-speech tagging accuracy ... correspond to large differences in selectivity"，即 selectivity 本身对探针选择敏感。

**F-12 从零重训 / 重初始化基线**
- *识别问题*：机器学习内部——"某结构带来收益"与"产生该结构的那条优化路径带来收益"混淆。这正是 NeuroTopo 的 `β = Δ + (e_U − e_S)`。
- *协议做法*：对每一个声称有收益的架构，加入"**同一架构 + 随机重初始化 + 相同预算从零训练**"基线；S28 摘要原文："fine-tuning a pruned model only gives comparable or worse performance than training that model with randomly initialized weights"，且"the pruned architecture itself, rather than a set of inherited 'important' weights, is more crucial"。S28 正文进一步把该观察提升为方法论要求："Our results suggest the need for more careful baseline evaluations"。
- *额外测量/资源*：每个格子一次额外训练（与主训练同预算）；S28 使用 5 次重复并报告标准差。
- *为何更抗混杂*：两臂标量差把"结构"与"优化轨迹"捆在一起；重初始化基线通过**切断继承权重**把两者分离，把 `(e_U − e_S)` 从残差变成可观测的对照。
- *失效条件*：若机制的收益**本来就来自优化路径**（如更好的初始化/条件数），该基线会正确地判为"结构无独立贡献"——这要求事先说明"收益"的定义域，否则结论会被误读为对机制的整体否定；预算匹配不严时该基线不公平。

### §2.2 其余装置的协议级细节（同一五字段）

| 装置 | 原领域识别问题 | 协议级做法（可照抄） | 额外测量/资源 | 为何比"两臂标量差"抗混杂 | 失效条件 |
|---|---|---|---|---|---|
| **F-01 NCO** | 未测量混杂在观测设计中不可见 | 选一个机制上不可能被处理影响、但与主结局**共享偏倚结构**的结局 `Y_nc`；事前声明"零结果"的定义；对 `Y_nc` 跑与主分析**同款**估计量 | 多一个完整结局的测量（同种子/同预算）；书面论证共享偏倚结构 | 提供一个"真值必为零"的可检验点，把偏倚从口头担忧变为可观测量 | `Y_nc` 被机制间接影响；或与 `Y` 不共享偏倚结构（S01 明确：NCO 对其他误差源的敏感性"additional work is needed"） |
| **F-02 负对照人群** | 处理在某些亚组内**先验无效** | 在机制被构造性关闭的亚组（如零跨组耦合 `c=0`）上做同一比较；该处任何非零差**必为偏倚**，据此标定偏倚量级 | 需要能构造"机制惰性"条件，且该条件不改变其他一切 | 给出偏倚的**绝对标定**而非相对比较；两臂标量差没有零点 | 亚组选择本身与机制相关（S04 要求形式化的"负对照人群"条件，非任意亚组）；惰性条件同时改变了任务难度 |
| **F-04 设计因子化混杂（双随机化）** | 中介不可直接操纵，故中介—结局关联是相关的 | S12 摘要列举的协议族：对 `X` 与 `M` **双重随机化（double randomization）**、并发的双重随机化、平行设计；并区分"证明 M→Y 因果效应"与"操纵 M→Y 效应强度"两类操纵 | 需要把疑似混杂因子变成**可由实验者操纵**的因子（乘性计算成本） | 把混杂从"要匹配掉的量"变成"被正交化的设计因子"，混杂通道成为一个可估的**主效应** | 疑似混杂因子不可操纵或操纵本身改变机制；因子水平交叉后单格样本量不足（S12 亦承认这些设计"contain their own set of challenges"） |
| **F-05 主层分层** | 后处理变量分层引入选择偏倚 | 按中间变量的**潜在值**定义主层，只在两臂中间变量取值相同的层内比较；该层是构造性的预处理协变量 | 需在**两臂都测量**中间变量（如"解是否真的实现了分组结构"），加单调性/排除性假设 | 把"机制是否真的被启用"变成可比条件，避免用后处理变量做普通分层 | 目标层可能为空（需 `INCONCLUSIVE` 分支）；单调性与 principal ignorability 需逐条论证（S11） |
| **F-07 证伪终点/安慰剂** | 平衡性用"不显著"判定属误用 | 事前指定一个处理**不应影响**的终点，用**等价检验**而非 NHST 判平衡（S09）；S19 提出"预指定证伪终点"验证观测关联 | 一个额外终点 + 等价界 `Δ` | 把安慰剂检验从"没能拒绝零"升级为"能拒绝非零"，与 F-03 联合才有判别力 | 等价界选得过宽则自动通过；S09 的等价框架要求事前指定界 |
| **F-09 设计基准不确定性** | 超总体 i.i.d. 假设不符 | 只把**分配机制**当作随机性来源，在随机化分布上做精确推断（Fisher 型），而非在超总体模型下做 | 需要明确的分配机制记录（种子/初始化/任务实例如何抽取） | 不依赖"种子 i.i.d. 正态"即可给区间；两臂 t 区间在 n=8 时对该假设极敏感 | 分配机制不可枚举或不独立；精确推断的功效通常低于模型法 |
| **F-10 近端因果推断** | 混杂可被**校正**而非仅检测 | 用两个负对照（暴露侧、结局侧）满足相关性+独立性（**非**排除性），通过 bridge function 非参数识别平均处理效应（S05/S06）；S25 用负对照处理**无效工具变量** | 需要两个有效的代理变量及其理论论证 | 是唯一能把 `Δ` **识别**而非仅界定/检测的装置；两臂标量差无法识别 | 代理变量假定极强且不可完全检验；NeurTopo 当前无两个可信的容量通道代理，资源不足 |
| **F-13 事前功效与 MDE** | 不显著被当作无效应 | 事前以 SESOI 计算功效；每个不显著结果旁**必须**报告最小可检测效应 MDE（S24 标题即"Power to Detect What?"）；S29 摘要原文："statistical power ... has largely been ignored by the NLP community" | 无需新测量，需写作与计算规范 | 两臂标量差的不显著结果在低功效下无信息；MDE 让"我们测不出什么"成为显式数字 | 事前效应假设本身无依据；MDE 误用为事后功效（S10 引 Hoenig & Heisey 2001 警告） |
| **F-14 度量/协议敏感性** | 结论由度量选择决定 | 在同一结果向量上跑**全部可辩护终点**并展示整个分布（规格曲线，S15；多重宇宙，S14）；S31 展示了"能力涌现"随度量选择消失/出现 | 需要多终点评估管线（CPU 小模型成本低） | 两臂标量差只在一个终点上给一个数；规格曲线给出"若换终点结论是否翻转" | 终点集合若不完整则仍有自由度；S30 表明协议控制不严时"改进"会整体消失，故需同时固定训练/调参协议 |
| **F-15 级联随机化 sanity check** | 归因未反映模型本身 | 自顶向下逐层随机化模型参数，检验归因/解释是否随之改变（S32） | 需归因/解释管线；多组随机化重复 | 提供"机制未被使用"的阴性对照，防止把读出层产物当作机制证据 | 仅适用于归因类读出，难直接用于主终点；随机化本身改变输出分布 |

---

## §3 对 NeuroTopo 的可转移性逐条判定

基准：`G1-MATCH-v3`（决定0069）**单轴匹配**、oracle-relative KL 终点、CPU 小模型、**已知生成模型**、预注册与穷尽分支要求；停止记录 `G1-ALIGN-0001`（决定0072 的 S1–S5）与 `G1-BOUND-0001`（`β_trained = Δ + (e_U − e_S)`）。

| 装置 | 判定 | 与 G1-MATCH-v3 单轴匹配 | 与 oracle-relative KL 终点 | 与 CPU 小模型 | 与已知生成模型 | 与预注册/穷尽分支 |
|---|---|---|---|---|---|---|
| F-03 TOST | **可转移（优先）** | 兼容：等价界加在**声明轴**上，不新增匹配轴 | 兼容：直接对末端 KL 差做 TOST | 兼容：纯后处理 | 兼容：`Δ` 可取一个有机制含义的最小 KL 差 | **正向**：直接补 S3 的缺口（"零真值点通过率"即 1−I 类错误，由构造控制） |
| F-06 冻结判决函数 | **可转移（优先）** | 兼容：多重宇宙含匹配口径版本作为一维 | 兼容：终点为规格曲线一维 | 兼容：成本在算力乘性 | 兼容 | **正向且必需**：直接补 S5（"(I=0, CI 跨零) 无分支"） |
| F-08 部分识别界 | **可转移（优先）** | **改造匹配哲学**：把"匹配到同量级"换成"证明 `C_ub < δ_min/2`"，与 S1 完全同构 | 兼容：`C_ub` 就是 `Δ` 的界，oracle 已在管线内 | 兼容：oracle 投影代价在 CPU 上可算 | 兼容：`KL_projection` 依赖已知生成模型 | 正向：`C_ub` 须事前冻结并给覆盖证明（S4） |
| F-12 从零重训基线 | **可转移（优先）** | 兼容：不新增匹配轴，但**要求两臂都各加一格同预算重训** | 兼容 | 兼容（成本×2） | 兼容 | 正向：须事前注册为必备臂 |
| F-11 对照任务 | **可转移（优先）** | 兼容 | 兼容：selectivity = 主终点 − 控制终点 | 兼容 | **强兼容**：控制任务可由生成模型直接采样随机目标 | 正向：控制任务须事前冻结 |
| F-01 NCO | 可转移 | 兼容 | 需要新增一个"只受容量通道影响"的 KL 型结局（如逐臂 `KL_projection` 差） | 兼容 | 兼容 | 正向 |
| F-02 负对照人群 | 可转移（改造） | 兼容 | 兼容：在机制惰性条件上做同款 KL 比较 | 兼容 | 兼容：`c=0` 可由生成模型构造 | 正向：惰性条件须事前固定 |
| F-04 设计因子化 | 可转移（成本乘性） | **兼容但需澄清**：v3 只要求声明轴匹配、其余轴披露；把其余轴**升级为交叉因子**是加强而非违反 | 兼容 | 兼容（小模型使因子格可行） | 兼容 | 正向 |
| F-13 MDE | 可转移 | 兼容 | 兼容 | 兼容 | 兼容 | 正向：须事前以 `δ_min` 计算 |
| F-14 规格曲线 | 可转移 | 兼容：多个匹配口径版本作为一维 | 兼容：多终点含 oracle-relative KL | 兼容 | 兼容 | 正向 |
| F-09 设计基准推断 | 可转移 | 兼容 | 兼容 | 兼容 | 兼容 | 正向：需记录种子分配机制 |
| F-07 证伪终点 | 可转移 | 兼容 | 兼容 | 兼容 | 兼容 | 正向 |
| F-05 主层分层 | 需改造 | 兼容 | 兼容 | 兼容 | **有优势**：可用生成模型直接判定"解是否实现了分组结构" | 需新增 `INCONCLUSIVE` 分支（空层） |
| F-15 级联随机化 | 需改造 | 兼容 | 不直接适用（面向读出/归因，非主终点） | 兼容 | 兼容 | 中性 |
| F-10 近端因果推断 | 不可转移（当前） | — | — | — | 已知生成模型**理论上**能构造负对照，但需两个可信容量代理 | 假设不可检验，与"可证伪"要求冲突 |

### §3.1 三条最高优先级的落地建议（不改仓库其他文件）

1. **用 F-06 + F-03 联合替换 S3/S5 的判据架构**：事前冻结"规格曲线 × 判决函数"，把 `INCONCLUSIVE` 定义为由 `C_ub ≥ δ_min/2`、区间跨 `δ_min`、功效低于目标三者之一**机械触发**的分支。这使分支**穷尽**是构造性的，而非事后补充。
2. **用 F-08 替换"匹配到同量级"的目标**：证明 `C_ub < δ_min/2`（S1）本身就是一个**部分识别**陈述，比"把容量通道做小"更可证伪，也更接近可交付。
3. **用 F-12 + F-11 把 `(e_U − e_S)` 从残差变成可观测量**：为每臂加"同架构随机重初始化 + 同预算从零训练"格，并为每个读出层加同预算控制任务，报告 selectivity 而非绝对 KL。这是把 `β = Δ + (e_U − e_S)` 的两个通道**分别测量**而非**分别假设**的唯一路径。

---

## §4 我没能核验的限制

1. **20 条来源仅到 `metadata` 层**（S03、S05、S06、S13、S15–S26、S30–S33）。我没有读过它们的摘要或正文，其内容描述来自标题与场所推断。**§2 中任何引用这些编号的机制细节都属于低置信度**，尤其：F-08 的 E-value 校准细节（S22/S23）、F-09 的设计基准推断形式化（S18）、F-10 的 bridge function 条件（S05/S06/S25/S26）、F-15 的级联随机化协议（S32）、F-14 的度量敏感性论证（S30/S31）。
2. **OpenAlex 通道失效**：首次查询即返回 `Rate limit exceeded`（日额度耗尽），**0 条取回**，已停用。因此本审计**没有**做摘要级的相关性排序扫描。
3. **arXiv API 部分失效**：`ti:`/`all:` 短语查询多次返回空响应，后段连续 `HTTP 429`；S28/S29/S32 的 arXiv 标识符最终经 HuggingFace Papers 检索通道取回，S28/S29 的正文经 `arxiv.org/pdf/` 直取成功，S32 **未取正文**。
4. **未核验的正文（8 条 `abstract` 层来源全部未读正文）**：S01（PMC 全文不可得）、S04、S07、S08、S09、S11、S12（作者手稿全文接口返回空）、S14。这些来源的"协议做法"描述**仅依据摘要文本**，操作细节可能有出入。
5. **未找到/未检索到**：起始查询词中的 `factorial design mediation analysis` 与 `placebo test design` 作为精确短语在 EuropePMC 返回 **0 命中**；F-04 的协议族描述**全部来自 S12 的摘要文本**（"double randomization, concurrent double randomization, and parallel"），我**未能**读到 S12 正文以确认每个设计的操作细节。**这不构成"无人做过"的结论。**
6. **未做的事**：无本地复现、无代码 commit 级核验、未运行仓库内任何脚本、未验证任何来源的数学正确性。§3 的兼容性判定是**我的推理**，不是文献主张。
7. **尺度不可迁移处**：E-value、MDE、非劣效界等工具的定义域（风险比、均值差）与 KL 差不同；本文件未提供任何换算公式，若要落地须单独论证校准（见 §3.1 第 2 条）。
