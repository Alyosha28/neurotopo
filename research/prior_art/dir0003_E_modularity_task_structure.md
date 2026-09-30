# dir0003-E｜模块化/条件计算：任务侧结构的事前测量与专化的因果确立

- **审计日期**：2026-09-24（UTC）
- **审计类型**：文献方法审计（design patterns），非综述；不评价 headline results
- **证据等级声明**：**无本地复现**。本文件所有条目均只来自本轮实际取回的原文 PDF 正文段落（`pdf_section`）或书目元数据（`crossref_metadata`）。未运行任何被审计论文的代码，未训练任何模型，未改动仓库其他文件。
- **检索边界（engine｜query）**：
  1. `arxiv`｜`ti:"task affinity"`
  2. `arxiv`｜`all:"gradient conflict" AND all:"multi-task"`
  3. `arxiv`｜`all:"which tasks" AND all:"learn together" AND all:"multi-task"`
  4. `arxiv`｜`ti:"modular architecture enough"`
  5. `arxiv`｜`ti:"are neural nets modular"`
  6. `arxiv`｜`ti:"emergent modularity"`
  7. `arxiv`｜`all:"expert specialization" AND all:"mixture of experts"`
  8. `arxiv`｜`ti:"modularity" AND all:"specialization"`
  9. `arxiv`｜`all:"causal" AND all:"expert" AND all:"ablation" AND all:"mixture-of-experts"`
  10. `arxiv`｜`ti:"optimization trajectories" AND all:"multi-task"`
  11. `arxiv`｜`ti:"taskonomy"`
  12. `arxiv`｜`ti:"criticality" AND all:"modules"`
  13. `arxiv`｜`ti:"task affinity"`（含 `Grad-TAG`）
  14. `crossref`｜`task affinity matrix transfer learning taskonomy`
  15. `crossref`｜`preregistration machine learning hypothesis confirmation`（相关性差，未产出采用来源）
  16. `openalex`｜`mixture of experts expert specialization measurement`（**仅首批成功，随后全 IP 免费额度耗尽，本文件不采用其结果**）
  17. **未命中记录**（不等于无人做过）：`arxiv` 上 `all:"preregistration" AND all:"deep learning"`、`all:"task structure" AND all:"architecture" AND all:"modular"`、`all:"modularity" AND all:"necessary" AND all:"ablation" AND all:"network"`、`all:"mixture-of-experts" AND all:"causal mediation"` 均返回 0 条。
- **通道限制**：web_search/web_fetch 不可用；OpenAlex 额度耗尽后改用 arXiv API + Crossref + 全文 PDF（`pdftotext`）。

## §1 来源表

| # | 逐字标题 | 年 | 场所 | URL | 核验层次 |
|---|---|---|---|---|---|
| S1 | How Modular Is a Frontier Mixture-of-Experts? A Pre-registered Causal Test in Which Apparent Expert Modularity Mostly Dissolves | 2026 | arXiv 预印本 | https://arxiv.org/abs/2606.25092 | pdf_section（§1–§6 全文） |
| S2 | Is a Modular Architecture Enough? | 2022 | arXiv 预印本 | https://arxiv.org/abs/2206.02713 | pdf_section（§3–§5、式16–19） |
| S3 | Are Neural Nets Modular? Inspecting Functional Modularity Through Differentiable Weight Masks | 2021 | ICLR 2021（PDF 页眉） | https://arxiv.org/abs/2010.02066 | pdf_section（§2、§3、§7） |
| S4 | "It's a Match!" -- A Benchmark of Task Affinity Scores for Joint Learning | 2023 | arXiv 预印本 | https://arxiv.org/abs/2301.02873 | pdf_section（Abstract、§1、§4） |
| S5 | Which Tasks Should Be Learned Together in Multi-task Learning? | 2020 | Proc. 37th ICML, PMLR 119 | https://arxiv.org/abs/1905.07553 | pdf_section（§1、§4、§5） |
| S6 | Can Optimization Trajectories Explain Multi-Task Transfer? | 2024/2025 | Transactions on Machine Learning Research (December/2024) | https://arxiv.org/abs/2408.14677 | pdf_section（Abstract、§1、§3） |
| S7 | Scalable Multitask Learning Using Gradient-based Estimation of Task Affinity | 2024 | ACM（PDF 载 ACM ISBN 979-8-4007-0490-1/24/08） | https://arxiv.org/abs/2409.06091 | pdf_section（§1） |
| S8 | The intriguing role of module criticality in the generalization of deep networks | 2020 | arXiv 预印本 | https://arxiv.org/abs/1912.00528 | pdf_section（§2、§3） |
| S9 | The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not Necessarily Domain Expertise. | 2026 | arXiv 预印本 | https://arxiv.org/abs/2604.09780 | pdf_section（Abstract、§1） |
| S10 | Modularity in Transformers: Investigating Neuron Separability & Specialization | 2024 | arXiv 预印本 | https://arxiv.org/abs/2408.17324 | pdf_section（Abstract、§1、§2） |
| S11 | Taskonomy: Disentangling Task Transfer Learning | 2018 | arXiv 预印本（Crossref 另有 CVPR 2018 记录 DOI 10.1109/CVPR.2018.00391，metadata 层） | https://arxiv.org/abs/1804.08328 | pdf_section（§3、§4） |
| S12 | Pattern Selectivity is Not Task-Causal Structure: A Cross-Architecture Mechanistic Study of Composed-Task Circuits in 1B-Class Language Models | 2026 | arXiv 预印本 | https://arxiv.org/abs/2606.05378 | pdf_section（Abstract、§1） |
| S13 | PiKE: Adaptive Data Mixing for Large-Scale Multi-Task Learning Under Low Gradient Conflicts | 2025 | arXiv 预印本 | https://arxiv.org/abs/2502.06244 | pdf_section（§1、§3 Def. 3.1/3.2） |
| S14 | Improving Expert Specialization in Mixture of Experts | 2023 | arXiv 预印本（正文含 github.com/aistats2023-1554/moe） | https://arxiv.org/abs/2302.14703 | pdf_section（§1、§3） |
| S15 | Task Grouping for Automated Multi-Task Machine Learning via Task Affinity Prediction | 2023 | arXiv 预印本 | https://arxiv.org/abs/2310.16241 | pdf_section（Abstract） |
| S16 | Multilinear Mixture of Experts: Scalable Expert Specialization through Factorization | 2024 | arXiv 预印本 | https://arxiv.org/abs/2402.12550 | pdf_section（Abstract、§4.1） |
| S17 | Emergent Modularity in Pre-trained Transformers | 2023 | arXiv 预印本 | https://arxiv.org/abs/2305.18390 | pdf_section（§3、§4、Fig.2） |

## §2 方法设计装置（核心）

### E1｜先测任务结构，再预测架构收益

**E1-D1 已知生成模型 + 结构参数作剂量（S2）**
- **识别问题**：真实任务套件的"结构"未知，因此任何"结构→收益"主张都可以被"你挑的任务恰好如此"反驳（对应项目困境 c）。
- **做法**：构造 `c ~ Categorical`、`y = α_c·x1 + β_c·x2` 的规则型生成分布，**规则数 R 由实验者设定**，并让模块数与规则数相等（R 规则 / R 模块）。随后扫 R ∈ {2,4,8,16,32}，观察各架构的崩溃/专化指标随 R 的变化。
- **资源**：CPU 级 MLP / MHA / RNN 小模型；R 与宽度是小整数。
- **为何优于"两臂比一个标量差"**：剂量 R 是**生成器的属性**，在比较任何架构之前就已确定且可无限复现，因此不存在任务选取通道；交互项（架构 × R）本身是可证伪预测。
- **失效条件**：合成规则的"结构"是否外推到真实任务未被证明；作者自己指出模块-规则数相等是刻意的强假设。指标随 R 恶化只是"问题变难"，需同时报告单任务基线才能分离难度通道。

**E1-D2 Oracle 臂 / 完美专化上界（S2）**
- **识别问题**：模块化架构的"潜力"与"端到端训练兑现的量"混淆（对应项目困境 b）。
- **做法**：在 Monolithic / Modular / Modular-op 之外增设第四臂 **GT-Modular**，其激活概率**直接由真值 c 设定**（不经反向传播），并证明函数类包含关系 GT-Modular ⊆ Modular-op ⊆ Modular ⊆ Monolithic。作者明确将其定位为 "an oracle benchmark" 并问 "how far are typical modular systems from obtaining similar performance"。
- **资源**：多一个同构臂，无额外数据。
- **为何抗混杂**：给出**上界参照**，把"实测臂差"拆成"潜力缺口"与"优化缺口"两段；两臂标量差无法区分二者。
- **失效条件**：oracle 臂在真实任务上不可得；只适用于生成结构已知的合成设定。

**E1-D3 任务侧指数工具箱的整体基准化（S4）**
- **识别问题**：单个指数被选中往往是因为它在作者偏好的任务对上好看。
- **做法**：同时定义并比较多族事前/训练中指数——分类学距离、输入归因相似度、表征相似性分析（RSA，Spearman）、梯度相似度（任务损失梯度余弦）、梯度可迁移性、以及作者提出的 label injection；在 Taskonomy 数据集上对**所有任务对**计算，然后与真实 MTL gain 求 Pearson、与最优伙伴任务排序求 Kendall。资源上需要对所有任务对做真实 MTL 训练以取得 ground truth（昂贵，这是该装置的主要成本）。
- **为何抗混杂**：被检验的对象是**指数本身的有效性**，不是某个架构；并且**报告负面结果**（"no scoring technique strongly correlates with the MTL gains"），因此不能靠挑任务对制造结论。
- **失效条件**：仅在小规模场景；作者自述 "even in a small-scale scenario, task affinity scoring does not correlate well with actual MTL performance"。相关性检验不构成交互预测。

**E1-D4 两类亲和度的分离检验 + 预算约束下的显式目标（S5）**
- **识别问题**：容易假定"可迁移 = 适合联合训练"，从而把 transfer 指数直接当多任务剂量。
- **做法**：(i) 分别测 transfer affinity 与 multi-task affinity，**直接报告二者相关系数**（作者报告 Pearson r = −0.14 / −0.153 等，no correlation）；(ii) 把任务分组形式化为**在显式推理时间预算 b 下的组合优化**（`argmin_{cost(S)≤b} L(S)`，branch-and-bound；亦可用 BIP），候选集包含全部 2^|T|−1 个分组。
- **资源**：五个任务时需 31 个候选网络的训练。
- **为何抗混杂**：资源约束（推理时间预算）被写进目标函数而非事后声明；同时把"指数有效性"与"分组决策"分成两步，各自可单独证伪。
- **失效条件**：作者报告亲和度依赖学习设定（网络容量、数据量），故指数**不可跨设定搬运**；候选集随任务数指数增长。

**E1-D5 先测剂量的动态范围，再决定是否下注（S13）**
- **识别问题**：梯度冲突被默认为普遍现象，但若剂量本身接近 0，则任何"冲突缓解"归因都无处可归。
- **做法**：先给出可检验定义——`c-conflicted`（`⟨∇Lj,∇Lk⟩ ≥ −c(‖∇Lj‖+‖∇Lk‖)`）与 `c̄-aligned`；**先测 c 与 c̄ 的分布**（只需在少量 checkpoint 上做梯度内积统计，相对训练可忽略），在 LLM 预训练上发现 "minimal gradient conflict"，再据此设计方法。
- **为何抗混杂**：把"指数在目标尺度上是否有方差"变成**前置可证伪问题**；若剂量无动态范围，实验设计本身就被否决，而不是产生一个无法解释的小效应。
- **失效条件**：作者明确该结论是尺度相关的——视觉 MTL 文献中冲突普遍，LLM 预训练中稀少；换模型族/尺度需重测。

**E1-D6 在匹配训练损失上比较，并把优化通道与泛化通道拆开（S6）**
- **识别问题**：实测臂差把"容量通道"与"优化难度通道"混在一起（对应项目困境 b）。
- **做法**：(i) 用**训练损失轨迹**对齐 STL 与 MTL 臂，在**可比的训练损失下**评估泛化（"a gap in generalization at comparable training loss"）；(ii) 把梯度冲突当作**剂量**做干预式扫描，检验它对梯度一致性、早期 Fisher 信息、损失面锐度的影响，再检验这些量是否解释任务泛化。资源为 5 个多任务设定 × 多次扫描，模型规模不大。
- **为何抗混杂**：结论是**双重的**——冲突与"任务优化受损"相关，但 "the amount of conflict has little-to-no correlation with task generalization"。因此一个指数若只预测优化难度，**不能**被用作机制收益的代理；这直接把优化难度通道从容量/组织通道中剥离出来。
- **失效条件**：作者自述其结论是"当前无法解释"，即该装置给出的是**排除性证据**而非正向识别；单种子/规模有限。

**E1-D7 在共享基座模型上廉价估计指数（S7、S15）**
- **识别问题**：逐对训练 MTL 模型的成本使剂量-反应扫描不可行，于是只能挑少数任务对（= 任务选择混杂的温床）。
- **做法**：S7 训练一个覆盖全部任务的 base model，再用 **gradient-based estimator** 估计 pairwise 与 higher-order affinity（报告 32×–48× FLOPs 降低），再对亲和度矩阵做半正定规划聚类；S15 用 "inherent task features and STL characteristics" 训练一个**预测器**，配合随机化搜索以减少真实 MTL 训练次数。
- **为何抗混杂**：让**全量任务对**的指数矩阵变得可负担，从设计上移除"只测几对"的选择空间。资源为一个基座模型 + 梯度估计，CPU/小规模可行。
- **失效条件**：这是**成本**论证而非**有效性**论证；S7/S15 均以与穷举分组的一致性/误差作为指标，不保证指数在多任务收益上有效（S4、S5 给出反例）。

**E1-D8 数据完全相同以消除输入分布混杂（S11）**
- **识别问题**：迁移/亲和度差异可能来自各任务输入数据本身的差异，而不是任务内在关系。
- **做法**：为所有任务在**完全相同的像素**上标注（400 万张室内图像，每张图都有每个任务的标注），作者明说这样 "eliminates the possibility that the observed transferabilities are affected by different input data peculiarities rather than only task intrinsics"；再定义 `D_{s→t}` 为一阶亲和度矩阵，并用 BIP 求全局迁移策略；迁移函数刻意采用**低容量 + 少量数据**以度量"可及性"。
- **资源**：极高（每个有序任务对训练一个迁移模型 + 自制数据集）。
- **为何抗混杂**：把输入分布设为常量，使指数成为任务内在量的估计；低容量读出函数把"信息存在"与"信息易取"分开。
- **失效条件**：成本不可移植到小项目；字典采样是否覆盖任务空间只能靠外推实验（作者以 out-of-dictionary 任务部分回应）。

### E2｜专化的因果确立与"模块化≠专化"的测量

**E2-D1 预注册 + 尺寸匹配随机零假设 + 可证伪选择性判据（S1）**
- **识别问题**：事后看到消融结果再挑指标/语料/阈值（对应项目困境 c、d）。
- **做法**（五项要素，逐项可抄）：(1) 由 routing mass atlas 定义专家 family，**在任何干预之前冻结** family→axis 假设（六个 family）；(2) 干预算子为推理期 **router masking + gate renormalization**（把被掩码专家的 router logits 置 −∞ 再 top-k softmax），并做 smoke test 确认 "leak = 0"；(3) 效应只有在**击败同尺寸（16 个）随机专家零假设**（约 10 次抽样，mean+2σ）时才算数；(4) **选择性判据**：最差 off-target 效应 ≤ on-target 的 1/3；(5) 预注册**两条判决规则**（点规则 + bootstrap 95% CI 保守规则）。
- **资源**：作者全研究 < 30 B200-hours；**零假设与判据本身不需要额外训练**。
- **为何优于"两臂比一个标量差"**：`routing 质量高` 与 `消融它有害` 被显式分开——作者发现最高 routing mass 的 shared-core "is causally redundant rather than load-bearing"，即观测指数对因果重要性**没有**预测力。尺寸匹配零假设排除了"任意 16 个专家都会造成损害"。
- **失效条件**：作者自陈零假设 "is a noise band, not a calibrated p-value"；只有**阳性对照无阴性对照**（"single control establishes sensitivity, not specificity"）；accuracy 腿 n=25–80、单种子、功效不足；家族定义本身来自 routing（若 routing 几何被表示空间决定，家族边界可能不是功能边界，见 E2-D12）。

**E2-D2 把"测量"当作自变量：多指标 × 多语料 × 多统计杠（S1）**
- **做法**：同一批消融在 **4 个指标**（task accuracy / problem-text NLL / solution NLL / 独立语料 per-language NLL）× **2 个语料**（FLoRes-200 与独立 Wikipedia）下评分，并在 W4A4 与 BF16 两种精度上重跑。
- **结果形态（方法要点，非 headline）**：判决随语料翻转（Spanish 在一个语料上选择性、在另一个上渗漏到 Arabic）、随指标翻转（math 在 accuracy 下与 general 纠缠、在 solution-likelihood 下看似选择性）、随统计杠翻转（es 仅以 0.002 通过点规则而在 BF16 下以 −0.015 失败）。
- **为何抗混杂**：把"结论对测量选择的稳健性"变成**报告项**；边界案例（straddle the boundary）被迫暴露，而不是被单一指标的噪声吸收。
- **失效条件**：成本随指标×语料×精度倍增；作者结论是"测量必须在事前被控制"，而非某个具体阈值可通用。

**E2-D3 可微权重掩码 + 双向重叠度做功能模块同一性（S3）**
- **识别问题**：按单元相似度聚类识别模块，但 "Units can be viewed as mere 'wires'"——共享单元不等于共享计算。
- **做法**：冻结已训练权重，对目标子任务训练一个 **binary weight mask**（每权重一个 logit，Gumbel-Sigmoid + straight-through 二值化，禁止连续缩放以免改变函数）；掩码即"负责该功能的子网"；再计算不同任务掩码之间的 **IoU 与 IoMin**（IoMin 取双向最小值，防止单向包含被判为共享）。
- **资源**：每个目标子任务一次掩码训练（权重冻结，比从头训练便宜）；FNN/LSTM/CNN/Transformer 均可。
- **为何抗混杂**：模块由**功能**（在指定目标上训练出的掩码）定义，而非由事后聚类或命名定义；IoMin 消除"模块 A 被判包含于 B"的方向性伪共享。
- **失效条件**：掩码训练本身是一个优化问题，可能未收敛；掩码非唯一（不同种子可能给不同子网）；作者结论是"典型网络满足 P_specialize 但不满足 P_reuse"，即该装置主要用于**证伪**共享主张。

**E2-D4 用 I/O 接口操纵检验"任务相似度"解释（S3）**
- **识别问题**：观察到不同任务使用不同权重，就归因于"任务不同"——但可能只是共享/分离的输入输出接口造成的平凡差异（对应项目困境 d 的"硬编码身份"）。
- **做法**：设计两组互补数据集：**shared I/O**（加/乘共用输入输出单元，偏向权重共享）与 **separate I/O**（同一函数做两次但 I/O 分离，偏向不共享）。作者结论：weight sharing "is mostly driven by shared I/O rather than task similarity"。
- **资源**：合成算术任务，极小。
- **为何抗混杂**：这是**平凡混杂的直接操纵**——把解释变量（I/O 结构）显式设成两个水平，而不是留作事后解释。
- **失效条件**：仅在合成算术任务上验证；"任务相似度"未给出独立测量，故该装置只能否决 I/O 解释、不能正向确立相似度作用。

**E2-D5 三重零模型：匹配随机扰动 + 分布零假设 + 未训练模型（S17）**
- **识别问题**：扰动/消融有效应，可能只是"扰动任意神经元都有害"或"结构在初始化时就存在"。
- **做法**：(1) 按 predictivity 取 top-1% 神经元扰动，**与随机扰动相同数量神经元对照**；(2) 对子功能神经元在专家上的分布做**超几何零假设**检验（p < 0.001）并拒绝独立随机分布；(3) 在同一分析下测**随机初始化模型**（作者报告随机初始化模型不显示该现象）。
- **资源**：需预训练 checkpoints；分析本身廉价。
- **为何抗混杂**：三个零模型分别封堵"剂量效应"、"分布巧合"、"训练无关的架构先验"三条通道；两臂标量差一条都封不住。
- **失效条件**：超几何零假设假定独立性与随机分布，若神经元存在相关性会低估显著性；predictivity 排名本身由同一数据估计，需交叉验证拆分（作者用 seen/unseen 数据集部分处理）。

**E2-D6 未训练模型作为专化下界对照（S10）**
- **做法**：对 ViT 与 Mistral 分别计算神经元重要性/聚类重叠，并在**随机初始化模型**上重复同一分析；作者观察到重要性模式在随机初始化模型中"persist to some extent"，因而把现象定性为"训练精炼的固有结构"而非学习出的专化。
- **为何抗混杂**：把"专化"的零水平从 0 改成**架构先验水平**；否则任何观察到的不对称都会被误读为学习结果。
- **失效条件**：仅"some extent"的定性比较，未见效应量匹配；随机初始化模型的行为与训练后不可直接比较（尺度不同）。**E2-D7 连续损伤函数：module criticality（S8）**
- **识别问题**：二值消融只有"掉/不掉"，无法给出分级的、可比较的模块依赖度量。
- **做法**：固定其他模块在训练后取值，令模块 i 沿 `θ_i^α = (1−α)θ_i^0 + αθ_i^F` 从初始化走到训练值；找性能（训练误差）下降未超过阈值 ε 的最小 α 作为 criticality，另加该路径上的"谷宽"（噪声稳健性）。
- **资源**：只需一个已训练模型 + 逐模块插值评估，**无需任何额外训练**。
- **为何抗混杂**：给出**连续、可排序**的模块依赖量；并证明常规复杂度度量（谱、Frobenius/∞ 范数）"fail to distinguish critical layers"，即该装置测的是别的东西。作者进一步主张是**整体网络 criticality** 而非 critical 模块的个数控制泛化。
- **失效条件**：路径依赖（沿 init→trained 直线插值，其他方向未测）；ε 需事先选择；理论部分是 PAC-Bayes 界，不构成对"模块化收益"的因果识别。

**E2-D8 剂量-反应掩码：看崩溃是"悬崖"还是"优雅退化"（S1）**
- **做法**：对最共享的专家做**剂量-反应掩码**（最多 32 个），观察下游指标随剂量如何变化；作者观察到 "degrades MMLU-Pro by only ∼0.11, gracefully and with no cliff, the signature of distributed, redundant computation"。
- **为何抗混杂**：用**曲线形状**而非单点差值做推断；"无悬崖"这一形态本身可被反证（若真存在关键模块，应出现悬崖）。
- **失效条件**：形态判读是定性的，未给悬崖/无悬崖的统计判据；"冗余"与"该模块不参与"在形态上不可区分。

**E2-D9 干预式编辑：零化专家权重 → 类别级 polysemanticity（S16）**
- **做法**：在前向传播中把第 n 个专家的权重矩阵置零（`W_n := 0`），其余完全不动；取每类测试精度变化向量 `d_c^(n) = (y_c − ŷ_c^(n))/y_c`，以该向量对 one-hot 的偏离量定义 "class-level polysemanticity"；并扫描专家总数（32→1024）观察 polysemanticity 随容量的变化，另报告**参数匹配**的稠密块精度以做资源对照。
- **为何抗混杂**：干预是**精确、可逆、单变量**的（只动一个专家）；用"损伤是否集中在少数类别"作为专化定义，而不是用门控权重或路由频率。
- **失效条件**：置零是分布外输入，可能触发非线性补偿；polysemanticity 是相对指标，未与随机专家置零对照（该文用参数匹配稠密块做精度对照，不是损伤对照）。

**E2-D10 跨架构可移植性检验 + 五类 screen 结果分类法（S12）**
- **识别问题**：把"某指标筛出的电路"当成任务因果结构，但筛选配方与具体电路可能不可分。
- **做法**：统一 screen-and-ablate 协议跑 **4 个任务 × 3 个 1B 级模型**（含一个 MoE），每格用 **matched-random null、10 个种子**；引入**五类结果分类法与量化阈值**：primary cause / secondary cause / correlate / interferer / null。结论形态："The recipe ports across pipelines; the specific circuit it does not."
- **为何抗混杂**：把"机制主张"的成立条件定义为**跨模型族复现**；五类分类法允许 "correlate" 与 "interferer" 这两个此前会被误报为机制的中间状态。
- **失效条件**：仅 3 个 1B 级模型、12 个 (task, model) 格；阈值由作者设定；作者自述所提 MoE 假设是 "falsifiable hypothesis" 而非已确立结论。

**E2-D11 预声明的标量专化指数（S14）**
- **做法**：用**专家使用与标签的互信息 `I(E;Y)`** 度量专化（越高越专化），并用**选择熵 `H_s`** 度量门控分解的任务分解熵；作者据此比较 attentive gating 与原 MoE，并给出可复现代码链接。
- **为何抗混杂**：专化的判据是**事前声明的标量**，不是事后挑选的定性观察；熵/MI 在同一量纲下可跨架构比较。
- **失效条件**：门控分布指标是**观测性**的——S1 与 S9 分别给出反例（观测 routing 不预测因果重要性；专化可由表示几何完全解释）。

**E2-D12 用机制性零模型吞掉"专化"信号（S9）**
- **识别问题**：观察到专化就归因于路由架构。
- **做法**：证明路由是**线性映射**，故隐状态相似性对专家使用相似性既是必要也是充分条件（Prop. 1，跨 5 个预训练模型在 token 与 sequence 两级验证），从而"专化是表示空间组织的涌现属性，而不是路由架构自身的属性"；进一步给出 load-balancing loss 压制共享隐方向的理论解释；最后用**安慰剂式比较**说明专化模式不可解释——"expert overlap between different models answering the same question is no higher than between entirely different questions (∼60%)"。
- **为何抗混杂**：这是最强形式的混杂消除——不满足于"控制掉"混杂，而是**证明该观测指数可被混杂完全复现**，因而其增量解释力为零。
- **失效条件**：理论部分针对线性路由与特定负载均衡损失；"不可人类解释"的结论依赖所测模型与 ∼60% 这一具体数值。

## §3 对 NeuroTopo 的可转移性逐条判定

| 装置 | 判定 | 与项目约束的兼容性说明 |
|---|---|---|
| E1-D1 已知生成模型 + 结构参数作剂量 | **可转移** | 完全兼容"已知生成模型"与"CPU 小模型"。**建议**：把结构参数（规则数/分组数/因子共享程度）作为注册时的剂量，并**预注册"剂量 × 架构"交互方向**。G1-MATCH-v3 单轴匹配在每一剂量水平上分别执行与声明即可，不改动匹配条款。 |
| E1-D2 Oracle 臂 | **可转移** | 与既有 "oracle-relative KL 终点"直接同构：GT-Modular 就是按真值 c 路由的 oracle 臂。**建议**：把 oracle 缺口（实测 KL − oracle KL）作为主报告量，替代裸臂差，从而拆分"潜力缺口"与"优化缺口"。注意 oracle 臂不消耗额外训练数据，但必须披露其推理/计算比（G1-MATCH-v3 要求 matched-or-operator-profiled executed compute）。 |
| E1-D3 指数工具箱基准化 | **需改造** | 需要全任务对真实 MTL ground truth（昂贵）。改造方向：在已知生成模型上用**穷举**取代近似（剂量数小，CPU 可承受），并把"指数 vs 真实 gain"的相关系数作为**独立于机制主张的注册产物**。禁止把该相关系数当作机制收益证据。 |
| E1-D4 两类亲和度分离 + 预算约束目标 | **需改造** | BIP/branch-and-bound 与"禁止事后挑任务"高度兼容（目标函数事前声明）。但作者报告亲和度依赖学习设定，故**不能跨设定搬运指数**；须在注册中固定容量/数据量并声明不跨设定外推。 |
| E1-D5 先测剂量动态范围 | **可转移** | 极低成本，纯诊断。**建议**：在 Gate 1 注册前先测候选 K 因子/冲突指数的方差，若无可测动态范围直接否决该剂量，避免产生无法归因的小效应。与 AGENTS.md"Gate 1 前须指定候选或参考 K 因子、可证伪预测"一致。 |
| E1-D6 匹配训练损失 + 优化通道剥离 | **可转移（高优先级）** | 直接针对困境 (b)。G1-MATCH-v3 只匹配"机制消耗的那一轴"，**不匹配状态/优化轨迹**，因此本项目特别需要一个"在可比训练损失下比较"的辅助报告。**注意**：这是**辅助诊断**，不得被读作隔离了机制的组织形式（AGENTS.md 已明确 state-unmatched 比较不得如此解读）。 |
| E1-D7 基座模型上的廉价指数 | **需改造** | 梯度估计方法与 CPU 小模型兼容；但需注意 S7 只证成本、不证有效性（S4/S5 是反例）。用途限定为"让全任务对矩阵可负担"，不得作为有效性依据。 |
| E1-D8 数据完全相同 | **可转移（原则）** | "同一批样本上标注所有任务"的原则与仓库"raw data 不可变、来源/许可显式"兼容；在合成设定中天然满足。低容量读出函数区分"信息存在/易取"的做法可选用于探针设计。 |
| E2-D1 预注册 + 尺寸匹配随机零假设 + 选择性判据 | **可转移（最高优先级）** | 三项全部与仓库预注册文化一致。**关键**：把"尺寸匹配随机对照"作为**非弱化对照**写入注册；把选择的指标/语料/阈值在**看到结果前冻结**。这不仅封堵困境 (c)，也直接回应"额外状态/计算"质疑——匹配的是**同一机制尺寸**的随机对照，而非仅参数总量。需注意其零假设只是噪声带，不要写成 p 值。 |
| E2-D2 测量作为自变量 | **可转移** | 与"冻结筛选阈值、偏离须记为新版本"兼容。**建议**：注册时声明"若判决随指标翻转则记为 INCONCLUSIVE"，把边界案例暴露而非吸收。 |
| E2-D3 可微权重掩码 + IoU/IoMin | **需改造** | 权重级掩码在 CPU 小模型上可行。**须改造点**：AGENTS.md 明确"不得从状态距离推断假设专化"，掩码是**功能性**度量（在目标上训练出的子网），与此约束相容；但仍须避免把掩码重叠直接读成"生物/机制专化"。 |
| E2-D4 I/O 接口操纵 | **可转移（针对困境 d）** | 这是对"硬编码身份/平凡接口解释"的**直接操纵**式反驳装置：构造 shared-I/O 与 separate-I/O 两个版本的任务，若"结构收益"在两版本下都出现，则平凡接口解释被否决。建议纳入 Gate 1 的 mandatory control 清单。 |
| E2-D5 三重零模型 | **可转移** | 随机扰动对照 + 分布零假设 + 未训练模型三条都可低成本实现于 CPU 小模型。与"非弱化对照"要求一致。 |
| E2-D6 未训练模型下界 | **可转移** | 与"不得仅凭生物学 > 随机推断"同源；把专化的零水平从 0 抬到架构先验水平。**注意**：随机初始化与训练后模型的指标不可直接比大小（S10 本身只有定性比较），只能作为**是否存在训练无关结构**的定性闸门。 |
| E2-D7 module criticality | **需改造** | 只用一个已训练模型、无需额外训练，CPU 友好。**与"oracle-relative KL 终点"天然兼容**：把 ε 阈值换成 KL 终点阈值即可得到连续的"从训练值退回初始化多少仍不破坏"。须改造点：路径方向单一（沿 init→trained），须在限制中声明；且作者明确"整体 criticality 而非 critical 模块个数"控制泛化，故**不得**据此声称"模块化带来收益"。 |
| E2-D8 剂量-反应形态 | **需改造** | 可低成本加入。但形态判读须事前给出"悬崖"的可证伪判据（例如相邻剂量间退化量的最大比值阈值），否则会退化为事后解释。 |
| E2-D9 干预式编辑 → 类别级 polysemanticity | **可转移** | 单变量精确干预、无需训练，CPU 友好。**建议**：与 E2-D1 的尺寸匹配零假设结合——把"零化机制指定的单元"与"零化同数量随机单元"的类别级损伤向量直接对比，而不是只报告前者。 |
| E2-D10 跨架构可移植性 + 五类分类法 | **需改造** | 跨模型族复现对本项目（CPU 小模型）成本可控。**须改造点**：本项目主张的是拓扑×动力学原则的抽象存活，因此"配方可移植、电路不可移植"这一结论形态特别适用：可把"五类分类法"改造成"机制主张是否跨种子/跨架构复现"的分级判据，允许 CORRELATE 与 INTERFERER 作为独立结论。 |
| E2-D11 预声明标量专化指数 | **需改造** | MI/熵可作为事前声明的指数，但 S1 与 S9 证明该类**观测性**指数对因果重要性无预测力。故只能用作**描述性**指标；若要用于机制主张，必须配 E2-D1 的干预式对照。 |
| E2-D12 机制性零模型 | **可转移（作为最高强度反驳工具）** | 与"已知生成模型"约束高度契合：在已知生成模型上，可以**推导**出观测指数（如路由/分组使用模式）能被某个平凡量（如表示相似性、隐状态几何、输入统计）完全复现，从而证明该指数的增量解释力为零。**建议**：把"我们能否用一个平凡零模型复现该观测指数"列为 Gate 1 注册中的必备反驳预演。 |

**对四条约束的总体判定**
1. **G1-MATCH-v3 单轴匹配**：E1-D1/D2/D5、E2-D1/D5/D9 与之完全兼容——它们不引入新资源轴，只是把对照做严。E2-D1 的"尺寸匹配随机对照"应被理解为**同一轴上的匹配**（机制尺寸），与"单轴匹配"条款一致而非冲突。
2. **oracle-relative KL 终点**：E1-D2（GT-Modular）与 E2-D7（criticality 阈值换成 KL 阈值）直接支持该终点；建议把 oracle 臂的消耗比一并披露。
3. **CPU 小模型**：E1-D1、E2-D3/D4/D7/D9/D11/D12 在 CPU 小模型上可执行；E1-D3/D8、E2-D1/D2/D10 需要规模或大量训练，**不可直接照搬**。
4. **已知生成模型 + 禁止事后挑任务**：E1-D1 是该约束下最强的装置（结构由生成器给定）；E2-D12 是该约束下最强的反驳工具（可解析地构造零模型）。

## §4 我没能核验的限制

1. **未核验代码**：全部来源停留在 `pdf_section`。S14 与 S16 提供了 GitHub 仓库地址（`aistats2023-1554/moe`、`james-oldfield/muMoE`），但本轮**未取回任何 commit**，故本文件无 `code_commit` 层证据。
2. **未核验数字的复现性**：S5 的相关系数（r = −0.14 / −0.153）、S1 的选择性边界（es +0.002、code −0.009）、S6 的相关性结论均为**作者报告**；我未重跑，也未逐一核对原始表格。
3. **未取得指定场所的正式出版版**：S1、S2、S4、S9、S10、S12、S13、S14、S15、S16、S17 我读到的都是 arXiv 预印本；其会议/期刊正式版（若有）的页眉、审稿后修订与补充材料未核验。S11 的 CVPR 2018 记录仅取到 Crossref metadata 层，未取到会议版 PDF。
4. **未覆盖的装置族**（检索未命中或未深挖，**不等于无人做过**）：
   - 条件计算/模块化的**预注册**实验设计（除 S1 外未找到第二例；`all:"preregistration" AND all:"deep learning"` 在 arXiv 返回 0 条）。
   - 显式的"**任务侧指数 × 架构**交互项"预注册预测——我只找到**交互设计**（E1-D1 的架构 × R），未找到把交互项本身写成预注册统计假设的论文。
   - 神经科学侧的**等损伤消融**（equal-damage lesion）传统：本轮 arXiv/Crossref 检索未有效覆盖该文献群（`all:"lesion" AND all:"specialization"` 被医学影像文献淹没）。
   - S2 的 "Adaptation" 专化指标定义（式 20 附近）我只读到截断处，未完整核验其公式。
   - S9、S12、S1 均为 2026 年预印本，**未经我核验的同行评审**；其"测量依赖"结论的普适性我只能报告为作者主张。
5. **通道导致的不完整**：OpenAlex 免费额度在第二次请求时耗尽（`Insufficient budget`，reset 于 UTC 午夜），故基于 OpenAlex 摘要的召回不完整；DDG/Mojeek/SearxNG/DBLP 触发反爬未使用。检索覆盖以 arXiv 为主，**书籍章节、非 arXiv 会议论文、非英文文献、以及未被 arXiv 索引的期刊论文基本未覆盖**。
6. **不得从本文件推断新颖性**：上述"未命中"仅说明本轮查询未返回结果，不构成任何"无人做过"的结论。
