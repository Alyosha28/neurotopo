# 领域 D 方法审计：连接组/网络神经科学的 null 纪律与"多对一"识别

**审计日期**：2026-09-25
**审计性质**：文献**方法设计装置**提取（design patterns），不是综述、不评价 headline results、不产出新颖性结论。
**证据等级声明**：本轮**无本地复现**。未训练模型、未安装依赖、未运行任何论文代码、未改仓库其他文件。每条来源标注核验层次：`abstract`（仅取回摘要正文）/ `pmc_fulltext_section`（Europe PMC fullTextXML 正文段落）/ `pdf_section`（arXiv PDF 经 `pdftotext` 取正文段落）/ `metadata_only`（仅 Crossref 元数据，未取摘要）。
**检索边界**：实际执行 50 条 Europe PMC 查询、7 条 arXiv 查询、3 条 Crossref 查询（共 60 条），命中并逐条核验 40 个不同来源；逐条见 `dir0003_D_search_log.json`。OpenAlex 因全 IP 免费额度耗尽未使用（返回 `Rate limit exceeded`）。**搜索未命中 ≠ 无人做过**，下述"失效条件"只描述该装置本身的结构性限制。

---

## §1 来源表

| # | 逐字标题 | 年 | 场所 | URL | 核验层次 |
|---|---|---|---|---|---|
| 1 | Null models in network neuroscience | 2022 | Nature Reviews Neuroscience | https://doi.org/10.1038/s41583-022-00601-9 | abstract |
| 2 | Specificity and stability in topology of protein networks | 2002 | Science | https://doi.org/10.1126/science.1065103 | abstract |
| 3 | Unbiased degree-preserving randomization of directed binary networks | 2012 | Physical Review E | https://doi.org/10.1103/PhysRevE.85.046103 | abstract |
| 4 | A Scalable Null Model for Directed Graphs Matching All Degree Distributions: In, Out, and Reciprocal | 2012 | arXiv preprint | https://arxiv.org/abs/1210.5288 | abstract |
| 5 | On the uniform generation of random graphs with prescribed degree sequences | 2003 | arXiv preprint | https://arxiv.org/abs/cond-mat/0312028 | abstract |
| 6 | Are we there yet? When to stop a Markov chain while generating random graphs | 2012 | arXiv preprint | https://arxiv.org/abs/1202.3473 | abstract |
| 7 | A simulated annealing algorithm for randomizing weighted networks | 2025 | Nature Computational Science | https://doi.org/10.1038/s43588-024-00735-z | pmc_fulltext_section |
| 8 | A simulated annealing algorithm for randomizing weighted networks | 2024 | bioRxiv preprint | https://doi.org/10.1101/2024.02.23.581792 | abstract |
| 9 | Weight-conserving characterization of complex functional brain networks | 2011 | NeuroImage | https://doi.org/10.1016/j.neuroimage.2011.03.069 | abstract |
| 10 | Exploring biological network structure with clustered random networks | 2009 | BMC Bioinformatics | https://doi.org/10.1186/1471-2105-10-405 | pmc_fulltext_section |
| 11 | Generative models of the human connectome | 2016 | NeuroImage | https://doi.org/10.1016/j.neuroimage.2015.09.041 | pmc_fulltext_section |
| 12 | Simple models of human brain functional networks | 2012 | PNAS | https://doi.org/10.1073/pnas.1111738109 | abstract |
| 13 | Network motifs: simple building blocks of complex networks | 2002 | Science | https://doi.org/10.1126/science.298.5594.824 | abstract |
| 14 | Superfamilies of evolved and designed networks | 2004 | Science | https://doi.org/10.1126/science.1089167 | abstract |
| 15 | Navigation of brain networks | 2018 | PNAS | https://doi.org/10.1073/pnas.1801351115 | abstract |
| 16 | Specificity and robustness of long-distance connections in weighted, interareal connectomes | 2018 | PNAS | https://doi.org/10.1073/pnas.1720186115 | abstract |
| 17 | A practical guide to methodological considerations in the controllability of structural brain networks | 2020 | Journal of Neural Engineering | https://doi.org/10.1088/1741-2552/ab6e8b | abstract |
| 18 | A null model of the mouse whole-neocortex micro-connectome | 2019 | Nature Communications | https://doi.org/10.1038/s41467-019-11630-x | abstract |
| 19 | Communication dynamics in complex brain networks | 2017 | Nature Reviews Neuroscience | https://doi.org/10.1038/nrn.2017.149 | abstract |
| 20 | Topological Sensitivity in Connectome-Constrained Neural Networks | 2026 | arXiv preprint | https://arxiv.org/abs/2604.04033 | pdf_section |
| 21 | A frozen rate operator from the complete larval connectome: degree and weight govern the gross response, exact wiring governs input routing and mushroom-body modes | 2026 | arXiv preprint | https://arxiv.org/abs/2606.17745 | pdf_section |
| 22 | Degeneracy and complexity in biological systems | 2001 | PNAS | https://doi.org/10.1073/pnas.231499798 | abstract |
| 23 | Measures of degeneracy and redundancy in biological networks | 1999 | PNAS | https://doi.org/10.1073/pnas.96.6.3257 | abstract |
| 24 | Similar network activity from disparate circuit parameters | 2004 | Nature Neuroscience | https://doi.org/10.1038/nn1352 | abstract |
| 25 | Variability, compensation and homeostasis in neuron and network function | 2006 | Nature Reviews Neuroscience | https://doi.org/10.1038/nrn1949 | abstract |
| 26 | Multiple models to capture the variability in biological neurons and networks | 2011 | Nature Neuroscience | https://doi.org/10.1038/nn.2735 | abstract |
| 27 | The promise and perils of causal circuit manipulations | 2018 | Current Opinion in Neurobiology | https://doi.org/10.1016/j.conb.2018.01.004 | abstract |
| 28 | Acute off-target effects of neural circuit manipulations | 2015 | Nature | https://doi.org/10.1038/nature16442 | abstract |
| 29 | Navigating the Neural Space in Search of the Neural Code | 2017 | Neuron | https://doi.org/10.1016/j.neuron.2017.02.019 | abstract |
| 30 | Universally sloppy parameter sensitivities in systems biology models | 2007 | PLoS Computational Biology | https://doi.org/10.1371/journal.pcbi.0030189 | abstract |
| 31 | Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood | 2009 | Bioinformatics | https://doi.org/10.1093/bioinformatics/btp358 | abstract |
| 32 | Measuring and Controlling Solution Degeneracy across Task-Trained Recurrent Neural Networks | 2025 | Advances in Neural Information Processing Systems | https://europepmc.org/article/MED/42111901 | abstract |
| 33 | Chronic vs acute manipulations reveal degeneracy in a thermosensory neuron network | 2021 | microPublication Biology | https://doi.org/10.17912/micropub.biology.000355 | pmc_fulltext_section |
| 34 | The role of connectivity for the degeneracy of the brain's resting state dynamics | 2026 | Journal of Computational Neuroscience | https://doi.org/10.1007/s10827-025-00919-0 | pmc_fulltext_section |
| 35 | Connectome-constrained networks predict neural activity across the fly visual system | 2024 | Nature | https://doi.org/10.1038/s41586-024-07939-3 | pmc_fulltext_section |
| 36 | Learning function from structure in neuromorphic networks | 2021 | Nature Machine Intelligence | https://doi.org/10.1038/s42256-021-00376-1 | metadata_only |
| 37 | Learning function from structure in neuromorphic networks | 2020 | bioRxiv preprint | https://doi.org/10.1101/2020.11.10.350876 | abstract |
| 38 | Connectome-based reservoir computing with the conn2res toolbox | 2024 | Nature Communications | https://doi.org/10.1038/s41467-024-44900-4 | abstract |
| 39 | Optimal modularity and memory capacity of neural reservoirs | 2019 | Network Neuroscience | https://doi.org/10.1162/netn_a_00082 | abstract |
| 40 | Comment on "Network motifs: simple building blocks of complex networks" and "Superfamilies of evolved and designed networks" | 2004 | Science | https://doi.org/10.1126/science.1099334 | abstract |

**与仓库已有记录的对照（以本轮取回内容为准）**
- `evidence_matrix.md` 把 conn2res 记为 Suarez 等 *Nature Communications* 2024、DOI `10.1038/s41467-024-44900-4`。本轮 Crossref 核验该 DOI 确实是 Nature Communications 2024、标题为 *Connectome-based reservoir computing with the conn2res toolbox*（工具箱论文）。但 conn2res 的**概念原始论文**是 #36/#37 *Learning function from structure in neuromorphic networks*（Nature Machine Intelligence 2021，DOI 经 Crossref 核验）。仓库记录未覆盖 2021 论文的方法设计（结构×动力学扫描、临界性、模块组织），只覆盖了 2024 工具箱。
- flyvis 记录（DOI `10.1038/s41586-024-07939-3`、Nature 2024）与本轮 Europe PMC 取回一致；本轮另取回其正文的模型集成与 ablation 段落（见 §2.2）。
- `topology_null_feasibility.md` 对 arXiv 1210.5288 的描述与本轮取回一致：该文标题为 *A Scalable Null Model for Directed Graphs Matching All Degree Distributions: In, Out, and Reciprocal*。本轮补充一点区别：它是 **Chung–Lu 型生成式（期望度）null**，与 #3 的**逐节点精确度序列边交换**是两类不同工具。仓库关于"不能直接替换主 null"的判断属其自身推断，本轮未核验该文实现或许可。

---

## §2 方法设计装置

### §2.1 D1 — null 模型层级与重连纪律

| 装置 | 识别问题（原论文处境） | 具体做法 | 需要什么资源 | 为什么比"两臂比一个标量差"更抗混杂 | 失效条件 |
|---|---|---|---|---|---|
| **N1 精确度序列边交换 null**（#2 Maslov–Sneppen 型；#1 综述归类） | 观测网络的局部/全局属性可能只是度序列的副产品 | 随机重连边对，保持每个节点的度不变，把观测值与重连集成比较 | 一次 O(\|E\|) 级重连 + N 次采样；CPU 可行 | 把"度分布"这一最强竞争解释固定住，剩余差异才可指向度以外的组织 | 度序列本身被结构机制改变时不可用；不守恒二阶结构（见 N2/N4/N5） |
| **N2 无偏采样纪律（detailed balance）**（#3） | 朴素"accept-all"边交换会**有偏**，集成不服从均匀测度，Z 值无意义 | 构造遍历性、满足细致平衡的马尔可夫链，带非平凡接受概率；可推广到目标其他测度（如指定度-度相关） | 需推导接受概率；实现成本高于朴素交换 | 把"null 集成是否真是零模型"从假设变成可验证的构造条件 | 接受概率推导错误或链长不足时仍偏；只对有向二值图给了显式构造 |
| **N3 混合时间/停止准则**（#6；#5） | 连续两次重连不独立，链未混合就采样会系统性偏离目标分布 | #6 用边出现/不出现的二值时间序列估计所需迭代数 N，发现 **N ∝ \|E\|**；#5 比较 switching/matching/go-with-the-winners，指出 matching 有非均匀采样、switching 无一般混合时间界，但偏离在实践中较小 | 需要额外诊断运行；go-with-the-winners 慢，只作校验 | 让"我跑了多少次交换"变成有量化依据的声明，而不是惯例数字 | N ∝ \|E\| 是在"指定联合度分布"下测得的，不外推到其他不变量组合；#5 的结论是经验性的 |
| **N4 强度（加权度）序列保持 null**（#7；#9 加权 null 的早期版本） | 只保持二值度会让权重结构成为混杂；加权指标（rich-club 等）推断会反转 | #7 在重连后的二值骨架上用**模拟退火**重新配置权重位置以匹配强度序列，同时保持权重多重集；与 Rubinov–Sporns 算法及 Maslov–Sneppen 二值重连三方对照 | 模拟退火迭代；#7 对每个网络生成 10 000 个 null 以做集成诊断 | 同时锁住度、强度与权重多重集，把"拓扑 vs 权重"的争论变成可分离的约束层级 | 模拟退火只近似全局最优；论文自陈不同 null 会给出不同结论——即"没有唯一正确 null" |
| **N5 多不变量保守 null**（#10 ClustRNet） | 度序列之外，聚类系数可能才是被误读的属性 | 马尔可夫链生成器，同时指定**度序列 + 聚类水平**（局部/全局、相对/绝对），其余随机 | 每张图一个优化链；作者提供 Python 实现 | 逐个增加"被控制的替代解释"，把混杂来源显式化 | 高聚类+给定度序列可能不可同时实现；生成的是"指定水平"而非"实证水平"的族 |
| **N6 生成模型 null 族**（#11 十三模型；#12 距离惩罚+拓扑项；#18 微观连接组生成模型） | 单纯随机化无法回答"哪些生成规则足以产生观测拓扑" | 枚举一族参数化生成模型，用多目标优化拟合实证网络的多个属性；#11 检验**未参与优化的属性**是否也被复现；#12 在**第二个独立样本**上检验第一样本估出的参数 | 需要多目标优化 + 至少两个独立数据集 | 预测**留出属性/留出样本**比"拟合优度更高"更难被容量解释；未被拟合的属性被复现是真正的非混杂证据 | 生成模型族本身不完备；参数可在不同数据集间漂移（#11 报告随年龄渐进变化），单点参数比较不可靠 |
| **N7 分层 null 阶梯 / control ladder**（#20；#21；#1 的 liberal→conservative 谱） | 单个 null 一旦同时改变多个图属性（跨块边、modularity、互反边、谱），效应无法归因 | 把 null 排成**单调严格化**的阶梯，在每一级重算效应；#20 三级（checkpoint 初始化+朴素随机 → 共享随机初始化+朴素随机 → 共享随机初始化+度序列保持），并配一个**预训练活动尺度诊断**；#21 四级（无结构 Gaussian → 度+权重匹配 n=1000 → 块（细胞类）保持 n=1000 → 连接组本体） | 每级一个集成；#20 只用 5 样本度保持集成；#21 用 1000 样本 | 报告的是"效应残差随约束增强的衰减曲线"，而非单点差值；哪一级消失就直接指出该效应由哪个被控属性解释 | 阶梯各级的图空间不同，同级之间不可比大小；仍无法排除阶梯外的未控属性（#21 自陈其 null 对"入向加权质量"是弱检验） |
| **N8 随机化轨迹剂量-反应**（#15） | 二元"原图 vs 随机图"无法区分"轻微破坏即失效"与"只有完全随机才失效" | 沿随机化/聚类化轨迹渐进推进，观察导航效率的连续变化 | 一路重连 + 每步评估 | 用连续剂量曲线替代阈值判定；还能反向说明"存在最优区间"而非单调更好 | 轨迹路径依赖（换重连算子会得到不同曲线）；只适用于对图扰动连续敏感的功能量 |
| **N9 显著性谱/跨 null 家报告**（#13；#14；#16） | motif 计数、长程连接特异性等的单点 Z 值受 null 选择影响 | 用**同一个随机化族**给全部子图算 Z 值，构成 significance profile；跨网络比较 profile 相似性；#16 用零模型判定长程连接的连接谱相似度是否非随机 | 子图枚举/采样 + 集成 | 把"哪个属性显著"压缩成一个向量后跨系统比较，减少逐属性挑选空间 | SP 对 null 族敏感；#14 发表后即有正式 comment 质疑（#40，本轮取回该 comment 的标题与年，未取正文） |
| **N10 建模选择敏感性审计**（#17） | 同一数据在不同建模选择下指标会变，导致结论不稳 | 系统扫描建模选择（如结构连接定义、径向传播），报告四个常用可控性统计的变动 | 重算成本低，但需要多重实现 | 把"方法学自由度"本身列为不确定度来源，给出结论的稳定区间 | 只审计了被枚举的那些选择；未枚举的默认选择仍是隐含自由度 |

**四点关键细节（取回原文段落）**
- #7 定位自身为"在保持二值度序列之外还保持强度序列"，并自陈"准确保持强度会对脑网络组织得出不同推断"；它同时用 **morphospace**（特征路径长度 × 聚类）嵌入 10 000 个 null 来评估集成变异，这是"派生集成是否只覆盖了实证网络附近"的显式检查。
- #10 的目标被明确表述为提供"degree 与 clustering 的副产品之外的"随机对照，即**逐个消解替代解释**，而非一次替换整个图。
- #20 原文（`pdf_section`，arXiv:2604.04033v1，第 1 页）："We then progressively remove two key confounds. First, we eliminate checkpoint bias by training both graphs from a shared random initialization. Second, we strengthen the null model by replacing the naive random graph with a degree-preserving rewired graph that matches directed in-degree and out-degree sequences in addition to global graph statistics."
- #21 原文（`pdf_section`，arXiv:2606.17745v2，Results）："First, the matched ensemble reproduces the connectome's non-normality ratio (1.31 against 1.325), so the deeper confinement cannot follow from how strongly the operator amplifies." —— 这是"先用 null 排除一个**具体**竞争机制、再声明剩余属性"的写法，而不是只报一个 p 值。

### §2.2 D2 — 退化/多对一与必要性-充分性主张

| 装置 | 识别问题 | 具体做法 | 需要什么资源 | 为什么比"两臂标量差"更抗混杂 | 失效条件 |
|---|---|---|---|---|---|
| **G1 退化 vs 冗余的可操作度量**（#23；#22） | "结构不同却同功能"（退化）与"相同元素重复"（冗余）被混为一谈，导致"结构决定功能"被过度解读 | 用信息论量定义系统相对一组输出的退化与冗余；对连接不同的模拟神经系统验证两度量可分 | 需要一组输出/扰动集合与互信息估计 | 把"多对一"从定性修辞变成可计算量，并给出"退化低/高"的判别性预测 | 依赖输出集合与粗粒化选择；概念性主张，不是对具体实验的检验 |
| **G2 模型总体而非单一模型**（#24；#26） | 单一拟合模型会被误当作唯一机制 | #24 模拟 2 000 万+ 参数组合，证明"几乎不可区分的网络活动来自差异巨大的机制组合"；#26 主张建**模型群体**以匹配实验数据的跨动物变异，并从群体中找未被怀疑的代偿机制 | 大规模参数扫描（#24 是 2×10^7 量级）；CPU 集群 | 报告的是"同一输出水平上的解集几何"，天然把"唯一结构"排除；代偿机制是从群体差异中**发现**的，不是被假定的 | 扫描范围界定了结论边界；群体成员不等于生物学可行解 |
| **G3 集成的共识预测 + 外部实验外推**（#35） | 结构约束可能不唯一确定参数，单一最优解的活动预测不可信 | 用同一连接组生成 **50 个模型集成**（各自是任务性能的局部最优），只把**集成预测**与 26 项独立实验研究的实测活动比对；并做随机参数集与"仅解码器"对照 | 50 次端到端训练；外部实测数据集 | 检验的是**样本外**预测一致性，而不是拟合优度；"许多参数都能解释"被显式承认为前提而非缺陷 | 只在有实测活动的系统可用；集成内一致性不等于唯一性，且论文自陈稀疏连接是成功条件 |
| **G4 消融阶梯（ablation ladder）**（#35） | "连接组模型能预测"可能是任务优化、随机参数或其他建模假设带来的 | 用同一框架构造一族模型：full DMN / 取消任务优化的随机 DMN / 仅解码器 / 随机参数配置，逐项比较 | 每项一个集成 | 逐个移除**单一**建模要素，效应可归到被移除要素；与 §2.1 N7 同构 | 只覆盖被枚举的要素；要素间可能交互 |
| **G5 急性 vs 慢性扰动分离必要性与代偿**（#33；#28） | 急性失活造成的功能丧失会被读成"该元件必要"，但永久损伤后功能常自行恢复 | #33 对同一批神经元先急性、后慢性沉默，比较行为输出；#28 对同一区域比较瞬时失活与永久损伤，并追踪下游核团的偏离效应随天数消退 | 同一制备上的两种时间尺度干预 + 纵向测量 | 用**时间尺度**作为额外因子，把"必要性"与"可代偿性"分开；恢复轨迹本身就是代偿存在的证据 | 需要可长期存活的制备；物种/回路特殊；恢复也可能来自未测量的外周或状态变化 |
| **G6 必要性/充分性主张的条件化框架**（#29；#27） | 扰动实验常被直接读作因果结论，忽略了尺度不匹配、间接效应与分布式功能 | #29 明确测量与扰动支持因果推断的条件，并给出"扰动是否尊重系统内在活动模式"的几何框架；#27 系统列出电路扰动在行为动物中的常见隐含假设及其局限 | 概念框架，不需新硬件 | 把"该干预能支持哪一类主张"变成设计前必须回答的问题，而不是事后解释 | 是框架而非判定程序；不能替代具体对照 |
| **G7 结构不可辨识 vs 实践不可辨识**（#30；#31） | 观测数据不足以确定参数，但常被误读为机制确定 | #31 用 profile likelihood 区分**结构性**不可辨识（参数函数相关）与**实践性**不可辨识（数据量/质量不足），并给出置信区间用于实验设计与模型约简；#30 证明大量系统生物学模型的参数敏感度谱是"sloppy"的（特征值跨多个数量级） | 需要似然/目标函数可重复评估；profile likelihood 计算量随参数数增长 | 直接回答"这个数据能不能识别这个机制"，而不是"哪个模型拟合更好"；sloppy 谱解释了为何集体拟合仍可预测却单参数无约束 | 依赖模型形式正确；sloppy 是系统生物学的经验规律，外推到神经回路需自证 |
| **G8 解退化的三层量化与控制**（#32） | 同一任务上表现相近的 RNN 内部解差异巨大，"反工程单个网络"得到的机制可能不可泛化 | 在**行为、神经动力学、权重空间**三层上统一定义解退化并测量；对 3 400 个 RNN（四个神经科学任务）系统改变任务复杂度、学习区制、网络规模与正则化 | 3 400 次训练 | 报告的是"解分布"而非单点机制；把网络规模/正则化作为**控制退化的旋钮**，即把"多对一"改造成设计变量 | 任务集有限；结论是"如何调退化"，不回答某机制是否必要 |
| **G9 "哪些属性由粗统计决定、哪些需要精确放置"的分解**（#21；#20；#34） | 连接组模型同时继承度、权重与精确布线，无法说明"布线在统计之外还固定了什么" | #21 用 null 阶梯把每个动力学属性归入"度+权重即可复现"/"需要细胞类结构"/"需要精确放置"三类；#20 用严格化控制阶梯把表观优势归因为初始化与 null 设计；#34 用脑网络建模识别静息态流形上的退化轮廓，并检验扰动下哪些模式被保留 | #21 每类 null 1 000 样本；#34 噪声驱动探索流形 | 输出的不是"结构重要/不重要"，而是**属性级的归因表**；每一项都有对应的、更强的对照 | 属性集合由研究者选定；#20/#21 为未经同行评审的预印本，其中 #20/#21 各为单作者 |

---

## §3 对 NeuroTopo 的可转移性逐条判定

| 装置 | 判定 | 与现有仪器的兼容性 |
|---|---|---|
| N1 精确度序列边交换 | **不可转移（在本项目当前图族上）** | 固定四等大分区、逐节点固定入/出度时，`Q=0.75−crossEdges/1024` 是精确恒等式：度保持重连无法在保持 Q 的同时改变跨块边。必须改换不变量集合（如固定跨块边数、随机化互反结构与块内放置）才可用。 |
| N2 无偏采样纪律 | **需改造** | 不必自己推导细致平衡；但若沿用任意现有实现，必须把"接受概率来源 + 目标测度"写进注册。与 G1-MATCH-v3 无关（这是图族构造，不是资源轴）。 |
| N3 混合时间/停止准则 | **可转移（低成本）** | 直接按 N ∝ \|E\| 定交换次数并报告；在 CPU 小模型上成本可忽略。属于注册时必须固定的协议参数。 |
| N4 强度序列保持 null | **需改造** | 本项目主对照若是二值掩码，则权重不在图族内；只有当机制声明"权重与拓扑可分"时才需要强度保持 null。与"已知生成模型"兼容：可把强度保持作为生成模型的一层约束。 |
| N5 多不变量保守 null | **可转移（作为阶梯的一级）** | 与 N7 合并使用：在固定分区与度数的恒等式约束下，唯一能独立变化的不变量需要显式列出（互反结构、块内放置、权重多重集）。这正好把"恒等式"从障碍变成设计约束。 |
| N6 生成模型 null 族 | **需改造** | 与"已知生成模型"最兼容的装置。可用留出属性/留出种子做预测性检验，替代两臂标量差。但在 CPU 小模型下多目标优化成本需压缩到少数模型、少量目标。 |
| N7 分层 null 阶梯 | **可转移（最高优先）** | 与"oracle-relative KL 终点"兼容：在阶梯**每一级**重算 KL，报告 KL 残差随约束增强的曲线，而不是单一 Δ。与 G1-MATCH-v3 兼容的前提是把阶梯声明为**图内对照**，不占用也不混淆资源轴；每一级必须固定参数、状态维度、内部步数与更新次数，否则会把容量通道重新引入。 |
| N8 随机化轨迹剂量-反应 | **需改造** | 与 KL 终点兼容（沿轨迹画 KL 曲线）。需先证明轨迹上除目标不变量外的协变量单调可控，否则退化为本项目已否决的"多属性同时变化"。 |
| N9 显著性谱/跨 null 家报告 | **不可转移（现阶段）** | 本项目不是"属性是否非随机"的问题，而是"结构机制是否带来可归因收益"。SP/Z 值无法回答资源匹配下的归因，且会诱导把"图属性非随机"当作收益证据。 |
| N10 建模选择敏感性审计 | **可转移（低成本）** | 对 M3/C0 与 G1-MATCH-v3 的声明轴做同样的敏感性审计（换匹配轴定义、换 KL 估计设置），报告结论是否翻转。CPU 小模型可承受。 |
| G1 退化/冗余度量 | **需改造** | 概念上直接对应"许多结构实现同一功能"。但要落到本项目需要一个**输出集合**（多任务/多扰动）与互信息估计，成本不低。可先作为**注册时的主张改写规则**：在没有输出集合度量前，不得声称结构唯一决定功能。 |
| G2 模型总体而非单一模型 | **可转移（CPU 可承受的缩水版）** | 与"CPU 小模型"兼容：以数十个种子/参数样本替代 2×10^7 扫描，报告"同一性能水平上的解集"而非单点机制。与 oracle-relative KL 兼容（对解集报 KL 分布）。 |
| G3 集成共识 + 外部预测 | **需改造** | 需要**留出**终点或留出任务；在已知生成模型的合成任务上可直接做（留出任务/留出延迟区间）。这是把"两臂比标量"换成"样本外预测一致性"的最低成本路径。 |
| G4 消融阶梯 | **可转移** | 与 G1-MATCH-v3 兼容：每次只移除一个建模要素，其余保持匹配；须逐级声明匹配轴是否改变。 |
| G5 急性 vs 慢性扰动 | **不可转移（生物学实验）** | 本项目无生物学制备。可迁移的是**逻辑**：把"必要性"与"可代偿性"分成两个不同对照，对应到训练中的"移除机制后立即评估"与"移除后重新训练到收敛"两臂。 |
| G6 必要性/充分性条件化 | **可转移（作为注册纪律）** | 零资源成本。要求在协议中先写明该对照支持 necessity 还是 sufficiency，以及分布外推的边界。 |
| G7 sloppy 谱 / profile likelihood | **可转移（高优先）** | 直接针对本项目"容量通道 vs 优化难度通道"混合问题：对候选机制做 profile-likelihood 式扫描，区分"结构不可辨识"（图族内不存在可区分该机制的终点）与"实践不可辨识"（训练预算不足）。CPU 小模型下可行。 |
| G8 解退化三层量化与控制 | **可转移（高优先）** | 与"oracle-relative KL 终点"和"CPU 小模型"都兼容：把行为/动力学/权重三层退化作为**结果变量**报告，并把网络规模与正则化作为显式旋钮。 |
| G9 属性级归因表 | **可转移（与 N7 同一实施）** | 与"已知生成模型"和"固定分区恒等式"最契合：在恒等式约束下，输出"哪些终点差异由粗统计解释、哪些必须由精确放置解释"的属性级表，替代单轴 Δ。 |

**与 `G1-MATCH-v3` 的边界（单独强调）**：§2 全部装置都是**图族/对照构造**技术，不构成资源匹配声明。任何把这些装置用于 Gate 1 的用法，必须把阶梯每一级都固定在同一已声明的消费轴上（参数、状态、内部步数、更新次数在级间不变），并声明该比较是图内对照；否则会同时改变资源轴，退回 `PACKAGE_EFFECT` 轨道。

---

## §4 本轮未能核验的限制

1. **#1（Váša & Mišić 2022）只有摘要**：Europe PMC 无 PMCID、无全文，其"liberal→conservative null 分类学"的**具体分类条目与判据未核验**，§2.1 N7 中对该谱的引用仅来自摘要措辞。
2. **conn2res 的 2021 原始论文（#36）只核到 Crossref 元数据**；方法细节（结构×动力学扫描、临界性、模块组织）来自 2020 bioRxiv 预印本摘要（#37），**不是**期刊版正文。仓库已记录的 2024 工具箱论文（#38）本轮只核到摘要，未读源码。
3. **flyvis 代码未核验**：本轮只读 Nature 正文（#35），未检查仓库提交、测试或实现；#35 的"50 模型集成""26 项研究"均来自正文段落，未独立复算。
4. **#2 Maslov–Sneppen 原文方法段未核验**：只有摘要（含"all links were randomly rewired"的措辞），未读其重连次数/收敛讨论。
5. **#20、#21 是未经同行评审的预印本**，且均为单作者（#20 Nalin Dhiman；#21 Stavros Therianos）。它们的装置逻辑清晰可复用，但其**结论数值未获独立验证**，本审计只提取设计，不背书其结论。
6. **arXiv API 在本轮后段出现 429**：`all:"reciprocal degree sequence"` 查询返回 0 条，同一主题改以 `arxiv.org/abs/1210.5288` 页面直取成功。该条 0 命中是限流/短语匹配问题，**不能**作为"无人做过"的证据。
7. **多条查询 0 命中**（`TITLE:"necessity" AND TITLE:"sufficiency" AND TITLE:"optogenetic"`、`TITLE:"many-to-one" AND TITLE:"structure" AND TITLE:"function"`、`TITLE:"geometric" AND TITLE:"null model" AND TITLE:"network"` 等）反映的是 Europe PMC 的字段化标题检索限制，不代表文献空白。
8. **无 `code_commit` 层核验**：本轮尝试核对 BCTpy（仓库 `topology_null_feasibility.md` 提及）的 `randmio` 源码未成功（GitHub API 未返回结果，raw 文件取回为空），因此**关于任何具体软件实现保留哪些不变量的陈述，本轮一律不写**。
9. **未做本地复现**：所有"装置"的复现成本估计是本审计基于原文描述与 CPU 可行性的判断，不是实测。
10. **未穷尽**：48 条 Europe PMC + 6 条 arXiv + 2 条 Crossref 是**便利抽样**，不是系统综述；尤其"报告一族 null 上剩余效应"的期刊级最佳实践、空间嵌入 null、以及 lesion/代偿在人类网络中的分析设计，本轮覆盖不完整。
