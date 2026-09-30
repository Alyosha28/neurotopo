# 领域 G 方法审计：拓扑/耦合 → 计算能力的**定量、可证伪预测**

**审计日期**：2026-09-25
**审计性质**：文献**方法设计装置**提取（design patterns），不是综述、不评价 headline results、不产出新颖性结论。
**证据等级声明**：本轮**无本地复现**。未训练模型、未安装依赖、未运行任何论文代码、未改仓库其他文件。核验层次：`abstract`（仅取回摘要正文）/ `pdf_section`（arXiv PDF 经 `pdftotext` 或出版社开放正文取段落）/ `pmc_fulltext_section`（Europe PMC fullTextXML 正文段落）。
**检索边界**：实际执行 Europe PMC 查询 21 条、arXiv 查询 8 条、Crossref 查询 8 条，另对 4 篇 arXiv PDF 与 2 篇 Nature 开放正文做直连全文取回；命中并逐条核验 **20 个不同来源**（其中 7 个核到正文段落）。逐条见 `dir0003_G_search_log.json`。
**不可用通道**：OpenAlex 全 IP 共享免费额度耗尽（`Rate limit exceeded`，`dailyRemainingUsd: 0.0008`）；DuckDuckGo/Mojeek/SearxNG/DBLP 触发反爬；Semantic Scholar 429。**因此覆盖面偏 Europe PMC + arXiv，对 ICLR/NeurIPS 类会议文献覆盖弱。搜索未命中 ≠ 无人做过。**

---

## §1 来源表

| # | 逐字标题 | 年 | 场所 | URL | 核验层次 |
|---|---|---|---|---|---|
| 1 | Information Processing Capacity of Dynamical Systems | 2012 | Scientific Reports 2:514 | https://doi.org/10.1038/srep00514 | pdf_section |
| 2 | Short-Term Memory in Orthogonal Neural Networks | 2004 | Physical Review Letters 92:148102 | https://arxiv.org/abs/cond-mat/0402452 | pdf_section |
| 3 | Optimal modularity and memory capacity of neural reservoirs | 2019 | Network Neuroscience 3(2):551–566 | https://doi.org/10.1162/netn_a_00082 | pmc_fulltext_section |
| 4 | Neuromorphic hierarchical modular reservoirs | 2026 | Nature Communications | https://doi.org/10.1038/s41467-026-74466-2 | pdf_section |
| 5 | Mutual Information and the Edge of Chaos in Reservoir Computers | 2019 | arXiv:1906.03186 | https://arxiv.org/abs/1906.03186 | pdf_section |
| 6 | Do Reservoir Computers Work Best at the Edge of Chaos? | 2020 | arXiv:2012.01409 | https://arxiv.org/abs/2012.01409 | pdf_section |
| 7 | Beyond the Edge of Chaos: Stability–Expressivity Transfer in Reservoir Forecasting | 2026 | arXiv:2607.17909 | https://arxiv.org/abs/2607.17909 | pdf_section |
| 8 | Memory traces in dynamical systems | 2008 | PNAS 105:18970–18975 | https://doi.org/10.1073/pnas.0804451105 | abstract |
| 9 | Edge of chaos and prediction of computational performance for neural circuit models | 2007 | Neural Networks 20:323–334 | https://doi.org/10.1016/j.neunet.2007.04.017 | abstract |
| 10 | Real-Time Computation at the Edge of Chaos in Recurrent Neural Networks | 2004 | Neural Computation 16:1413–1436 | https://doi.org/10.1162/089976604323057443 | abstract |
| 11 | Beyond the edge of chaos: amplification and temporal integration by recurrent networks in the chaotic regime | 2011 | Physical Review E 84:051908 | https://doi.org/10.1103/physreve.84.051908 | abstract |
| 12 | A small-world topology enhances the echo state property and signal propagation in reservoir computing | 2019 | Neural Networks 112:15–23 | https://doi.org/10.1016/j.neunet.2019.01.002 | abstract |
| 13 | A mean-field approach to criticality in spiking neural networks for reservoir computing | 2025 | Scientific Reports 15:34709 | https://doi.org/10.1038/s41598-025-18004-y | abstract |
| 14 | Structural determinants of soft memory in recurrent biological networks | 2025 | Biophysical Reviews 17:259–269 | https://doi.org/10.1007/s12551-025-01295-w | abstract |
| 15 | Deriving task specific performance from the information processing capacity of a reservoir computer | 2023 | Nanophotonics 12:937–947 | https://doi.org/10.1515/nanoph-2022-0415 | abstract |
| 16 | Memory-Non-Linearity Trade-Off in Distance-Based Delay Networks | 2024 | Biomimetics 9:755 | https://doi.org/10.3390/biomimetics9120755 | abstract |
| 17 | Illuminating the black box of reservoir computing | 2026 | Scientific Reports 16:15500 | https://doi.org/10.1038/s41598-026-53098-y | abstract |
| 18 | The Fundamental Tension in Topological Quantum Reservoir Computing: Why Unitarity Opposes the Echo State Property | 2025 | SSRN preprint | https://doi.org/10.2139/ssrn.5914523 | abstract |
| 19 | Edge of Many-Body Quantum Chaos in Quantum Reservoir Computing | 2026 | Physical Review Letters | https://doi.org/10.1103/j2qj-vwcl | abstract |
| 20 | Higher-order interactions, adaptivity, and phase transitions in a novel reservoir computing model | 2025 | Chaos 35 | https://doi.org/10.1063/5.0278312 | abstract |

---

## §2 方法设计装置（核心）

七个装置都满足"功能量 = 拓扑/耦合参数的显式函数 + 带符号或带形状的预测"。每条给出：识别问题 → 做法 → 资源 → 抗混杂机制 → 失效条件 → **训练前可冻结性**。

### D-G1 谱 → 记忆容量解析映射
- **源**：White/Lee/Sompolinsky 2004（#2）；Ganguli/Huh/Sompolinsky 2008（#8，仅摘要，未见正文公式）
- **识别问题**：把"记忆好"从事后任务分数反推，改为由连接矩阵的**谱**直接算出功能量。
- **做法**：线性递推 `x(n)=Wx(n-1)+vs(n)+z(n)`，读出 `u_k=C⁻¹p_k`。DSR 网络记忆函数 `m(k)=α^k/(α^k+ε_e(1-α^{k+1}))`；随机正交网络 `m(k)=α^k q/(1+α^k q)`，`q` 由自洽方程求得。容量 `k_C` = 使 `m(k)<1/2` 的最小 k；作者报告 `k_C ∝ N`（extensive），并给出 `α_opt=0.98`。
- **资源**：一次 `O(N³)` 谱分解 + 噪声水平，**不需要训练**。
- **抗混杂**：预测是**函数形式 + 尺度 + 非单调形状**（容量对 α 与噪声的联合曲线），不是两臂标量差。α 与噪声 `ε` 进入同一个闭式，天然产出**二维相图**而非单点比较。
- **失效条件**：仅线性网络。Rodriguez 2019（#3）表明模块化效应在 tanh 与 threshold 单元下**符号相反**，故该解析式不可外推到非线性核。正交网络结果依赖"退火近似"（W、v 每步重抽）。
- **训练前可冻结**：✅ **是**。谱量与闭式在构造后、训练前即可算出并冻结。

### D-G2 总容量守恒 + 按基函数阶数分解（IPC）
- **源**：Dambre/Verstraeten/Schrauwen/Massar 2012（#1，正文已核）
- **识别问题**：如何把"表征质量"写成**任务无关**的量，并从"某结构更好"升级为"某结构把容量放在哪个子空间"。
- **做法**：驱动动力系统 + 线性读出；用归一化 Legendre 多项式积张成 fading-memory Hilbert 空间；单项容量 `C_l = 1 − ‖y_l−ŷ_l‖²/‖y_l‖²`。**Theorem 7**：fading memory 且 N 个内部变量线性无关时，`Σ_l C_l → N`。正文并给出有限数据截断规则（ESN 用 θ=1.7×10⁻⁴ 等具体阈值）与噪声下容量向信号子空间投影的衰减分析。
- **资源**：CPU 可做（正文 ESN 用 N=50）；`T=10⁶` 时间步仿真 + 一次伪逆/岭回归读出；**不改内部核**。
- **抗混杂**：总量 `N` 是**与结构无关的守恒上界**。任何"某拓扑更好"的主张只能表现为**容量在基函数空间的重新分配**（memory vs nonlinearity 权衡曲线）。若测得的"改进"让总量超过 N，则按论文的截断规则判为有限数据高估——这是一个**可证伪的守卫条件**。
- **失效条件**：需要 i.i.d. 输入与可构造的正交基。正文自陈：非 i.i.d. 序列时正交基"通常不可行"，上界"主要是理论洞见"。**关键外部反证**：Hülser 等 2023（#15）报告**总 IPC 与任务特定性能相关性很差** → IPC 不可直接充当任务性能代理。
- **训练前可冻结**：✅ 定理与上界解析；容量度量只需对已冻结的核做一次前向仿真。

### D-G3 边界/临界点作为**带形状**（非单调）预测的自变量
- **源**：Bertschinger & Natschläger 2004（#10）；Legenstein & Maass 2007（#9）；Toyoizumi & Abbott 2011（#11）
- **识别问题**：把"增益该调多少"从超参搜索改为**由动力学不稳定性边界给出的定位预测**。
- **做法**：Legenstein & Maass 明确限定其预测效力——edge of chaos **预测哪些参数值给出最大性能，但不预测其他参数值下的性能大小**；Toyoizumi & Abbott 解析给出边缘附近解码性能由**临界指数**刻画，且**两侧指数取值不同**：奇数饱和非线性下，混沌侧的增益与积分时间衰减**慢得多**。
- **资源**：Lyapunov 指数/谱 + 解析推导；定位最优参数不需训练。
- **抗混杂**：预测形态是**"峰值位置 + 两侧不同幂律指数"**。两侧指数不等是一个强制的形状约束，两臂标量比较**无法**产生这种约束，因而也无法伪装成满足它。
- **失效条件**：Carroll 2019（#5）与 Carroll 2020（#6）给出直接反例：容量/性能可在**趋近边界时下降**，机制是广义同步破坏、或 reservoir 与训练信号 Lyapunov 谱重叠/频率响应失配。Carroll 2020 明确以"edge of stability"取代"edge of chaos"，并声明该规则"not in general true"。
- **训练前可冻结**：✅ 临界点与指数可解析/半解析算出。

### D-G4 三体制 Lyapunov 谱 + stability–expressivity transfer index
- **源**：Du & Wang 2026（#7，正文已核）
- **识别问题**：谱半径扫描要"大量算力 + 额外验证数据，且给不出动力学洞见"。能否**在训练前**用一个被驱动动力学的标量定位最优？
- **做法**：区分同一 recurrent core 的**三种体制**——孤立、teacher-forced、闭环生成——并各自求 Lyapunov 谱；作者证明最优预报谱半径与**三种边界都不重合**。随后对 teacher-forced reservoir 求**有限时间** Lyapunov 谱，发现目标动力学主要由**平均稳定但在有限时间被输入强烈调制**的模式承载；据此构造 stability–expressivity transfer index，在混沌与准周期目标、对称与非对称 reservoir 共 4 种组合上定位最优谱半径。
- **资源**：teacher-forced 前向轨迹 + QR 求谱；**不需要 readout 训练**。
- **抗混杂**：指标**训练前、只用被驱动动力学算出**，且跨目标类型 × 网络对称性报告；预测对象是"最优谱半径"这一**连续量的位置**，不是二值胜负。
- **失效条件**：作者只声明适用于 autonomous forecasting；对分类、记忆容量等其他任务类型未验证。仍是单一标量指标，作者亦未给出其失效边界。
- **训练前可冻结**：✅ **是**（teacher-forced 前向仿真即可）。

### D-G5 结构保持零模型系综（本领域对"识别性"最直接的装置）
- **源**：Milisav 等 2026（#4，正文已核）
- **识别问题**：**这正是 NeuroTopo 的绑定困难**——如何把"层级模块化"的效应与"任意稀疏有向图""度数序列""环数"等替代解释分开。
- **做法**：用生成式 block model 造 1/2/3 级层级模块网络（N=400 有向，无自环，正权，层间连接每级减半 `r=0.5`），并**同时**构造四类零模型：(a) degree-preserving rewiring（`randmio_dir_connected`）；(b) modularity-preserving（按 within/between 边类别分别 rewiring）；(c) **hierarchical-modularity-preserving**（L 层嵌套划分 → L+1 个边类别分别 rewiring）；(d) cycles-preserving（模拟退火，代价函数为 3/4 环计数平方差）。报告量同时给 p 值与 **CLES**（common language effect size）。敏感性分析覆盖 N∈{200,600}、r∈{0.25,0.75}、对称/非对称、含/不含负权。
- **关键用法**：层级模块 > 严格模块 > random（p<10⁻⁹，CLES≥75.75%）；但**与 cycles-preserving null 打平** → 作者据此**主动收窄机制主张**为"低阶环结构"，并明说"层级性不能完全解释收益"。
- **资源**：CPU 完全够（N=400 有向 × 100 网络/系综 × 2100 时间点训练+测试）；需 bctpy + 自写 rewiring。
- **抗混杂**：把"结构效应"重新定义为**在保持指定图统计量的零模型分布上的位移**；并且**当一个更弱的零模型就能解释掉效应时，放弃更强的机制主张**。这直接对治任务选取混杂与机制过度归因。
- **失效条件**：作者自陈加入负权 `W~U(−1,1)` 后**层级各级之间无显著差异**，即效应依赖权重符号约束；输入/输出固定在"第一级模块"这一放置本身是设计选择；作者明确"目标不是提出新的 SOTA 架构"。
- **训练前可冻结**：⚠️ 零模型系综与环数统计可在训练前生成并冻结；性能统计量需跑任务。

### D-G6 拓扑效应的**符号反转** = 把主效应改写成交互
- **源**：Rodriguez/Izquierdo/Ahn 2019（#3，正文已核）；#4 对其结论的复述
- **识别问题**：**"拓扑→功能"没有无条件符号**。在有混杂风险时如何给出可证伪预测？
- **做法**：用 LFR benchmark 生成模块化 reservoir（N=500，每节点度 6），**固定度分布与总边数**，只扫桥接边比例 μ；额外控制谱半径与权重尺度。对比 step-like 激活（广义 sigmoid，k=10）与 linear/tanh。结果：threshold-like 单元下 μ≈0.1–0.2 出现**最优模块度**（倒 U 形）；linear/tanh 下模块化**降低**记忆容量（社区成为信息瓶颈）。作者原文：最优模块度现象"emerges only for neural networks of threshold-like neurons and does not exist for neural networks of linear or tanh neurons"。
- **资源**：CPU 可做；需对每类激活函数各跑一遍。
- **抗混杂**：把"拓扑主效应"改写为**"拓扑 × 单元非线性"的交互**，并给出**两侧符号**（threshold: +，tanh: −）。所有臂在度分布、总边数、平均强度、谱半径上对齐 → 是逐轴匹配的雏形。
- **失效条件**：依赖 LFR（低聚类、低最短路）而非 Watts–Strogatz 拓扑；作者自陈"weight scale 仍是决定性能的最大因素"；未涵盖 STDP/神经调质。与 #4 存在**明显张力**：#4 在 tanh reservoir 中报告层级模块 > 严格模块 > random，而 #3 报告 tanh 下模块化降低 MC——这一张力本身即可检验。
- **训练前可冻结**：❌ 需任务运行；但**交互的符号方向**可预注册。

### D-G7 训练前解析定位工作点（消除"调参预算"这条混杂通道）
- **源**：Freddi/Cicala/Marzetti/Basti 2025（#13）；相邻装置见 Kawai/Park/Asada 2019（#12）
- **识别问题**：临界工作点通常靠**试错或可塑性自适应**寻找 → 各臂的调参预算不可比，构成额外的资源混杂通道。
- **做法**：对 LIF 脉冲神经元推导并求解**平均膜电位演化的均值场方程**，给出临界点的近似解析式；在"高度一般拓扑"的 SNN reservoir 上按解析式直接配置到临界；用 Lempel-Ziv-Welch 复杂度量化临界附近的计算容量；再以局部稳态学习规则探索自组织准临界。
- **资源**：均值场方程求解很便宜；验证需 SNN 仿真。
- **抗混杂**：**把"调到最优工作点"变成训练前的解析给定**，使后续各臂共享同一、非搜索得来的工作点 → 直接移除"调参预算"这一通道。这是对 `G1-MATCH-v3` 单轴匹配的天然补充：匹配轴之外的超参由解析式统一钉死。
- **失效条件**：LIF 特定、平均场近似；LZW 复杂度是**代理量**而非任务性能（参见 #15 的警告）。
- **训练前可冻结**：✅ 临界点解析近似。

### D-G8 记忆–非线性二维报告（强制容量守恒记账）
- **源**：Dambre 2012（#1，正文已核）；Iacob & Dambre 2024（#16）
- **识别问题**：容量不是标量；任何"提升记忆"的操作很可能只是把容量从非线性搬到线性。
- **做法**：把容量归一化为基函数上的分布，定义线性记忆 `L[C]`（仅线性项占比）与非线性 `NL[C]`（按总阶数加权）；在参数扫描下画互补曲线。作者原文结论："the linearity L and non-linearity NL are complementary … a fundamental tradeoff"。距离基延迟网络（#16）用同一框架分析 MC 与非线性。
- **资源**：与 D-G2 同（固定核 + 闭式读出）。
- **抗混杂**：**强制报告二维（memory, nonlinearity）而非一维任务分数** → 任何"提升"必须说明容量**从哪里搬到哪里**，堵住"把容量重新分配谎称为总量增加"的通道。
- **失效条件**：与 D-G2 同（i.i.d. 输入、正交基）；且 #15 表明二维分解也不自动映射到任务性能。
- **训练前可冻结**：✅ 对固定核训练前可算。

---

## §3 对 NeuroTopo 的可转移性逐条判定

判定基准：`G1-MATCH-v3` 单轴匹配（声明并只匹配机制消耗的**一个**轴）／oracle-relative KL 终点／CPU 小模型／已知生成模型／**不训练内部核**的协议。

| 装置 | 判定 | 与 NeuroTopo 约束的兼容性说明 |
|---|---|---|
| D-G1 谱→容量解析 | **需改造** | 闭式仅线性；不可外推到非线性核。可转移的是**形式**：把功能量写成连接矩阵谱量的显式函数并在训练前冻结。与 oracle-relative KL 的接口需改写为"用谱预测量去预测 KL 的**排序**"，不预测 KL 绝对值。 |
| D-G2 IPC 守恒+分解 | **可转移**（带约束） | 与"不训练内部核"**完全兼容**（固定核 + 闭式读出）。CPU 小模型够用（正文用 N=50）。与"已知生成模型"兼容：i.i.d. 输入由生成模型给出。**约束**：因 #15，IPC 不得充当 KL 终点的代理，只能作机制通道分解的预注册中间量。 |
| D-G3 边界带形状预测 | **需改造** | 原论文目标是"调增益到最优"，不是"拓扑可归因收益"。但其**预测形态**（峰值位置 + 两侧不同指数）可直接借为预注册的形状约束。#6 要求把"edge of chaos 必最优"降格为待检假设。 |
| D-G4 三体制+训练前指标 | **可转移** | 指标在训练前由被驱动前向仿真算出，**不消耗训练预算**，天然契合 `G1-MATCH-v3` 的"executed compute 匹配"。可直接改造为"用训练前动力学量定位耦合强度最优值"。 |
| D-G5 结构保持零模型 | **可转移（最高优先级）** | 直接对治本项目的任务选取混杂与机制过度归因。零模型匹配图统计量，与单轴匹配**并列声明**不冲突（零模型不改机制消耗轴）。作者训练的是每输出模块的 ridge readout、不改内部核 → 与"不训练内部核"协议一致。CPU 足够。**须照抄其纪律**：更弱零模型打平时主动放弃更强主张。 |
| D-G6 拓扑×非线性交互 | **可转移** | 把主效应改写为双侧符号交互，是抗混杂的标准形式。终点仍是 oracle-relative KL，只把主效应项换成交互项。**风险**：#3 用 threshold-like 单元 + step 读出，若 NeuroTopo 用 tanh 则须重新标定符号；#3 与 #4 在 tanh 下的张力需自行判定。 |
| D-G7 解析定位工作点 | **需改造** | 均值场式仅对 LIF/脉冲单元直接可用。价值在于**消除调参预算通道**——若 NeuroTopo 的候选机制是脉冲型，可直接迁移；否则须另找解析或半解析的工作点规则。 |
| D-G8 memory–nonlinearity 二维 | **可转移** | 与 oracle-relative KL 终点兼容：KL 仍是主终点，另加预注册的容量分解强制记账。与"不训练内部核"兼容。 |
| D-G7 配套：SW + 输入输出分离（#12） | **需改造** | #12 的收益条件是"谱半径大 **且** 输入/输出节点空间分离"——这构成**第二个轴**。若采用，必须**先固定输入输出放置**，否则违反 `G1-MATCH-v3` 的单轴声明。我仅核到摘要，未核其匹配细节。 |
| 物理 reservoir 整类（自旋冰、胶体、光散射、菌丝、蛋白质等） | **不可转移** | "拓扑"不可编程，且无已知生成模型 → 无法构造 oracle-relative KL 终点。仅可作为"容量度量可在物理系统上实施"的存在性证据。 |
| 量子 reservoir（#18、#19） | **不可转移** | 单西演化与 echo state property 存在结构性冲突（#18 标题即此张力）；无 CPU 小模型 oracle 对照。 |

**一条跨装置的组合建议（形式层面，非新颖性主张）**：D-G4 的"训练前指标定位连续参数" + D-G5 的"结构保持零模型系综" + D-G6 的"双侧符号交互"三者可以叠成同一份预注册：主效应为交互项、定位量为训练前冻结、对照为分层零模型。三者各自的失效条件互不覆盖，因此组合也**不自动**产生稳健性。

---

## §4 我没能核验的限制

1. **零复现**。未运行任何作者代码，未固定 commit，未检查 license。已知 #4 的代码与数据在 `https://github.com/netneurolab/milisav_hierarchical_modularity`，但我**只见到该 URL，未取回仓库内容**。
2. **#8（Ganguli 等 2008）只到摘要层**。该文"记忆容量作为连接矩阵特征值谱的泛函"这一核心结论，我**没有取回正文公式**；本文件因此**不写其公式**，D-G1 的公式全部来自 #2 正文。
3. **#9/#10/#11/#12/#13/#14/#15/#16/#17/#18/#19/#20 均只到摘要层**。其统计方法、匹配细节、效应量我未核验。#15 的"总 IPC 与任务特定性能相关性差"只有摘要措辞，**未见其基准集合与相关系数**。
4. **#3 与 #4 的张力未解决**。#3 报告 tanh 单元下模块化降低 MC，#4 报告 tanh reservoir 中层级模块优于严格模块。两者任务、读出与拓扑生成器不同，我**未做归一化对照**，不得据此判定任一方错误。
5. **通道偏倚**。OpenAlex 全天不可用；通用网页检索被反爬；Europe PMC 对 CS 会议（ICLR/NeurIPS/ICML）覆盖弱。因此"耦合强度临界现象"与"inter-module coupling → capacity"这两条起始查询词，我**只取到相邻证据**（如 #17、#20），**未取到以耦合强度为显式自变量、给出带符号容量预测的专文**。这是覆盖缺口，**不是该工作不存在的证据**。
6. **未检索中文/非英文文献**；未检索学位论文与预印本以外的灰色文献。
7. **"训练前可冻结"是我的判定**，基于各文所述方法所需输入（谱量、均值场方程、teacher-forced 前向仿真），**不是作者的原话**，也未在本地实现验证过任何一条的可冻结性。
