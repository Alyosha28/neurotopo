# DIR-0003 / 领域 B：可证明的结构收益 —— 结构何时改变"可学习性"

**审计日期**：2026-09-24（UTC）。
**审计类型**：文献方法审计（提取**实验设计/识别装置**），不写综述、不做新颖性判断。
**证据等级声明**：本次**无本地复现**。没有训练任何模型，没有运行任何被引论文的代码，没有把外部代码带入仓库。
所有条目标注核验层次：`pdf_section`（本轮真的下载 PDF 并 `pdftotext` 读取正文段落）／`abstract`（本轮真取回摘要文本）／`metadata`（本轮只取回标题+编号+日期）。**未取回的一律不写。**

## 检索边界（本轮实际执行的查询词；引擎与计数见 `dir0003_B_search_log.json`）

| 引擎 | 可用性 | 实际执行 |
|---|---|---|
| OpenAlex API | 首次查询成功后**全 IP 共享免费额度耗尽**，后续全部 `Rate limit exceeded` | 1 条查询成功（`information exponent gradient descent`，返回 5 条／命中 89 条）；其余作废 |
| arXiv API | 大量 `HTTP 429` | 4 条成功（`all:"information exponent"`→12；`all:"leap complexity"`→2；`all:"merged staircase"`→4；`all:"staircase" AND all:"sparse functions"`→1），其余 13 条 429 作废，批次 2 剩余 18 条主动中止 |
| arXiv HTML 检索 | 可用（限流较松） | 10 条查询，n=1/1/2/12/16/19/1/0/0/3 |
| Crossref | 可用 | 5 条查询，各返回 5 条 |
| DuckDuckGo HTML / Mojeek / SearxNG / DBLP | 反爬拦截 | 0 |
| EuropePMC / OpenAIRE | 本轮未使用（领域 B 为 ML 理论，非生物医学） | 0 |

查询词清单（53 条，逐条引擎与计数见 JSON）：`information exponent gradient descent`；`leap complexity saddle-to-saddle`；`merged staircase property sparse functions two-layer`；`staircase property hierarchical target learnability`；`benefits of depth hierarchical functions learnability`；`sparse parity statistical query lower bound`；`sample complexity equivariant neural network generalization`；`multi-index model gradient descent sample complexity`；`hierarchical generative model learnability`；`teacher-student sample complexity exponent`；`hidden progress SGD learns parities computational limit`；`learning polynomials few relevant dimensions`；`low-degree polynomial target learnability`；`finite-size scaling collapse learning curve exponent`；`lazy training kernel regime versus feature learning separation`；`statistical query versus gradient descent separation`；`invariance sample complexity benefit provable equivariance`；`k-sparse parity matching statistical query lower bound sign SGD`；`PAC-Bayesian generalization bound equivariant networks`；`learning rate phase transition generative exponent`；`provably strict benefit of using the right equivariance`；`sample complexity benefits invariant equivariant neural network architectures`；`merging staircase / staircase + sparse functions`；`multi-index + gradient descent`；`hierarchical + depth separation`；`teacher-student + gradient descent + sample complexity`；`benefits of depth`；`lazy training + feature learning`；`low-degree + polynomial + learnability`；`scaling collapse + neural network`；`sample complexity + exponent + feature learning`；`statistical query + gradient descent + separation`；`intrinsic dimension + sample complexity + gradient descent`。

## §1 来源表

| # | 逐字标题 | 年 | 场所 | URL | 核验层次 |
|---|---|---|---|---|---|
| S1 | The Benefits of Reusing Batches for Gradient Descent in Two-Layer Networks: Breaking the Curse of Information and Leap Exponents | 2024 | arXiv:2402.03220 | https://arxiv.org/abs/2402.03220 | pdf_section |
| S2 | SGD learning on neural networks: leap complexity and saddle-to-saddle dynamics | 2023 | arXiv:2302.11055 | https://arxiv.org/abs/2302.11055 | pdf_section |
| S3 | Neural network learns low-dimensional polynomials with SGD near the information-theoretic limit | 2024 | arXiv:2406.01581 | https://arxiv.org/abs/2406.01581 | pdf_section |
| S4 | Online Learning and Information Exponents: On The Importance of Batch size, and Time / Complexity Tradeoffs | 2024 | arXiv:2406.02157 | https://arxiv.org/abs/2406.02157 | pdf_section |
| S5 | Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis | 2024 (v2 2025) | arXiv:2410.09678 | https://arxiv.org/abs/2410.09678 | pdf_section |
| S6 | Smoothing the Landscape Boosts the Signal for SGD: Optimal Sample Complexity for Learning Single Index Models | 2023 | arXiv:2305.10633 | https://arxiv.org/abs/2305.10633 | pdf_section |
| S7 | Online stochastic gradient descent on non-convex losses from high-dimensional inference | 2020 (v4 2021) | arXiv:2003.10409 | https://arxiv.org/abs/2003.10409 | pdf_section |
| S8 | The merged-staircase property: a necessary and nearly sufficient condition for SGD learning of sparse functions on two-layer neural networks | 2022 (v2 2024) | arXiv:2202.08658 | https://arxiv.org/abs/2202.08658 | pdf_section |
| S9 | Matching the Statistical Query Lower Bound for k-Sparse Parity Problems with Sign Stochastic Gradient Descent | 2024 | arXiv:2404.12376 | https://arxiv.org/abs/2404.12376 | pdf_section |
| S10 | A PAC-Bayesian Generalization Bound for Equivariant Networks | 2022 | arXiv:2210.13150 | https://arxiv.org/abs/2210.13150 | pdf_section |
| S11 | Provable Guarantees for Nonlinear Feature Learning in Three-Layer Neural Networks | 2023 (v3 2025) | arXiv:2305.06986 | https://arxiv.org/abs/2305.06986 | pdf_section |
| S12 | Emergence and scaling laws in SGD learning of shallow neural networks | 2025 | arXiv:2504.19983 | https://arxiv.org/abs/2504.19983 | pdf_section |
| S13 | Gradient-Based Feature Learning under Structured Data | 2023 | arXiv:2309.03843 | https://arxiv.org/abs/2309.03843 | pdf_section |
| S14 | From Information to Generative Exponent: Learning Rate Induces Phase Transitions in SGD | 2025 | arXiv:2510.21020 | https://arxiv.org/abs/2510.21020 | pdf_section |
| S15 | Iteratively reweighted kernel machines efficiently learn sparse functions | 2025 | arXiv:2505.08277 | https://arxiv.org/abs/2505.08277 | abstract |
| S16 | Feature Learning in Linear-Width Two-Layer Networks: Two vs. One Step of Gradient Descent | 2026 | arXiv:2605.17767 | https://arxiv.org/abs/2605.17767 | abstract |
| S17 | Symmetric Single Index Learning | 2023 | arXiv:2310.02117 | https://arxiv.org/abs/2310.02117 | abstract |
| S18 | Limit Theorems for Stochastic Gradient Descent in High-Dimensional Single-Layer Networks | 2025 | arXiv:2511.02258 | https://arxiv.org/abs/2511.02258 | abstract |
| S19 | The Uncanny Similarity of Recurrence and Depth | 2021 | arXiv:2102.11011 | https://arxiv.org/abs/2102.11011 | abstract |
| S20 | Matching the Statistical Query Lower Bound for $k$-Sparse Parity Problems with Sign Stochastic Gradient Descent | 2024 | NeurIPS 37（Crossref 确认场所） | https://doi.org/10.52202/079017-3591 | metadata |
| S21 | On the non-universality of deep learning: quantifying the cost of symmetry | 2022 | arXiv:2208.03113 | https://arxiv.org/abs/2208.03113 | pdf_section |

共 21 条记录 / **20 个不同来源**（S20 是 S9 的会议版场所确认）。S1–S14、S21 为 `pdf_section`。

## §2 方法设计装置（核心）

| # | 装置 | 原论文解决的识别问题 | 具体做法 | 需要什么资源 | 为什么比"两臂比一个标量差"更抗混杂 | 失效条件 |
|---|---|---|---|---|---|---|
| B1 | **双侧指数夹逼**（上界算法类 ⟷ 下界查询类） | "结构有收益"与"我的优化器/容量恰好赢了"不可分 | 把预测写成维数指数 `n, T ≍ d^θ`，θ 由目标的结构不变量决定；同时给出**同一算法类**的匹配下界 | 可控维数 d 的目标族 + 每个 d 多次重复；理论侧需要目标的正交多项式展开 | 水平差可被常数因子、超参、种子吸收；log-log 斜率不能。两侧夹逼使"更好"必须同时通过"不许超过下界" | 只差 polylog 的 regime（S3：`n ≃ T = Θ(d·polylog d)`）会伪装成同指数；下界只对该查询类成立 |
| B2 | **算法类阶梯**（同目标、换算法类、指数换挡） | 收益可能来自算法/协议而非结构 | 固定目标族，只换算法类：kernel/线性 → one-pass online SGD → smoothed SGD → full SQ → 重用 minibatch 的 SGD；**预测每一个 rung 的指数表** | 同一任务的多条曲线 × 多个 d | 把"结构 vs 算法"的混淆变成**换挡预测**：收益若来自结构，换算法类应改变指数而非平移水平 | CSQ 下界**不约束** ERM（S3 自陈 CSQ 模型不含 ERM）；S15 反例：迭代重加权**核**也能拿到 leap 收益，故"神经专属"不成立 |
| B3 | **非单调临界线** | "更多样本/算力更好"是单调、不可证伪的混淆 | 预测**最优**批量作为幂律 `n_b ≲ d^{ℓ/2}`，超过 `n_b ≫ d^{ℓ/2}` 反而损害时间复杂度的改进；另一最优点在 `n_b = O(d^{ℓ-1})`（correlation loss） | 批量 × 维数二维扫描；S14 显示学率是第三个轴 | 单调效应永远"可调大"，**非单调 + 带临界指数**的预测一旦方向错了就是反例 | S4 自陈临界线 `n_b = Ω(d^{ℓ-1})` **未被其形式化结果覆盖**（"is not addressed by our formal"）；相图依赖 ℓ≥3 |
| B4 | **训练曲线形状：saddle-to-saddle 平台长度分级** | 端点标量差掩盖"究竟学到了什么" | 预测 loss 是**阶梯**：第 k 个平台长度 `Θ̃(d^{max(L_k−1,1)})`，`L_k` 是该鞍点的 leap；总时间 `Θ̃(d^{max(Leap,2)})`，并对 kernel/CSQ 给出 `c d^{Leap}` 下界 | 需要逐 checkpoint 的 loss/重叠曲线 + 多个 d | 平台**个数**与**每段对 d 的斜率**是多点结构，伪造水平差容易、同时伪造平台数与各段斜率难 | S2 自陈只在 leap **递增**的构造下证明；leap 递减时 saddle-to-saddle 是否成立**理论未决** |
| B5 | **谓词式判定**（merged-staircase，必要且几乎充分） | 连续可调的"结构量"会退化成自由参数 | 在**固定 `O(d)` 样本档**上，SGD 可学性由非零 Fourier 支撑集是否满足 merged-staircase 性质**充要（几乎）**决定；同时证明**任何**线性方法（含 NTK）不能 | 布尔输入 + 低维 latent + mean-field 两层网络 | 谓词把预测从"更好"变成"能不能"，排除水平差解释；并把"稠密可模拟"内置为被证明不行的对照类 | "几乎"= 仅对 generic 系数成立，测度零退化集除外；只覆盖 mean-field 两层；对一般正则网络未证 |
| B6 | **把"线性/NTK 下界"作为独立算法类必备对照** | 最强拒绝：稠密单流模型可以模拟分组组织 | 用 `c d^D`（D=度数）证明线性方法需要**指数级于度数**的样本，而 SGD 只需 `d^{Leap}`，`Leap ≪ D`；下界对**整个类**成立 | 理论侧；实验侧需要 kernel/线性读出基线 | 下界不是针对某个基线，而是针对**一类算法**，因此给基线加算力/调参不能推翻它 | 下界必须**附带正则性假设**（等向初始化、mean-field 等）；S2 明确：无约束架构可 emulate 任意高效 PAC/SQ 算法（[AS20]），故下界对无正则约束网络**失效** |
| B7 | **"同一样本用几次"作为离散协议旋钮** | "额外状态/计算带来收益"需要被指认到具体消耗 | 同一 mini-batch 上做**两次** GD：`w₂ = ησ'(0)·y·x + η·yσ'(ησ'(0)‖x‖·y)x`，第二项含 `y` 的**二次**信息 → 超出 correlational query，允许标签变换降低有效指数 | 极低：一个整数旋钮 + 学率范围 | 把"多用了什么"精确指认到**离散量**（1 次 vs 2 次），而不是"计算更多"这类连续资源 | S14：该机制**只在学率足够大时**成立，存在"information exponent regime ↔ generative exponent regime"的**相变**；不报学率相图则该旋钮自身变成混杂 |
| B8 | **不变性盲区的零结果预测** | 测一个在结构上**不可识别**的量会得到假零效应 | S5：`L=2` 时二阶项**旋转不变** → 只能恢复**子空间**、不能恢复**方向**；只用高阶项恢复方向需 `Õ(P d^{L−1})`。作者的做法是**同时**用二阶项（先学子空间）与高阶项（再学精确方向），整体复杂度降为 `Õ(d P^{L−1})`（省掉一个 `P` 因子） | 需要**子空间恢复**与**方向恢复**两个分开的终点 + 多档 `P`、`d` | 它预测"在什么结构下方向不可识别"，是**零结果预测**；且把"子空间 vs 方向"两层分开测，避免了把不可识别量当成效应量 | 需要 `log^C d ≤ P ≤ d` 与 orthonormal 真方向、i.i.d. Gaussian；`L>2` 才有方向可恢复 |
| B9 | **在所有标签重参数化下不变的难度不变量（生成指数）** | 观测难度可能只是链接函数**表示选择**的产物 | 定义 `p* = inf_{T∈L²} IE(T∘g)`；上界随 `p*` 而非 `p` 标度；给出 `p* < p` 的显式例（`IE(He_k)=k` 而 `GE(He_k) ≤ 2`） | 需要可控的标签变换类；理论侧 | `p*` 在重参数化下不变 → 把"优化难度来自表示选择"这一通道**显式参数化**并可分离检验 | 需要**重用样本**的 SGD 才能真正实现该上界；对非多项式 σ，`p*` 与算法可达性的关系未完全解决（S3） |
| B10 | **优化型深度分离**（不是近似型） | 深度/层级的好处在近似意义上会被"稠密宽模型也能逼近"击穿 | 构造一个函数：**在梯度下降下**三层可高效学、两层不可；分离依赖三层能学**非线性特征** | 需要显式构造 + 两层/三层同协议对照 | 近似型分离只谈表达能力；优化型分离把结论写在**算法轨迹**上 | 分离依赖具体构造；S11 的 v3 revision 明说把 Theorem 1 的界从 `‖K_{f*}‖^{-2}` 加强为 `‖K‖_op‖K_{f*}‖^{-2}`，说明常数/宽度依赖仍在收紧 |
| B11 | **数据协方差结构作为"隐式引导"对照臂** | 数据结构本身会引导算法，被误记为架构好处 | S13：spiked covariance 下球面梯度动力学**即使 spike 与目标完全对齐也可能失败**；类 BN 的权重归一化可修复；spike 足够大时样本复杂度**与信息指数无关** | 可控的各向异性输入 + 归一化开关 | 同时给出同架构的**失败**情形与**修复**条件，即一个负向预测 | 依赖 spike 模型与对齐假设；一般各向异性数据未证 |
| B12 | **幂律缩放指数 + 多时间尺度突变叠加** | 端点水平差无法区分"常规参数/样本收益"与"结构收益" | S12：给出 MSE 对样本数、SGD 步数、参数数的**缩放指数**；单个 teacher 神经元的学习是**突变**的，`P ≫ 1` 个不同时间尺度的突变叠加产生平滑幂律 | extensive-width 条件 + 条件数分布可控 | 幂律指数是形状量；且"平滑曲线 = 多个突变叠加"本身是可证伪的机制解释 | 依赖 extensive-width 与正交 teacher 方向；`P=O(1)` 不适用 |
| B13 | **学率相变计数** | 超参而非机制在决定标度 | S14：学率 `η` 诱导"information exponent regime ↔ generative exponent regime"相变；例：`σ*=He₃` 时临界点 `η ≍ d^{−1/2}`；**相变个数 = 不同有效指数个数 − 1** | 学率 × 维数二维扫描 | 给出**临界点位置**与**相变个数**两个可数、可定位的预测，水平差无法伪造 | 相变位置依赖具体 `μᵢ` 谱与假设（S14 自陈假设"difficult to verify in general"） |
| B14 | **算法等变性作为上界条件**（对称性的代价） | "结构有收益"的反面：架构的对称性会**限制**GD 能学什么 | S21：只要 GD 训练是 `G`-等变的（正交初始化、循环卷积、无位置编码的 Transformer 等均满足），就存在**不能被高效学**的分布；并据此把 merged-staircase 的**必要性**从 mean-field 推广到 mean-field 之外；密码学假设下给出 SGD 硬度 | 理论侧；实验侧需要一个"打破等变性"的开关（非等变初始化/位置编码/非循环卷积） | 它把"结构"写成**算法不变群**这一可检验对象，并预测：打破该群应改变可学性，而水平差不会随不变群改变 | 结论形式是"存在不可学的分布"，不是对所有分布；密码学假设部分依赖标准假设 |

## §3 对 NeuroTopo 的可转移性逐条判定

前提：现行匹配判据 `G1-MATCH-v3`（**单轴匹配**、状态不要求匹配、其余轴作为实测比值披露）；终点含 **oracle-relative KL**；算力约束 **CPU 小模型**；任务侧原则上**已知生成模型**。

| 装置 | 判定 | 与 G1-MATCH-v3 单轴匹配的兼容性 | 与 oracle-relative KL 终点的兼容性 | CPU 小模型可行性 | 与"已知生成模型"的兼容性 |
|---|---|---|---|---|---|
| B1 双侧指数夹逼 | **需改造**（指数对象要换刻度） | 兼容：指数比较发生在**已匹配的那一轴**上，斜率不引入新的资源轴 | 需改造：终点仍可用 KL，但**读出量**改为"达到 KL 阈值所需的样本/更新数"，其幂律指数才是主终点 | **受限**：幂律拟合需跨数量级，CPU 小模型的 d 或 K 范围可能不足 | 兼容：已知生成器可直接扫结构刻度 |
| B2 算法类阶梯 | **可转移**（推荐） | 兼容：各 rung 只需声明各自消耗的那一轴 | 兼容：同一 KL 阈值下的指数表 | 可行：rung 数 = 臂数，臂数与 d 档数相乘，需控制预算 | 高度兼容：生成器固定，只换学习算法 |
| B3 非单调临界线 | **可转移** | 兼容：批量/重用次数是**协议旋钮**，不是资源轴，无需匹配 | 兼容 | 可行：3–4 档即可观察非单调 | 兼容 |
| B4 平台长度分级 | **需改造 → 降级为定性排序预测** | 兼容 | 需改造：读出为逐 checkpoint 的 KL 轨迹，非终点标量 | 可行（KL 轨迹比 d 扫描便宜）：改为"平台**个数**与**坐标被拾起次序**"预测 | 兼容（已知生成器可定义坐标/因子） |
| B5 谓词式判定 | **需改造**（本领域最高价值项） | 兼容：谓词是**任务属性**，不是资源轴 | 兼容 | 可行：布尔谓词两侧各构若干实例 | 高度兼容：生成器可直接构造"满足/不满足谓词"的实例对，并做**谓词 × 模型交互检验** |
| B6 线性/NTK 下界作对照 | **可转移为对照臂**（非 Gate 1 臂） | 部分：核基线消耗的轴与本项目机制不同，须按 `PACKAGE_EFFECT` 处理并挂资源未匹配标记 | 兼容 | 可行（核回归在 CPU 上便宜） | 兼容 |
| B7 一条样本多次更新 | **可转移**（成本最低） | 兼容：纯协议旋钮 | 兼容 | 极易 | 兼容 |
| B8 不变性盲区零结果 | **需改造**：并入表示探测 | 兼容 | 兼容：前提是终点只建在**不变量意义下可定义**的目标上 | 可行 | 兼容 |
| B9 生成型不变量 | **需改造** | 兼容 | 兼容：oracle 已知 ⇒ 该不变量可计算 | 可行（是分析量，不是训练量） | 高度兼容 |
| B10 优化型深度分离 | **可转移为设计模板** | 兼容：分离是对**同协议**的两臂 | 兼容 | 可行（2 层 vs 3 层小模型） | 兼容 |
| B11 协方差/结构对照 | **可转移为负向预测臂** | 兼容 | 兼容 | 可行 | 高度兼容（生成器可控协方差） |
| B12 幂律缩放律 | **需改造 → 仅作辅助诊断** | 兼容 | **风险**：缩放指数本质上是"学习效率"终点，与决定0062 B（学习效率不得作主终点，未决）冲突；须按 B1–B7 处理或**不进入 Gate 1 判词** | 可行但需大量曲线 | 兼容 |
| B13 学率相变计数 | **需改造 → 作为混杂排除步骤** | 兼容：学率是超参，不是资源轴 | 兼容 | 可行 | 兼容 |
| B14 算法等变性作为上界条件 | **需改造**：需要"打破等变性"的架构开关 | 兼容：不变群是**架构/初始化属性**，不是资源轴；但打破等变性可能改变有效参数量，须在 manifest 披露 | 兼容 | 可行（非等变初始化在 CPU 上零成本） | 兼容 |

**三条跨装置结论。**

1. **"稠密可模拟"在理论里有一个正式对应物，而且它恰好界定了收益可证的边界。** S2 原文把无约束架构（可 emulate 任意高效 PAC/SQ 算法）排除在结论之外，只对满足"正则性假设"（等向初始化等）的网络给下界；S21 把这个条件精确化为**算法的等变群**：只要 GD 训练是 `G`-等变的，就存在学不动的分布，而等变群随架构/初始化而变。对本项目的直接含义：任何"结构收益"主张若不写明**算法类的正则性/等变假设**，在理论上就不可证伪——这正是决定0043 的正式形态，而 B6 与 B14 是把它转成可执行对照的两条路。
2. **抗混杂的形态是"形状预测"，不是"水平预测"。** 本领域可证伪的预测一共只有五种形态：(i) 维数指数 `d^θ`；(ii) 非单调临界线；(iii) 平台/相变的**个数与位置**；(iv) 布尔谓词分侧；(v) 不变群下的存在性零结果。没有一条是"臂 A 的标量 > 臂 B 的标量"。这与项目"识别性问题而非效应大小问题"的判断一致。
3. **有两个必须先处理的攻击面。** (i) S15（`abstract` 层）显示**迭代重加权核**也能高效学习有限 leap 的层级多项式——"结构收益"不自动是神经架构专属，候选机制必须把迭代重加权核列为算法类基线。(ii) S21 指出架构等变与**算法**等变是两件不同的事（全连接网络不正交不变，但其 SGD 训练在正交初始化下是正交等变的）；把架构对称性当成收益来源时，必须分开这两个概念。

## §4 我没能核验的限制

1. **无本地复现，无源码审计。** 本文件没有任何 `code_commit` 层证据。所有机制陈述均为"作者在论文正文中如此写"，非本轮验证。
2. **S15–S19 只到 `abstract` 层**（S20 只到 `metadata`）。它们只用于指出攻击面与相邻方向，**不承担**任何装置的主要证据。
3. **理论结论的适用范围未独立检查。** 各定理的假设（mean-field、extensive-width、orthonormal 真方向、i.i.d. Gaussian、低维 latent `P=O_d(1)`）是本轮直接从正文段落读到的；未逐一回归附录证明。
4. **两条自陈的开放点被原样保留、未解决**：S4 的临界线 `n_b = Ω(d^{ℓ−1})` 未被形式化覆盖；S2 的 saddle-to-saddle 仅在 leap 递增时证明。
5. **下界类不覆盖 ERM**（S3 自陈），因此"匹配对照会混淆"这一处境在理论上**没有被现有下界工具完全解决**。
6. **检索通道受限导致覆盖不完整**：OpenAlex 额度耗尽、arXiv API 大量 429、DDG/Mojeek/SearxNG/DBLP 被反爬。等向/等变方向的检索深度明显弱于信息指数/leap 方向（B10、B11、S13 各只有 1 条）。**检索未命中不等于无人做过**，本文件不产出任何新颖性结论。
7. **未核验的相邻方向**（本轮只看到标题、未取回摘要正文者一律不列入 §1，此处仅记账）：`benefits of depth` 检索返回的多数结果与"层级函数可学习性"无关（图像/池化/机器人），说明该查询词未命中目标族。**等变方向的样本复杂度理论是本次最弱的一环**：S10 只给 PAC-Bayes 界的结构（组大小、不可约表示的重复度与次数如何进入界），S21 给的是**等变性作为限制**而非收益；正文中出现的"不变/等变架构有样本复杂度收益"一系工作（S21 的参考文献 [MMM21, EZ21, Ele21, BVB21, Ele22]）**本轮未取回原文**，因此本文件**不列其标题与标识符**，也不据此作任何判断。该线需在检索通道恢复后补核。
8. **未做的事**：没有读任何一篇的附录证明细节到"可独立复核定理成立"的程度；没有检查任何一篇的代码；没有做任何跨论文的定量复算（例如把不同论文的 `d^θ` 放在同一刻度上比较，因各文对 `≍/≲`、polylog 与常数依赖的约定不同，本轮不做此类比较）。
