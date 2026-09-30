# dir0003_C：优化误差与近似误差的实测分离（方法学设计装置审计）

**审计日期**：2026-09-24（UTC）。**领域**：C_optimization_vs_capacity。
**证据等级声明**：**无本地复现**。本文件不含任何本仓库运行结果；全部内容为对外部文献的检索与阅读记录。
核验层次只有三种：`abstract`（取回 arXiv abs 页并读到标题+摘要原文）、`pdf_section`（下载 PDF 并 `pdftotext` 后读到该段落原文）、`code_commit`（本文件**没有**任何条目达到此层）。
**检索边界（实际执行的查询词，逐字）**：

- HF papers 检索（`huggingface.co/api/papers/search`）：`lazy training feature learning`；`deep bootstrap framework online learners offline generalizers`；`linear mode connectivity lottery ticket hypothesis`；`loss surfaces mode connectivity fast ensembling`；`essentially no barriers in neural network energy landscape`；`progress measures grokking mechanistic interpretability`；`deep bootstrap good online learners good offline generalizers`；`do deep nets really need to be deep`；`the early phase of neural network training break even point`；`saddle to saddle dynamics deep linear networks`；`deep learning requires rethinking generalization random labels`；`edge of stability gradient descent`；`the deep bootstrap framework`；`knowledge distillation capacity gap teacher student`；`born again neural networks self distillation`；`sharpness aware minimization critique measures of sharpness`；`critical learning periods in deep networks`；`gradient starvation neural network capacity bottleneck`；`deep bootstrap generalization infinite data oracle`；`model stitching representation equivalence`；`similarity of neural network representations kernel alignment`；`training time to reach target accuracy threshold scaling`；`permutation symmetry linear mode connectivity failure`
- arXiv API：`all:"lazy training" AND all:"neural tangent kernel"`；`all:"rich regime" AND all:"feature learning"`；`ti:"Deep Bootstrap"`
- EuropePMC：`TITLE:"approximation error" AND TITLE:"optimization error"`；`"optimization error" AND "approximation error" AND "neural network"`；`"grokking" AND "generalization"`
- Crossref：`Deep Bootstrap Framework Good Online Learners are Good Offline Generalizers`
- **已弃用通道**：OpenAlex（首查返回 38 hits 后全 IP 429，按指示停止使用）；DuckDuckGo HTML（反爬）；DBLP（bot 墙）。

**范围声明**：检索未命中 ≠ 无人做过。本文件**不产出任何新颖性结论**。

---

## §1 来源表

| # | 逐字标题 | 年 | 场所 | URL / ID | 核验层次 |
|---|---|---|---|---|---|
| S1 | Disentangling feature and lazy training in deep neural networks | 2020（v1 2019） | J. Stat. Mech.（DOI 取自 abs 页） | https://arxiv.org/abs/1906.08034 · DOI 10.1088/1742-5468/abc4de | **pdf_section** |
| S2 | Grokking as the Transition from Lazy to Rich Training Dynamics | 2023 | ICLR 2024（PDF 页眉逐字） | https://arxiv.org/abs/2310.06110 | **pdf_section** |
| S3 | Linear Mode Connectivity and the Lottery Ticket Hypothesis | 2020 | ICML 2020（PDF 页眉 "PMLR 119, 2020"） | https://arxiv.org/abs/1912.05671 | **pdf_section** |
| S4 | Progress measures for grokking via mechanistic interpretability | 2023 | ICLR 2023（PDF 页眉逐字） | https://arxiv.org/abs/2301.05217 | **pdf_section** |
| S5 | The Deep Bootstrap Framework: Good Online Learners are Good Offline Generalizers | 2020 | 未核验 | https://arxiv.org/abs/2010.08127 | abstract |
| S6 | Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs | 2018 | 未核验 | https://arxiv.org/abs/1802.10026 | abstract |
| S7 | Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets | 2022 | 未核验 | https://arxiv.org/abs/2201.02177 | abstract |
| S8 | Towards Understanding Grokking: An Effective Theory of Representation Learning | 2022 | 未核验 | https://arxiv.org/abs/2205.10343 | abstract |
| S9 | Understanding deep learning requires rethinking generalization | 2016 | 未核验 | https://arxiv.org/abs/1611.03530 | abstract |
| S10 | Does Knowledge Distillation Really Work? | 2021 | 未核验 | https://arxiv.org/abs/2106.05945 | abstract |
| S11 | On Warm-Starting Neural Network Training | 2020 | NeurIPS 2020（abs 页 JREF 逐字） | https://arxiv.org/abs/1910.08475 | abstract |
| S12 | Git Re-Basin: Merging Models modulo Permutation Symmetries | 2022 | 未核验 | https://arxiv.org/abs/2209.04836 | abstract |
| S13 | Landscaping Linear Mode Connectivity | 2024 | 未核验 | https://arxiv.org/abs/2406.16300 | abstract |
| S14 | Similarity of Neural Network Representations Revisited | 2019 | 未核验 | https://arxiv.org/abs/1905.00414 | abstract |
| S15 | A Modern Look at the Relationship between Sharpness and Generalization | 2023 | 未核验 | https://arxiv.org/abs/2302.07011 | abstract |
| S16 | Critical Learning Periods Emerge Even in Deep Linear Networks | 2023 | 未核验 | https://arxiv.org/abs/2308.12221 | abstract |
| S17 | Gradient Starvation: A Learning Proclivity in Neural Networks | 2020 | 未核验 | https://arxiv.org/abs/2011.09468 | abstract |

---

## §2 方法设计装置（核心）

### D1｜把"优化难度"做成连续可调的一维旋钮：懒惰↔丰富交叉（S1, S2）
- **识别问题**：同一臂学不到解，无法区分"解不在可达函数类内"（近似误差）与"SGD 找不到"（优化误差）。
- **做法**：在**同一臂、同一任务、同一数据**上加一个输出尺度 $\alpha$。S1 令最后一层权重按 $\alpha h^{-1/2}$ 初始化，扫描 $(\alpha,h)$ 平面；S2 直接令 $f=\frac{\alpha}{N}\sum_i\phi(w_i\cdot x)$，$\alpha$ 控制输出幅度。$\alpha$ 大→惰性（NTK 冻结，退化为核回归）；$\alpha$ 小→丰富（特征学习）。观测量不是最终准确率，而是 S1 的**核相对变化** $\lVert\Theta(w)-\Theta(w_0)\rVert/\lVert\Theta(w_0)\rVert$（作者报告曲线在 $\alpha\sqrt h$ 上坍缩，交叉点 $\alpha^*\sim h^{-1/2}$，丰富端指数 $a\approx1.3$）。
- **资源**：$(\alpha,h)$ 网格点数 × 训练运行的算力。S1 用 MNIST/Fashion-MNIST/EMNIST/CIFAR10 的降采样训练集（10k 训练样本，逐字："The train set is smaller than usual (10000 instead of 50000) in order to shorten the training time"）。
- **为何比"两臂比一个标量差"抗混杂**：它把一个**事后辩解**（"差是因为优化难"）变成**自变量的轴**。结论形式是"优势只出现在旋钮的哪一侧"，而不是"臂 A 比臂 B 高 X"。不需要匹配对照臂——比较发生在同一臂内部。
- **失效条件**：$\alpha$ **不是纯净的优化难度旋钮**，它同时改变可达函数类与线性化程度（S2 逐字："the $\alpha\to\infty$ limit is one where equation 1 [线性化]"）；S1 明确交叉点标度依赖宽度 $h$，且非光滑激活（ReLU）存在第三个中间区域。必须在论文自己的坐标系（$\alpha\sqrt h$）内报告，否则把容量通道重新引入。

### D2｜随机标签/加噪拟合探针：同臂的"容量下界证书"（S9）
- **识别问题**：如何在不依赖"训练成功"的前提下证明该臂**能**表示解。
- **做法**：保持架构、优化器、超参不变，只把训练标签随机化（或加噪）后训练。若训练损失仍降到零，则该架构的容量足以拟合任意标签映射。
- **资源**：单次训练，CPU 小模型极廉价。
- **为何抗混杂**：它给出"近似误差≈0"的**实证下界**（在训练集上），于是测试集上的失败被干净地推到优化/泛化一侧，而不是靠两臂性能差来反推。无需匹配臂。
- **失效条件**：只证明**训练集**容量，不证明目标函数在假设空间内；随机标签的最优解与任务解无关，因此不能反过来当作"结构能表达任务解"的证据。

### D3｜Grokking 的"训练损失已达标 vs 测试损失延迟"作为同臂内表示—读出分离（S2, S7, S8）
- **识别问题**：最终测试指标是标量，看不出"解已被拟合但还没被选中"。
- **做法**：在同一臂同一任务上，**训练损失先降到零**（S7 报告的原始现象），测试损失在很久之后才下降。S2 用 $\alpha$ 连续调节这个延迟；S8 给出四相图（comprehension / grokking / memorization / confusion）与 "Goldilocks zone"，并报告在 transformer 上 grokking 相更靠近 memorization 相。
- **资源**：需要足够长的训练预算才能观察到延迟；CPU 小模型 + 合成任务可行。
- **为何抗混杂**：训练损失达标构成"该臂**能**表示一个拟合解"的同臂证明；此后的测试损失差异是**表示选择**问题而非容量问题。全程无额外参数、无第二臂。
- **失效条件**：S2 逐字给出——需要 "a 'goldilocks' train set size so that generalization is possible but not immediate (for instance, grokking is impossible in the infinite data limit where train loss perfectly tracks test loss)"。因此该装置**依赖有限训练集**，与"无限数据 oracle 参照"互斥。

### D4｜机制可解释性派生的进度测量：restricted / excluded loss（S4）
- **识别问题**：标量终点掩盖"计算电路是否已形成"。
- **做法**：先逆向工程出模型实际使用的算法（模加法上的 Fourier 乘法），再对 logits 做 2D DFT。**restricted loss**＝只保留常数项与五个关键频率对应项后的损失；**excluded loss**＝只删掉这些关键频率、保留其余后的损失。训练中逐点测量。
- **资源**：一次训练 + 每次测量的前向传播（投影/消融），CPU 极廉价。
- **为何抗混杂**：作者报告 excluded loss 在 train/test loss 平坦期就已上升、restricted loss 在 test loss 之前下降——给出**连续的中间量**，使"电路已形成但读出未完成"与"电路未形成"可区分，而不是靠终点差推断。
- **失效条件**：**必须先知道电路是什么**（依赖已完成的机制解释）；对未解释的任务无法定义"关键频率"。S2 另报告：在 S4 所用的一层 transformer 上，仅调 $\alpha$ 就足以让 grokking 连续消失——即 D4 观察到的相位结构本身受优化难度旋钮控制。

### D5｜线性模式连通性 / 对 SGD 噪声的稳定性（S3），及置换对称失效（S12, S13）
- **识别问题**：两个配置（或两个种子）结果不同，是表示不同，还是落在不同优化盆地？
- **做法**（S3）：从**同一初始化** $W_0$ 训练两份拷贝，只更换 SGD 噪声（数据顺序、增广）；在线性路径上插值 $W(\alpha)=\alpha W_1+(1-\alpha)W_2$，定义 **error barrier height** $=\sup_\alpha E_\alpha - \bar E$，称为 linear interpolation instability。操作化细节逐字：稳定性阈值 <2%，30 个均匀 $\alpha$，3 个初始化 × 3 次运行取平均。
- **资源**：约 6–9 倍单次训练算力（3 init × 3 run），每次加 30 次插值评估。CPU 小模型可行，大模型成本显著。
- **为何抗混杂**：把**初始化作为共享随机化**固定住，剩余差异只能来自 SGD 噪声。barrier≈0 意味着"结构差异"可被重参数化吸收——这是对"结构带来不同表示"这类结论的直接反证，且**不需要第二个臂**。
- **失效条件**：隐藏单元**置换对称性**会人为制造虚假 barrier；S12 报告"计入所有置换对称后常常是单一盆地"，S13 给出 LMC 出现/不出现的景观拓扑模型（"mountainside and ridge"）。因此 barrier **不是无参数的不变量**，任何 barrier 结论必须先做置换对齐。

### D6｜非线性模式连通路径：barrier 是测量方法的性质（S6）
- **识别问题**：直线插值出现高 barrier，是否等于"两个解不可达/属于不同模式"？
- **做法**：用曲线（端点固定的参数化路径 + 沿路径训练）搜索连接两个独立解的**高精度通路**，并据此构造 Fast Geometric Ensembling。
- **资源**：路径搜索本身是一个额外优化问题，算力可达单次训练的若干倍。
- **为何抗混杂**：它证明"直线 barrier 高"不能直接读作"优化困难"——barrier 依赖于所选的路径族。因此它是**任何用 barrier 论证优化难度的工作的前置反驳条件**。
- **失效条件**：找到一条低 barrier 路径 ≠ SGD 会走这条路径；路径存在性不构成可达性证据。

### D7｜Deep Bootstrap：理想世界（总体损失）/ 现实世界（经验损失）耦合分解（S5）
- **识别问题**：测试误差里优化误差与泛化误差混在一起，无法分离。
- **做法**：把 Real World（在经验损失上做随机梯度步）与 Ideal World（在**总体损失**上做梯度步，等价于无限数据）耦合。作者逐字给出分解："an alternate decomposition of test error into: (1) the Ideal World test error plus (2) the gap between the two worlds"。若间隙小，离线泛化问题归约为在线优化问题。作者的实证结论是 CNN 与 MLP 的差别"because they optimize faster on the population loss in the Ideal World"。
- **资源**：需要**已知生成分布**才能采样无限数据；每次比较要跑两条曲线（现实 + 理想）。
- **为何抗混杂**：给同一臂一个 **oracle 参照**，优化难度由理想世界曲线**直接测量**，而不是由两臂端点差反推。与本项目 oracle-relative KL 终点同构。
- **失效条件**：需要已知生成模型；作者的"optimize faster"是把优化速度当解释变量——**直接采用该表述会落入学习效率终点**（见 §3 冲突项）。

### D8｜oracle 教师蒸馏匹配探针：容量够但匹配不上 ⇒ 优化困难（S10）
- **识别问题**：学生达不到教师，是容量不够还是优化不到？
- **做法**：令 student 匹配 teacher 的**预测分布**（teacher 是已知函数，故可实现性可判定）。作者报告："there often remains a surprisingly large discrepancy between the predictive distributions of the teacher and the student, even in cases when the student has the capacity to perfectly match the teacher"，并把原因归为 "difficulties in optimization"。
- **资源**：一次教师训练 + 一次学生训练；CPU 小模型可行。
- **为何抗混杂**：把"函数匹配"变成可独立检查的目标；容量足够而匹配失败**只能**是优化问题，无需两臂性能差。
- **失效条件**：作者明确报告 "more closely matching the teacher paradoxically does not always lead to better student generalization"——该探针测的是**优化误差**，不能把匹配度当作能力或收益终点。

### D9｜初始化随机化 / 热启动对照（S11）
- **识别问题**：性能差是否来自初始化运气（优化路径）而非结构？
- **做法**：同一优化问题序列上比较 warm start（用前一解初始化）与 fresh random init。作者报告后者泛化更好，尽管**最终训练损失相近**。
- **资源**：多次训练；CPU 廉价。
- **为何抗混杂**：把初始化变成**可随机化的因子**，用多次运行的分布代替单点，避免把一次坏初始化误读为结构缺陷。
- **失效条件**：warm start 改变的不只是计算预算，还有可达解的**隐式偏置**；因此它是"解选择"干预，不是纯粹的优化难度操纵。

### D10｜表示相似度（CKA）：把"表示是否改变"从性能差里分离（S14）
- **识别问题**：两臂性能相同可以是不同表示，性能不同可以是同表示——性能差不指示表示差。
- **做法**：不比较表示本身，而比较**表示相似矩阵**（centered kernel alignment）；作者指出基于 CCA 的统计量在维度大于样本数时失效，而 CKA 不受此限，能可靠识别不同初始化网络间的对应关系。
- **资源**：仅前向传播 + 核矩阵计算，CPU 极廉价。
- **为何抗混杂**：提供性能标量之外的**第二条独立读数**，使"结构改变了内部组织"与"结构只是改变了运气"可区分。
- **失效条件**：高 CKA ≠ 功能等价；本仓库 `AGENTS.md` 明确"不得从状态距离推断假设专门化"，故 CKA 只能作**操纵检查**，不能作收益终点。

### D11｜负面装置：sharpness 作为优化难度代理已被实测证伪（S15）
- **识别问题**：能否用一个损失面标量（尖锐度）代理优化难度/泛化？
- **做法**：在 ImageNet/CIFAR-10 从头训练、CLIP 微调、BERT/MNLI 等设置上系统比较各种 adaptive sharpness 定义。作者报告 sharpness **与泛化不相关**，而与学习率等训练参数相关（相关方向依赖设置），并在多例中与 OOD 误差呈一致**负**相关。
- **为何重要**：这是一条**关闭式证据**——本项目不应把 sharpness/flatness 当作"优化难度"或机制有效性的证据。
- **失效条件**：结论绑定于其研究的规模、架构与定义族，不外推到所有几何量。

### D12｜解析代理 + 特征学习失败的动力学诊断（S16, S17）
- **识别问题**：临界期/特征学习失败是否只是架构或优化器的偶然？
- **做法**：S16 在**深度线性网络**（可解析）中复现临界学习期，隔离架构与优化细节；S17 用动力系统工具形式化"交叉熵被只捕捉部分特征的解最小化"（gradient starvation），证明在特定数据统计结构下可预期，并据此设计解耦正则化。
- **资源**：线性网络/小型 MLP，CPU 极廉价。
- **为何抗混杂**：S16 用**表达能力受限的代理**排除"非线性容量"这条通道，从而把效应归因于学习动力学本身；S17 提供"优化收敛到次优特征子集"的**可测量诊断量**，而不是从损失差反推。
- **失效条件**：线性/低容量代理的结论不自动迁移到非线性与循环结构；S17 的理论条件依赖数据统计结构。

---

## §3 对 NeuroTopo 的可转移性逐条判定

判定基准：`G1-MATCH-v3` 单轴匹配（决定0069）／oracle-relative KL 终点／CPU 小模型／已知生成模型／决定0062 B 禁学习效率主终点。

| 装置 | 判定 | 与现有约束的兼容性 |
|---|---|---|
| D1 α 尺度旋钮 | **可转移** | 与单轴匹配**天然兼容**：全部测量在同一臂内完成，不产生第二个臂，故无轴需要匹配。终点用最终损失或 oracle-relative KL 即可。CPU 小模型足够（S2 是两层 MLP + 多项式回归；S1 用 10k 训练样本）。**冲突项**：不得把"time to grok"当终点。 |
| D2 随机标签拟合探针 | **可转移** | 作为**操纵检查**（"该臂能拟合训练集"）完全兼容，无对照臂、无匹配问题、CPU 极廉价。不得作为收益终点。 |
| D3 grokking 延迟 | **需改造** | 装置自然终点是**延迟/时间**，直接使用即学习效率终点 → **与决定0062 B 冲突**。改造方式：把"训练损失达标"降级为前置条件（容量证书），主终点仍为 oracle-relative KL。另需注意 S2 要求有限训练集，与本项目"已知生成模型可采样无限数据"存在张力。 |
| D4 restricted/excluded loss | **需改造** | 终点是**消融后的损失**（非时间），与 oracle-relative KL 兼容且可构造 KL 版本。CPU 廉价。改造点：需要先定义"关键子空间"——本项目有已知生成模型，可用**真实后验/生成因子的子空间**代替 Fourier 关键频率，但必须事前注册该子空间定义，否则是事后拟合。 |
| D5 LMC / SGD 噪声稳定性 | **可转移** | 与单轴匹配兼容（同臂内共享初始化的随机化，无需第二臂）。直接回应审查拒绝项"性能差可能来自硬编码身份/额外状态"：若 barrier≈0，则臂间差异可被重参数化吸收。算力约 6–9×。**必须先做置换对齐**（S12/S13），否则结论无效。 |
| D6 非线性模式连通路径 | **需改造** | 作为**反驳条件**（要求任何 barrier 论证附带非线性路径结果）成本低、价值高；作为独立装置则算力昂贵且"路径存在≠SGD 可达"。 |
| D7 Deep Bootstrap 理想/现实分解 | **需改造** | 本项目有**已知生成模型**，Ideal World 可构造，与 oracle-relative KL 终点**高度同构**，是最贴合本项目的分解框架。改造点：(i) 理想世界需要在总体损失上优化，对循环/生成任务可能与原问题同难；(ii) **必须只报告间隙，不得报告"优化更快"**——后者是学习效率表述，与决定0062 B 冲突。 |
| D8 oracle 教师蒸馏探针 | **需改造** | 与 oracle-relative KL 终点天然一致（"对 oracle 的逼近质量"就是终点）。须防两条：S10 自己报告"匹配更紧≠泛化更好"，故只能作优化误差诊断；且 oracle 教师不得把答案作为**硬编码身份/额外状态**注入（审查拒绝项）。 |
| D9 初始化随机化 | **需改造** | 多初始化取分布＝降低单点运气，成本低、应当采纳。但 warm start 本身是隐式偏置干预，不能当作"纯优化难度"操纵；在本项目里宜只取"多次 fresh init 的分布"这一半。 |
| D10 CKA 表示相似度 | **可转移（限操纵检查）** | CPU 极廉价，可作为"结构是否真的改变了内部组织"的第二读数。受 `AGENTS.md` 约束：不得由状态距离推断假设专门化，故不可上升为 Gate 1 终点。 |
| D11 sharpness 负面装置 | **不可转移（作为证据）** | 该装置的价值是**排除**：NeuroTopo 不应把 sharpness/flatness 作为优化难度或机制有效性的证据。作为"禁用清单"条目可转移。 |
| D12 解析代理 + 梯度饥饿诊断 | **需改造** | S16 的线性代理可作**健全性检查层**，不能作机制证据。S17 的"特征解耦"诊断量可改造成本项目的"分组组织是否真的被使用"读数，CPU 廉价；但理论条件依赖数据统计结构，需在已知生成模型下逐条核对。 |

**必须标出的冲突汇总（学习效率终点）**：D3（grokking 延迟）、D7（"optimize faster"表述）、以及任何以 "steps-to-threshold / time-to-accuracy / 收敛速度" 为主终点的用法，**本质上都是学习效率终点，与决定0062 B 直接冲突**。D1/D2/D4/D5/D8/D10 的终点可全部表述为"最终损失 / 消融损失 / oracle-relative KL / barrier 高度"，不涉及时间，因而与 0062 B 不冲突——但前提是在注册时把终点**逐字**写成非时间量，任何"更快达到"的措辞都会重新触发冲突。

**对"识别性问题"的净结论（方法学层面）**：本领域分离优化误差与近似误差的主流做法，共同结构是**把辩解项变成自变量或独立读数**——D1 变成扫描轴，D2/D3 变成前置证书，D4/D10 变成第二条读数，D5/D7 变成同臂的 oracle/共享随机化参照，D11 变成关闭式负面证据。这些都不需要"两臂比一个标量差"。这与本项目"资源匹配对照会混淆"的处境直接相关：**同臂设计 (within-arm) 不产生需要匹配的第二臂**，因此单轴匹配口径在这些装置上不构成瓶颈。

---

## §4 我未能核验的限制

1. **场所（venue）**：仅 S1–S4、S11 的场所取自 PDF 页眉或 abs 页 JREF 字段的逐字文本；S5–S10、S12–S17 的会议/期刊归属**未核验**，表中一律写"未核验"。
2. **核验层次**：只有 S1–S4 达到 `pdf_section`。S5–S17 全部只到 `abstract`（读到标题与摘要原文），其方法细节、图、超参与失败条件**我未读正文**，§2 中相应描述只基于摘要所声称的内容，并且我已在 §2 中明确标注哪些数字来自 PDF 正文（S3 的 2% 阈值/30 个 α/3×3 运行；S1 的指数 $a\approx1.3$ 与 10k 训练样本），哪些只来自摘要（S8 的四相描述即属此类，我**未**读 S8 正文）。
3. **S5 的标题—ID 对应**：Deep Bootstrap 由 arXiv API `ti:"Deep Bootstrap"` 取回，同一查询的前两条返回**完全不同**的同名论文（`2602.10587` 条件扩散 bootstrap、`2205.15374` Bayesian bootstrap）。我据摘要内容判定 `2010.08127v2` 为 "The Deep Bootstrap Framework: Good Online Learners are Good Offline Generalizers"；HF 与 Crossref 均**未能**检索到该标题，故该条目的元数据只凭 arXiv API 单通道。
4. **通道缺口**：OpenAlex（全 IP 429）、DuckDuckGo HTML（反爬）、DBLP（bot 墙）不可用；EuropePMC 三次查询全部返回生物医学文献，对本领域**零有效命中**；Crossref 该次查询返回语言教学类无关条目。因此本领域的发现**几乎完全依赖 HF papers 检索 + arXiv 核验**两个通道，存在单通道偏差，未命中的工作我无从知道。
5. **未追溯**：我没有回溯任何一篇的引用网络（无前向/后向引文遍历），也没有检索 S1–S17 之外的相关线（如 saddle-to-saddle 的解析文献、edge-of-stability 线、model stitching/CKA 的功能等价线、loss landscape barrier 的可视化线），这些在线索出现后被我主动放弃以保聚焦。
6. **术语边界**：任务指定的起点词中，`approximation estimation optimization error decomposition neural networks`、`steps to threshold training time`、`distillation from oracle teacher`、`supervised fit probe`、`optimization difficulty control architecture comparison` 这几条我**没有**找到直接以该措辞命名的文献；§2 中相应装置是我从取回文献的方法学中**归纳**的，不是原论文的自称。**这不构成新颖性结论**——只说明该措辞未命中。
7. **无本地复现**：本文件不含任何实验、不含任何对本仓库数据的计算；所有"可转移"判定都是**设计层面的判断**，不是已验证的工程结论。
