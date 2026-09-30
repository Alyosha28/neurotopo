# DIR-0003 A 区：资源匹配下的架构比较方法学（真实取回）

2026-09-24。本文是 DIR-0003「新问题的领域方法审计」的 A 区，B–G 区见同目录其余 `dir0003_*` 文件。**本区任务**：相邻领域用什么实验设计，在"匹配对照本身会改变结论""调参预算不等""基准可选"的处境下仍得到可辩护的架构/机制归因。

## 0. 检索通道与证据等级（必须与结论分开读）

- `web_search`：**不可用**，`HTTP 402 Insufficient Balance`（endpoint `api.deepseek.com/...`）。
- `web_fetch`：**不可用**，对 arxiv.org / api.semanticscholar.org 等一律返回 *"resolves to a non-public IP address"*（本机 DNS 解析到 `198.18.x.x` fake-IP 段，透明代理）。
- bash `curl`：**可用**，本区全部来源经此取回。
- `api.openalex.org`：本轮**额度耗尽**（`Rate limit exceeded ... free daily budget shared by everyone on your network's IP`），A 区未使用。
- DuckDuckGo HTML：**首次查询后即反爬**（返回 anomaly 页）；Mojeek / SearxNG / DBLP 同样被挡。A 区未使用。
- 实际使用通道：`export.arxiv.org/api/query`（元数据）、`arxiv.org/abs/<id>`（摘要逐字）、`arxiv.org/pdf/<id>` + `pdftotext`（正文段落）。
- **证据等级**：§1 表内标注每条核到哪一层。**无本地复现**；未运行任何外部代码；未检查任何外部仓库的许可或测试。

## 1. 取回来源（逐字标题来自取回页面/API 响应）

| # | 逐字标题 | 年 | id | URL | 本次核验层次 |
|---|---|---|---|---|---|
| 1 | The Efficiency Misnomer | 2021 (ICLR 2022) | 2110.12894 | https://arxiv.org/abs/2110.12894 | **全文 PDF**（§3.1、§3.2、§3.2.1 段落） |
| 2 | The Benchmark Lottery | 2021 | 2107.07002 | https://arxiv.org/abs/2107.07002 | **全文 PDF**（摘要 + §1 + 目录） |
| 3 | On the State of the Art of Evaluation in Neural Language Models | 2017 | 1707.05589 | https://arxiv.org/abs/1707.05589 | 摘要逐字 |
| 4 | Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches | 2019 | 1907.06902 | https://arxiv.org/abs/1907.06902 | 摘要逐字 |
| 5 | Show Your Work: Improved Reporting of Experimental Results | 2019 | 1909.03004 | https://arxiv.org/abs/1909.03004 | 摘要逐字 |
| 6 | Deep Reinforcement Learning that Matters | 2017 | 1709.06560 | https://arxiv.org/abs/1709.06560 | 摘要逐字 |
| 7 | A Metric Learning Reality Check | 2020 | 2003.08505 | https://arxiv.org/abs/2003.08505 | 摘要逐字 |
| 8 | Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer | 2022 | 2203.03466 | https://arxiv.org/abs/2203.03466 | 摘要逐字 |
| 9 | Training Compute-Optimal Large Language Models | 2022 | 2203.15556 | https://arxiv.org/abs/2203.15556 | 摘要逐字 |
| 10 | Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design | 2023 | 2305.13035 | https://arxiv.org/abs/2305.13035 | **仅元数据**（arXiv API 标题行） |
| 11 | An Empirical Study of μP Learning Rate Transfer | 2024 | 2404.05728 | https://arxiv.org/abs/2404.05728 | **仅元数据** |

## 2. 方法设计装置

### A1 「没有公平比较」时披露多轴而不宣布胜者
- **出处**：#1（§3.1–3.2.1 原文）。
- **识别问题**：两个不同族架构在同一标量资源轴上匹配，但该轴与真正相关的成本并不一致。
- **原文可引用内容（本次核到正文）**："not every parameter is created equal"；参数匹配与 FLOP 匹配是**两种互不等价**策略；"the relative positions of different model architectures are not consistent as the performance metric is varied"；"there is no model that is clearly more efficient here and conclusions may differ strongly depending on which efficiency metric is employed"；当确实无法做公平比较时，作者的建议是**尽力给出一个最佳匹配设置并展示多个备选**，而不是宣布单一结论。
- **需要什么**：多轴实测（参数/FLOPs/吞吐/时延），而非估计。
- **为什么比"两臂比一个标量差"更抗混杂**：它把"选哪条轴"这一自由度**显式暴露**为结论的一部分，使结论无法仅靠轴选择获得。
- **失效条件**：若某轴是机制**实际消耗**的轴（如推理候选预算），多轴披露不能替代该轴的匹配。

### A2 基准抽签：单一任务的排序不可外推
- **出处**：#2。摘要逐字："many factors, other than fundamental algorithmic superiority, may lead to a method being perceived as superior. On multiple benchmark setups that are prevalent in the ML community, we show that the relative performance of algorithms may be altered significantly simply by choosing different benchmark tasks"。
- **识别问题**：候选/对照的排序是**任务抽样**的结果，而非机制属性。
- **做法**：预注册**任务电池**（多个任务族）或**留出任务族**；报告排序一致性，而非单一任务上的差值。
- **需要什么**：多任务实现成本；每个任务各自的控制资格。
- **为什么抗混杂**：把"任务选取"从一个可被事后操纵的自由度变成**事前冻结的设计维度**。
- **失效条件**：任务电池内部高度相关（同族）时，抽签方差不降——本项目 CAL-0008 正是"同族换配置买不到新信息"的实测版本。

### A3 调参平价（tuning parity）：先给基线同等预算，再谈增益
- **出处**：#3、#4、#7。
  - #3 摘要逐字："these have been evaluated using differing code bases and limited computational resources, which represent uncontrolled sources of experimental variation. We reevaluate several popular architectures and regularisation methods with large-scale automatic black-box hyperparameter tuning and arrive at the somewhat surprising conclusion that standard LSTM architectures, when properly regularised, outperform more recent models."
  - #4 摘要逐字："we report the results of a systematic analysis of algorithmic proposals ..."（对近年推荐算法提案的系统分析）。
  - #7 摘要逐字："We find flaws in the experimental methodology of numerous metric learning papers, and show that the actual improvements over time have been marginal at best."
- **识别问题**：新架构的"增益"其实是**基线未被调优**（不同代码库、有限算力、不同预算）的产物。
- **做法**：给每个臂**相同数量/相同搜索空间规模**的调参；把"基线的最优"作为比较对象；报告调参**空间**上的分布而非单点。
- **需要什么**：调参预算翻倍（时间成本），以及可复现的搜索空间定义。
- **为什么抗混杂**：把"调参运气"从臂的属性变成设计的**共同预算**。
- **失效条件**：搜索空间本身对某臂更有利（不对等空间 = 换了另一种不公平）；本项目用"冻结 LR 网格"部分满足，但**未报告搜索空间上的分布**。

### A4 报告"性能 vs 预算"曲线，而不是单点最好成绩
- **出处**：#5。摘要逐字："we demonstrate that test-set performance scores alone are insufficient for drawing accurate conclusions about which model performs best ... We present a novel technique for doing so: expected validation performance of the best-found model as a function of computation budget (i.e., the number of hyperparameter search trials or the overall training time)."
- **识别问题**：单点最优成绩不可比，因为各臂的开发预算不同。
- **做法**：主终点改为**预算→期望最优验证性能**的曲线（或该曲线在冻结预算处的取值）。
- **需要什么**：多次开发试验 + 验证集；预算轴必须实测。
- **为什么抗混杂**：调参预算成为**显式报告轴**而非隐含差异。
- **失效条件**：本项目主轨（决定0062 B）目前**不允许**把学习效率/曲线类终点作项目主终点——此装置在新设计中只能作次级或另轨。

### A5 超参迁移（μTransfer）：让小尺度调参在大尺度仍然有效
- **出处**：#8。摘要逐字："in the recently discovered Maximal Update Parametrization (muP), many optimal HPs remain stable even as model size changes ... parametrize the target model in muP, tune the HP indirectly on a smaller model, and zero-shot transfer them to the full-sized model"。
- **识别问题**：大规模比较中"每臂都调到最优"成本不可承受，于是比较退化为"谁的超参恰好更好"。
- **做法**：用 μP 参数化，在**小模型**上完成调参，把 HP 零样本迁移到大模型，使各臂都拿到自己尺度上的近似最优 HP。
- **需要什么**：μP 实现与该架构的宽度/深度缩放关系；验证迁移有效性（#11 是这一点的后续经验研究，本次仅取回标题，**未核验其结论**）。
- **为什么抗混杂**：消除"臂间调参预算不等"这一最常见审稿拒绝。
- **失效条件**：对**非标准/循环**结构，μP 的最优 HP 稳定性本身需要先验证；本项目 CPU 小模型上收益有限。

### A6 计算最优前沿（iso-compute / frontier）而非单预算点
- **出处**：#9、#10（#10 仅元数据）。#9 摘要逐字："We investigate the optimal model size and number of tokens for training a transformer language model under a given compute budget ... for compute-optimal training, the model size and the number of training tokens should be scaled equally."
- **识别问题**：在单一预算点上比较，结论可能只在该点成立（哪一臂恰好处在自己的最优点）。
- **做法**：在每个预算下各臂**各自**选其最优配置，比较**前沿**（截距 + 斜率）而不是单点。
- **需要什么**：多预算 × 多配置的训练矩阵（成本高）。
- **为什么抗混杂**：把"谁恰好被放在了最优点"变为设计的一部分；斜率差不容易被单点调参运气伪造。
- **失效条件**：成本；且斜率型终点在本项目主轨触及决定0062 B。

### A7 种子方差与显著性：不许"最好一次"出场
- **出处**：#6。摘要逐字："non-determinism in standard benchmark environments, combined with variance intrinsic to the methods, can make reported results tough to interpret. Without significance metrics and tighter standardization of experimental reporting, it is difficult to determine whether improvements over the prior state ..."。
- **识别问题**：报告的提升落在种子方差内。
- **做法**：多 seed、报告不确定区间与配对统计量、统一报告规范。
- **需要什么**：seed 预算。
- **为什么抗混杂**：把"随机抽取的一次运行"变成分布上的比较。
- **失效条件**：seed 数不足时区间宽到无法判定（本项目 A1–A7 的多份停止判词都撞在这上面）；**区间宽不是效应不存在的证据**。

## 3. 对 NeuroTopo 的可转移性判定

| 装置 | 判定 | 依据与限制 |
|---|---|---|
| A1 多轴披露 | **已被采纳**（`G1-MATCH-v3` 全轴披露） | 本项目比 #1 更严：v3 要求**声明消耗轴必须匹配**，A1 只要求披露。无新增内容。 |
| A2 任务电池 / 留出族 | **可转移，且未被满足** | 本项目 `G1-MATCH-v2` 曾定义"通用层"谓词（≥2 任务族或 1 留出族），0068 被 0069 取代后该操作化**消失**；现行 Gate 1 文本仍写 `Generic`，但**没有可核验的通用性判据**。这是一个**真空缺**。 |
| A3 调参平价 | **可转移，部分满足** | 本项目用冻结 LR 网格（如 EXP-0013 的两 LR）——是"同等菜单"，但**没有报告搜索空间分布**，且菜单规模远小于 #3/#7 的自动搜索。可作为新设计的显式条款。 |
| A4 曲线终点 | **与现行规则冲突** | 决定0062 B 禁止项目主轨用学习效率/曲线类主终点（未决）。只能作次级探索或另轨。 |
| A5 μTransfer | **不可直接转移** | 面向宽度/深度缩放；本项目是固定小 CPU 模型 + 循环结构，μP 对循环块的稳定性**未核验**。 |
| A6 前沿比较 | **与现行规则冲突（同 A4）**；且成本高 | 斜率/前沿型终点在主轨受限。 |
| A7 种子分布 | **已被采纳** | 本项目一直用 seed 配对、CI、全尝试保留。**残余问题不是方法缺失，而是功效不足**（多份停止判词由 `MDD > δ_min` 类分辨率上限触发）。 |

**A 区的净结论**：本项目在 A 区（比较方法学）**已经采用了大部分最佳实践**，因此**"再改比较方法学"不是新问题的来源**。A 区唯一暴露的真实空缺是 **A2 的"通用性判据"在 0069 之后失效**——Gate 1 文本要求 `Generic`，但当前没有任何可核验的操作化。另外 A7 表明：靠增加 seed/收紧统计量救不了**分辨率上限**，那是设计层问题（效应量级 ≤ 自身混杂量级），必须在**问题选择**层解决。

## 4. 本次未能核验的部分

- #10、#11 仅取回标题与年份，**未核验摘要或正文**；不得据此引用其结论。
- #2 只读到摘要、§1 与目录，**未**逐节核验其任务电池与"抽签"统计构造。
- #3–#9 的**具体数字**（提升幅度、预算规模）未核验；本文只引用其摘要中逐字可查的**方法学主张**。
- 未检索非 arXiv 场所（期刊/会议论文集）的方法学工作；未做引用图扩展。
- 未检查任何外部仓库的许可、测试或维护状态；未复现任何结果。
- 本文件不构成新颖性结论，也不改变 Gate 1 状态或任何既有判词。
