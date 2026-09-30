# DIR-0011：架构资格方法学的定向先例核查

2026-09-29 定向核查，针对"把 ARCH-0001 一轮的过程沉淀为方法/工具交付物"这一写作线。核查在**论文提纲之前**完成，用于确定可主张与不可主张的边界。检索方式与边界：`arxiv.org` API 元数据与摘要直取（`export.arxiv.org/api/query`，检索日期 2026-09-29），加 JMLR 与 ReScience C 官方页面；**只读了标题与摘要**，未读正文、未复现代码、未核作者实现。检索词包括：fair baselines / evaluation pitfalls / underspecification / shortcut learning / benchmark saturation / ceiling effect / neural architecture search random baseline / reproducibility program / preregistration / memory benchmark / padding-masked evaluation / trivial solution / headroom。

## 一、被直接覆盖的主张（不可主张为贡献）

| 主张 | 直接先例（已核元数据与摘要） | 重叠程度 |
|---|---|---|
| "强基线必须调参后才可比较；不调参会得出错误的方法排序" | Lucic 等，*Are GANs Created Equal? A Large-Scale Study*（arXiv:1711.10337，2018）；Melis 等，*On the State of the Art of Evaluation in Neural Language Models*（arXiv:1707.05589，ICLR 2018）；Henderson 等，*Deep Reinforcement Learning that Matters*（arXiv:1709.06560，AAAI 2018）；Ferrari Dacrema 等，*Are We Really Making Much Progress?*（arXiv:1907.06902，RecSys 2019） | 完全覆盖；本项目不能把"要公平调控制基线"当发现 |
| "搜索/随机基线可以追平复杂方法" | Li & Talwalkar，*Random Search and Reproducibility for NAS*（arXiv:1902.07638，2019）；Sciuto 等，*Evaluating the Search Phase of NAS*（arXiv:1902.08142，2019） | 完全覆盖 |
| "同一流水线可给出多个行为不同的模型（欠定）" | D'Amour 等，*Underspecification Presents Challenges for Credibility in ML*（arXiv:2011.03395，2020） | 完全覆盖"实现选择影响结论"的一般主张 |
| "捷径解/平凡解使基准失去判别力" | Geirhos 等，*Shortcut Learning in Deep Neural Networks*（arXiv:2004.07780，Nature MI 2020） | 概念完全覆盖；本项目的"显式解参照"是其在具体任务上的实例 |
| "基准饱和/天花板效应掩盖差异" | *When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation*（arXiv:2602.16763，2026）；*Position Bias is Hidden Behind Ceiling Effects*（arXiv:2607.20864，2026）；*Unsolvability Ceiling in Multi-LLM Routing: An Empirical Study of Evaluation Artifacts*（arXiv:2605.07395，2026） | 主题高度活跃且已被占位，但均在 LLM 评测语境；本项目是小型合成架构对照语境 |
| "硬件/算力差异是混淆项" | Hooker，*The Hardware Lottery*（arXiv:2009.06489，2020） | 一般主张覆盖 |
| "可复现性清单与流程" | Pineau 等，*Improving Reproducibility in ML Research*（JMLR 22(78)，2021，已核 JMLR 页面）；ReScience C（期刊，页面已核，专门接收复制研究） | 实践层面完全覆盖 |
| "预注册/注册报告" | 心理学注册报告制度成熟；ML 语境已有预注册与独立复现研究（如 arXiv:2609.33843，2026，摘要含 preregistered 与 independent reproduction） | 流程不新；**未找到**把 packet-only 反思＋哈希链用于"候选架构资格"的小样本案例 |

## 二、检索未命中的部分（不等于新颖，只说明本次边界内无直接对应）

1. **"确定性填充位置抬高聚合准确率"**：以 `abs:"padding" AND abs:"accuracy" AND cat:cs.LG` 检索，命中都是模型结构中的 padding（卷积/MoE 路由），**未见**把它作为"评测端点被确定性位置污染"的方法学条目。
2. **"无记忆基线集合不完整"**：以 `abs:"trivial solution" AND abs:"benchmark"`、`abs:"headroom" AND abs:"baseline"` 检索均无对应条目；未见把常数策略、直通、逐位置多数作为必需参照集的规范。
3. **"匹配算力复现推翻探索性优势"这一具体链条**：未见同类记录（NAS/GAN/RL 的公平比较文献覆盖"调参后排序变化"，但均非"先出探索性优势、再以匹配算力控制复现为负"的账目式案例）。
4. **"负向判词的对称不确定度标准"**（正向要求区间下界>0、负向只看点估计，会从跨 0 区间读出方向性结论）：检索未命中 ML 领域针对此点的条目；等价性检验（TOST）与区间决策属统计学成熟内容，但**未见到把该不对称作为基准判词缺陷来写的条目**。

## 三、边界声明（写入任何写作）

- 本项目**不主张**发现"调参公平性""捷径解""天花板效应""可复现性清单""预注册"等一般现象；这些均已有直接先例，见上表。
- 可主张的窄范围是：在**小型合成状态追踪任务上做候选架构 vs 强控制的资格与匹配对照**这一设置里，给出一套**可执行的资格检查与参照集**（显式解参照、填充安全端点、完整无记忆参照、受界 oracle、控制菜单算力匹配、负向判词对称区间），并用一条**包含两次自我推翻的案例链**证明这些检查各自能挡住什么。
- 该窄范围仍是"未找到直接对应"，不是"首次"；写作必须逐项标注与上述先例的重叠，并把条目 1–4 作为待进一步检索的问题保留。
- 结构层面的负结论（3 维不变量状态＋标量读出下，直接重生成控制类的函数类包含嵌套假设计算候选，优势只能来自优化偏置）在本仓库内可自证，但它是对**特定模型族**的陈述，不是对"嵌套假设计算"的一般否证。
