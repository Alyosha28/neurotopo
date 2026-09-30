# DIR-0005 先例补查：C7 类（"为可训练性而设计的组织"）的标准操作已被覆盖

2026-09-25。**性质：审计（`research/evidence/DIR-0005_candidate_spec_audit.md`）在 packet 冻结之后新增的取证，专为收窄其唯一未否证类别 C7 的 `K7 = UNKNOWN`。** 不是判词、不是注册、不改变 Gate 1 `NOT_PASSED`。证据为**元数据级**（标题/年份/DOI/引用数来自 OpenAPI 响应，**正文未读**）。

## 1. 为什么要查这个

审计把 `G1-MATCH-v3` 与 K3 的合取读成：可过门的效应只能是**同容量下的优化动力学效应**（C7）。于是绑定约束从"余量"移到了 **K7 新颖性**：若 C7 类里那些"为可训练性而设计的组织操作"都已被先例覆盖，则 C7 目前同样没有可注册内容。本轮据此检索四类标准操作。

## 2. 检索通道

| 引擎 | 端点数 | 结果 |
|---|---|---|
| OpenAlex | 5 | 全部 `200` |
| Crossref | 5 | 全部 `200` |
| arXiv | 0 | 本轮未查（DIR-0004 期已记录该通道 `429 Rate exceeded` 与超时；本轮不再浪费配额） |

脚本 `scripts/dir0005_prior_art_fetch.sh`，原始响应 `research/prior_art/dir0005_raw/`，逐条日志 `fetch_log.txt`。

## 3. 命中（逐字标题与标识）

| 组织操作 | 逐字标题 | 年 | 标识 | 引用数 |
|---|---|---|---|---|
| 正交/受约束循环权重 | SGORNN: Combining scalar gates and orthogonal constraints in recurrent networks | 2023 | `10.1016/j.neunet.2022.11.028` | 2 |
| 正交参数化 | Fast orthogonal recurrent neural networks employing a novel parametrisation for orthogonal matrices | 2019 | `10.1016/j.sigpro.2019.05.003` | 0 |
| **正交初始化对优化的可证收益** | Provable Benefit of Orthogonal Initialization in Optimizing Deep Linear Networks | 2020 | `10.48550/arxiv.2001.05992` | 60 |
| 时间尺度分离（显式时钟） | Flexible Clockwork Recurrent Neural Network for multirate industrial soft sensor | 2022 | `10.1016/j.jprocont.2022.09.008` | 50 |
| 多时间尺度层级组织 | Emergence of Functional Hierarchy in a Multiple Timescale Neural Network Model: A Humanoid Robot Experiment | 2008 | `10.1371/journal.pcbi.1000220` | 490 |
| 结构化状态空间初始化 | Efficiently Modeling Long Sequences with Structured State Spaces | 2021 | `10.48550/arxiv.2111.00396` | 500 |
| 神经微分方程（连续时间组织） | On Neural Differential Equations | 2022 | `10.48550/arxiv.2202.02435` | 124 |

## 4. 判断（对 C7 的收窄）

**四类标准"可训练性组织"操作全部有直接先例**，其中"正交初始化带来可证优化收益"是**定理级**覆盖（第 3 行）。因此：

- C7 的 `K7` 由 `UNKNOWN` 收窄为：**这四类操作不可作为新颖性来源**；任何 C7 候选必须（i）指名一个**不在这四类内**的组织操作，或（ii）对既有操作给出**新的、可数字否定的机制预测**（而非"它也许有帮助"），并做到正文/定理级核查。
- 这不否定 C7 作为**类别**的可行性（EXP-0004 是其可行实例），但说明**类别内的空位比审计当时估计的更窄**。
- 与审计的 T1/T2 一致：把组织操作选在"表征上几乎等价、只在优化路径上不同"的窄带里，同时还要避开这批先例，是当前组合为空的直接原因。

## 5. 边界与未取回项

- **元数据级**：正文与定理条件**未读**；任何注册前的 K7 判定必须升级到正文/定理级。
- 本轮**未**检索的方向（可能仍有空位，但需要单独取证）：任务侧"何种函数类使某组织操作的优化优势可被**水平量**捕获"；以及非标准操作（如非正态/瞬态放大、精确重整化的**任务侧**而非数值侧用途）。
- 不改变任何判词、注册或阶段门；不重开非回归清单任何条目。
