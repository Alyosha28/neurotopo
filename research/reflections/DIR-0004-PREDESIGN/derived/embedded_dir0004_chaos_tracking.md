# DIR-0004 先例核查：混沌潜轨迹跟踪的"精度分离"与不稳定子空间组织

2026-09-24。**核查对象**：DIR-0004 事前设计 v0 的两条主张——(C1) 隐藏不稳定模态的跟踪所需精度随视界线性增长；(C2) 在匹配状态预算下，显式扩张/收缩状态组织买到更长的可跟踪视界。**目的**：履行 AGENTS.md"实现机制前先查相关先例"，并结清设计稿 §9 自认的欠账（"与精度/混沌跟踪直接对应的定理级来源未取回"）。

**结论先行**：两条主张**都有直接先例**。C1 是教科书可预报性理论；C2 的机制属于成熟的**不稳定子空间同化（AUS）**领域，且"卡尔曼滤波的误差协方差坍缩到不稳定子空间"已有专文。故本设计**不得主张机制或定律新颖**，只能主张一个受限的、匹配预算下的架构/优化命题——这一点已并入设计 §3（见 `research/design/dir0004_claim_correction_v0.md` §3）。

**核验层次（重要）**：本轮的证据是**元数据级**——标题、年份、DOI、引用数来自 OpenAlex / Crossref 的 API 响应，**正文均未读取**。任何依赖具体定理陈述或数值的推论都标为"未核验（正文未读）"。原始响应逐字留档在 `research/prior_art/dir0004_raw/`，检索日志同目录 `fetch_log.txt`。

## 1. 检索方式与通道事实

| 引擎 | 端点数 | 本轮结果 |
|---|---|---|
| Crossref | 5 | 全部 `200` |
| OpenAlex | 9 | 全部 `200` |
| arXiv API | 9 | **全部失败**：`http` 端点返回 `301`；改 `https` 后 4 条返回 **`429` ＋ 正文 `Rate exceeded.`**（14 字节），5 条 `curl (28) Operation timed out … 0 bytes received`（>`--max-time 45`）。**该通道本轮未取得任何结果**，与 DIR-0003 期记录的通道不稳一致 |

`web_search` / `web_fetch` 本会话**未使用**（DIR-0003 期已记录：`web_search` 返回 HTTP 402、`web_fetch` 因本机 DNS 为 fake-IP 段而对每个主机返回 "resolves to a non-public IP address"）。本记录只使用 `curl` 直连的 API 元数据。

脚本：`scripts/dir0004_prior_art_fetch.sh`（原始响应 ＋ 逐条 `引擎/名称/HTTP 码/查询串` 日志）。补充查询由同一命名约定写入同一目录。

## 2. C1（精度—视界定律）：教科书结果

| 逐字标题 | 年 | 出处 | 标识 | 引用数 |
|---|---|---|---|---|
| Predictability: a way to characterize complexity | 2002 | Physics Reports 356(6) 367–474 | `10.1016/s0370-1573(01)00025-4` | 312 |
| Nonlinear finite-time Lyapunov exponent and predictability | 2007 | Physics Letters A | `10.1016/j.physleta.2006.11.094` | 152 |
| Visual Analysis of Nonlinear Dynamical Systems: Chaos, Fractals, Self-Similarity and the Limits of Prediction | 2016 | Systems 4(4) 37 | `10.3390/systems4040037` | 341 |
| Periodic orbits in chaotic systems simulated at low precision | 2023 | Scientific Reports | `10.1038/s41598-023-37004-4` | 24 |

**判断**：C1 即经典可预报时间关系 `T_p ≈ (1/λ)·ln(Δ/δ)` 的等价重写（精度每提高 1 比特，可预报视界延长 `ln2/λ` 步；`λ = ln 2` 时恰为 1 步/比特）。**"1 步/比特"不是本项目的发现**，是本设计在**一个特定生成器＋固定精度存储模型**下的重算。该重算仍有价值：它是本任务族的**可核注册预测**，且非线性复核（帐篷 `0.971`、逻辑斯蒂 `0.932` 步/比特）是**我们对这一模型类自己做的数值核对**，不是文献结论的引用。**正文未读**：`T_p` 公式的精确形式与适用条件未逐字核对（元数据级证据）。

## 3. C2（不稳定子空间组织）：成熟领域 AUS ＋ 协方差坍缩

| 逐字标题 | 年 | 标识 | 引用数 |
|---|---|---|---|
| Assimilation of Standard and Targeted Observations within the Unstable Subspace of the Observation–Analysis–Forecast Cycle System | 2004 | `10.1175/1520-0469(2004)061<0103:aosato>2.0.co;2` | 80 |
| Adaptive observations and assimilation in the unstable subspace by breeding on the data-assimilation system | 2007 | `10.1111/j.1600-0870.2006.00210.x` | 44 |
| Detecting unstable structures and controlling error growth by assimilation of standard and adaptive observations in a primitive equation ocean model | 2006 | `10.5194/npg-13-67-2006` | 20 |
| Four-dimensional variational assimilation in the unstable subspace and the optimal subspace dimension | 2010 | `10.1002/qj.571` | 93 |
| **On the Kalman Filter error covariance collapse into the unstable subspace** | 2011 | `10.5194/npg-18-243-2011` | 61 |
| Lyapunov vectors and assimilation in the unstable subspace: theory and applications | 2013 | `10.1088/1751-8113/46/25/254020` | 58 |
| Four-dimensional ensemble variational data assimilation and the unstable subspace | 2017 | `10.1080/16000870.2017.1304504` | 76 |
| Data Assimilation Fundamentals | 2022 | `10.1007/978-3-030-96709-3` | 211 |

**判断（对设计最要紧的一条）**：`On the Kalman Filter error covariance collapse into the unstable subspace`（2011）表明"误差协方差（即不确定性）集中到不稳定子空间"是该领域的**既有结果**。据此，本设计 §2 的机制直觉——"收缩方向的信息会被动力学自动抹掉，存储它们是纯浪费"——**不是我们的观察**，而是 AUS 的理论出发点之一。2010 年那篇更进一步讨论**最优子空间维数**，2013 年给出理论与应用综述。

**这如何改变 C2 的表述**：设计不能再把 C2 说成"显式扩张/收缩组织是否有用"（AUS 已给出理论上的"是"）。**变得可问且未被覆盖的只剩**：在**匹配状态预算**（同 `H` 个浮点）与匹配执行计算下，**一个被声明如此组织的架构**能否比**一个未被这样声明的稠密控制**更接近 oracle——这是**架构与优化**层面的受限命题，不是表示论层面的命题。若连这一点也在 oracle 拟合探针下失败，则本设计没有可注册的内容。

## 4. 任务族与学习方法：混沌预测本身是标准基准

| 逐字标题 | 年 | 标识 | 引用数 |
|---|---|---|---|
| Data-driven forecasting of high-dimensional chaotic systems with long short-term memory networks | 2018 | `10.1098/rspa.2017.0844` | 350 |
| Using a reservoir computer to learn chaotic attractors, with applications to chaos synchronization and cryptography | 2018 | `10.1103/physreve.98.012215` | 111 |
| Data-driven predictions of a multiscale Lorenz 96 chaotic system using machine-learning methods: reservoir computing, artificial neural network, and long short-term memory network | 2020 | `10.5194/npg-27-373-2020` | 166 |
| Chaos as an intermittently forced linear system | 2017 | `10.1038/s41467-017-00030-8` | 630 |
| Emerging opportunities and challenges for the future of reservoir computing | 2024 | `10.1038/s41467-024-45187-1` | 337 |
| Machine Learning With Data Assimilation and Uncertainty Quantification for Dynamical Systems: A Review | 2023 | `10.1109/jas.2023.123537` | 302 |
| Combining data assimilation and machine learning to emulate a dynamical model from sparse and noisy observations: A case study with the Lorenz 96 model | 2020 | `10.1016/j.jocs.2020.101171` | 206 |

**判断**：用 RNN/储备池预测混沌系统、以及"数据同化＋机器学习"的混合路线，都有大量成文先例。故 DIR-0004 的**任务本身**（跟踪/预测混沌动力学）**不构成任何新颖性**；它作为任务族被选中的理由是**筛查闸的形式**（"精确解不存在有界维充分统计量"），这是**方法学理由**，不是新颖性主张。

## 5. 对设计的净影响

1. **§3 必须点名 AUS**，并把"误差协方差坍缩到不稳定子空间"列为既有结果；本设计的主张收窄为"匹配预算下的架构／优化命题"。
2. **C1 不得写成发现**，只能写成**本任务族内的可核注册预测**（且已由本项目脚本在自己声明的模型类上复算）。
3. **C2 的判定权仍在资格阶梯第 1 级**（oracle 拟合探针）：AUS 只说明"该组织原则在表示论上正确"，不说明"在匹配预算下某个具体架构能兑现它"。
4. **未取回项**：arXiv 通道本轮全部失败（`429 Rate exceeded` 与超时混合，逐条见 `fetch_log.txt`）；AUS 的**正文与定理条件未读**；"最优子空间维数"的具体判据未读。若审查或后续注册需要定理级引用，须另开一次通道可用的检索，并**不得**用本条记录冒充正文级证据。

## 6. 原始响应索引

`research/prior_art/dir0004_raw/`：`crossref_{aus1,pred1,rc1,lat1,pred_review}.json`、`openalex_{aus1,aus2,prec1,rc1,lat1,aus_review,aus_ml,prec_bits}.json`、`arxiv_*.xml`（6 个含 `Rate exceeded.`、3 个空，失败已逐条记录）、`fetch_log.txt`（逐条引擎/名称/HTTP 码/查询串）。
