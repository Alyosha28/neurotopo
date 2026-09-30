# G1-KSI v0 草案审查：action 摘要

审查 packet 所需的最小仓库／执行证据，核对日期 2026-09-24。

## 当前项目状态

- `python -m neurotopo.loop status`（按 `docs/reproduction.md` 使用 `/private/tmp/neurotopo-venv`）显示当前为 EXP-0013 / ITER-0009，阶段 `DECIDED`，判定 `INCONCLUSIVE`。已分析事实：冻结阶段显露任务有信息，但 SC64 与 D64_full 在两档 LR 下均未通过 IID32 学习资格；RS4 未接受任务训练。独立 review 已完成，Gate 1 仍 `NOT_PASSED`。
- `research/decisions/0054_dir0002_direction_portfolio.md` 判为 `RULE_CHANGE_FIRST`，要求前瞻裁定：交叉耦合条件是否构成决定0049意义上的另一任务族；学习效率能否作为 Gate 1 主终点。
- `research/open_questions.md` 的 TH-0001 条写明：在任务有理由、资源匹配且实际控制有余量的前提下，另行注册的 K×model learning-efficiency 或 transfer 交互仍逻辑可测；该结论不支持以增加 K 论证容量优势。
- 决定0049只关闭现行 G1-RS 阶段显露任务的继续搜索，并要求另一个任务族先通过独立审查。决定0053要求实际控制每臂按预写性能规则选 LR；过去协议不得被本草案追认。

## 规则原文摘录

- 决定0049 §1：“只有新的独立架构机制/研究问题通过事前批评后，才能提出另一个任务族。”
- 决定0054 §2.1：要裁定结构失配中的交叉耦合是否构成决定0049所指的“另一个任务族”。
- 决定0054 §2.2：要裁定学习效率（updates-to-ε／学习曲线面积）能否作为 Gate 1 主终点，以及是否需要前瞻版本化。
- `research/open_questions.md` TH-0001 原文：“A separately registered K×model interaction on learning efficiency or transfer remains logically testable if it has a justified task, matched resources and actual-control headroom; TH-0001 only stops unmotivated K inflation as a capacity story.”
- `research/stage_gates.md` Decision0043 原文摘要：Gate 1 可做“explicit independent recurrent organization against resource-matched sparse and strong dense single-GRU controls”；效应识别有限预算组织／连接偏置，不代表纯 K 数或 Gate 3 语义。
- `docs/experiment_protocol.md` §3要求在实际主控制上独立校准 oracle-relative headroom；§6要求学习曲线逐检查点记录累计样本、token、更新、实测 wall time 与 scoped compute estimate。

## 草案承诺

`research/design/g1_k_structure_learning_efficiency_v0.md` 标记为 `DRAFT_FOR_RULE_REVIEW_AND_DESIGN_CRITIQUE`，提出学习效率为有限预算 estimand、K={2,4} 与 factorized/cross-coupled 条件、保度随机稀疏 primary control、强稠密 guard、AUC_B 主终点、updates-to-ε 次终点、种子级推断及不改写旧结果。数值门槛、正式种子数、预算 B、epsilon、模拟器与 oracle 资格仍明确未冻结；文件声明无新数据、无训练、无 Gate 变化。

`research/design/g1_m3_sham_intervention_method_pilot_v0.md` 把旧 M3 检查点作为零训练只读方法试点，明确块没有候选身份，排除 H3-semantic 和 Gate 1 解释。

## 完整设计文本（供 packet-only 复核，不要求审查者读取工作区）

### G1-KSI-0001 主设计

问题：在相同隐状态总量、训练样本和每步执行计算下，块独立递归组织相对同度数随机稀疏递归的学习效率优势，是否随任务的潜变量耦合结构改变？研究对象是有限预算组织／归纳偏置，不是容量下界、纯 K 容量或槽语义。

待议规则解释A：交叉耦合是同一新注册生成器的预定结构条件，而非对已关闭 EXP-0013 生成器的再次搜索，前提是变量、观测模型、目标、长度、同步、状态字母表、先验族、边缘先验与边缘信号不变，仅耦合矩阵改变。该改变造成的联合先验／熵变化属于处理本身，需报告和资格校准。不得重开阶段显露任务；独立审查若认定新问题仍是 G1-RS 搜索延续，则 Gate 1 分支关闭或另行定位为独立 benchmark。

待议规则解释B：学习效率可作有限预算 Gate 1 行为终点，但须新注册且不追认历史曲线。控制在独立资格流上须到达冻结误差阈值并仍有动态范围；资格失败不能改称架构效率效应。

候选设计：`z∈{-1,+1}^4`，Ising先验`p_J(z)=Z_J^{-1}exp(Σ_(i,j)∈E J_ij z_i z_j)`，无外场，`E={(1,2),(2,3),(3,4),(4,1)}`。固定F矩阵J=0，X矩阵四条环边权重均为0.25，不筛其他矩阵；`y_t=z+ε_t`, `ε_t~N(0,I4)`, T=16，每步同步观测四维，oracle对16态按Gaussian log likelihood和Ising prior精确枚举。模型K={2,4}固定。H=64独立块模型对同状态、同边数／度数、自环和有效递归参数的保度随机稀疏模型是主比较；相同完整输入、投影、门和读出，dense kernel形状一致，循环掩码是唯一模型连接差异。全输入稠密H64 GRU为强学习保护臂。F/X共用状态字母表、联合支持、单变量边缘先验、长度、观测同步和噪声；联合熵因J改变，明确当作处理特征并报告。该生成器/oracle仍未编码或资格验证。

主要终点：固定预算B内、按两条件共用联合状态支持 `log |Z|` 归一化的 oracle-relative posterior KL 对累计更新的梯形积分 `AUC_B`，越低越好；updates-to-ε为次终点。设 `D_{k,c}=AUC_B(random_sparse)-AUC_B(block)`，`β_c=(D_{4,c}-D_{2,c})/log(4/2)`，主交互 `I=β_F-β_X`，预注册方向 `I>0`。资格须用独立流核对 F/X oracle、状态支持、控制共同 ε、oracle-relative 难度可比性和非瞬达动态范围；数字容差待冻结。若难度／动态范围容差不通过，结果只限于这两种生成条件的总交互，不能窄称耦合矩阵本身的效应；主控制任一条件学习失败则停止正式效应比较。K仅两级，结论是有限差分。B、ε、最小效应、正式种子、oracle资格容差和bootstrap尚未冻结。所有模型种子是推断单位，保留每个失败种子与资源记录。

停止与解释：主控制未学会、任务/oracle资格失败、资源不匹配或区间跨阈值均不支持主张且不改任务参数救结果；效应只限已测条件和预算，不支持候选语义、自然任务、生物拓扑或普遍K scaling。

### G1-M3-SHAM-0001 方法试点

只读使用10个EXP-0009 M3最终权重，在新生成且分离的校准／测试episode上运行。每个checkpoint和每条流都按固定顺序逐一测试4个架构块，不按结果选块。在最终readout前，将同一16维扰动向量加到当前块；sham将完全相同向量放置于独立随机抽取的16个hidden坐标。支持大小、向量、L2幅度、时间点和计算相同，只改放置位置。幅度由校准流的隐藏状态RMS按预写比例确定；校准／测试各1024 episodes／每个d∈{1,4,8}。分别测联合KL和H/R边缘KL，形成4块×3延迟×2输出的固定24维签名；使用cosine similarity比较跨流稳定性，零范数记0。每个checkpoint计算目标签名跨流相似度减100组预生成sham相似度的中位数；10个训练种子为单位，配对95%CI下界>0为方法试点正向，否则否定或不确定。扰动比例和sham随机种子仍待完整协议冻结。没有候选身份，因此不能得出H3-semantic或Gate1结论。

## 主终点与资格规则补充

Episode先抽一个静态`z~p_J`，给定z后16个四维观测条件独立；模型完整读取`y_1...y_16`并输出16态posterior `q_e(z)`，训练目标是oracle soft-target交叉熵`−Σ_z p_e(z)log q_e(z)`。同seed/cell的block与random-sparse共用batch和held-out episodes；train、qualification、formal test彼此分流。

每个held-out episode的checkpoint损失为`KL(p_e||q_e)/ln16`；`L_j`是固定held-out episodes的均值，`u_j`是累计optimizer updates。唯一主终点为`AUC_B=(1/B)Σ_{j=0}^{M−1}(u_{j+1}−u_j)(L_j+L_{j+1})/2`，checkpoint比例固定`u_j/B∈{0,0.10,0.25,0.50,0.75,1}`，B须为20的倍数。主端点针对唯一矩阵对与四个K×F/X单元，不根据结果换网格或终点。

种子级对比：`D_{k,c,s}=AUC_B(random_sparse)-AUC_B(block)`；`D_{k,c}=mean_s D_{k,c,s}`；`β_c=(D_{4,c}−D_{2,c})/(ln4−ln2)`；`I=β_F−β_X`。逐seed交互`I_s=[(D_{4,F,s}−D_{2,F,s})−(D_{4,X,s}−D_{2,X,s})]/(ln4−ln2)`，CI按配对模型seed重抽样20,000次、PCG64分析seed 694201，使用percentile 95%CI。方向是I>0。唯一裁定：CI下界>`δ_min`且所有资格／资源门通过则SUPPORT；CI上界<`δ_min`则FALSIFIED最小有用预测（如CI仍正，另报正但低于阈值）；CI覆盖δ为INCONCLUSIVE。`δ_min`、模型seed数、B数值和所有资源容差尚未冻结，必须在资格数据解封前锁定。

次要updates-to-ε使用同一个共同阈值ε；定义为首次达到ε的checkpoint，未达到者记`>B`和达标率，不插补且不能替代AUC_B。Qualification须在K=2/4各自的F/X四个random-sparse primary-control单元和F/X两个dense H64 guard单元分别确认控制达到共同ε、落在预写非瞬达动态范围，并通过oracle-relative难度、state、compute及实测时延标准。任一控制资格单元失败就停止，不删格、不合并F/X、不换endpoint、不开始正式效果轮。正式block候选即使某单元未达ε也按原计划计入AUC_B；其次要updates-to-ε记`>B`，不得剔除。

## 权重与外部先例

- `artifacts/EXP-0009/v1/` 有 10 个 `multi_gru/checkpoints/final.pt`。保存的是 M3 检查点；它们没有候选指派。
- 后续定向搜索补记于 `research/prior_art/dir0002_direction_prior_art.md`：Nature Communications 2024 的 resource-constrained specialization 研究报告结构模块化不保证功能专化，并使用模块消融；arXiv:2309.16042 报告 activation-patching 指标与干预选择会影响结论；group recurrent networks 论文讨论分组递归与运行时实现。只检查到网页论文信息，未审代码、提交、许可或测试；这不是全面先例审计或新颖性判断。

以上首条记录首轮只读论文网页的状态；本轮又对下述固定源码提交做了只读检查。

补充源码核查：`specialization-dynamics` 固定到 `520dd88183a6fcf51ae6995956829ea10c2d906c`，检查 `DoubleDataset.cov_ratio`、模块消融汇总与 masked-readout ablation；该快照未见 license 或专用 tests。`coactivation-closure` 固定到 `7e48183c0ee6f97c0d720337e1b50cd37880049e`，检查 attention-head 置零和 5 个同规模随机头控制；MIT、7 commits，未见专用 tests。两者均未运行或复用。D1/D4 先例边界已收窄，完整证据见 prior-art 文件。

重叠判定：D1 的“任务变量 covariance × 模块结构／资源约束”已有直接先例；本地问题只能定位到解析后验任务中的 K={2,4} 学习曲线 interaction，不声称首次发现任务结构条件化模块收益。D4 的“组件消融＋matched-random＋逐样本损伤”已有直接先例；本地 M3 试点是 recurrent posterior state 上的应用，不是新方法。两项具体组合都未被本次有限搜索证明为新颖。

## 核验边界

本摘要未生成样本、未运行模型、未修改旧配置或实验注册。独立审查若发现上述事实与仓库冲突，应要求核对；审查本身不能把拟议规则变成正式决定。
