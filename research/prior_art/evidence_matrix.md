2026-09-29 DIR-0011：[架构资格方法学定向核查](dir0011_qualification_methodology.md)。把 ARCH-0001 一轮沉淀为方法/工具交付物的写作线；已核 arXiv 元数据与摘要：公平调参（Lucic 1711.10337、Melis 1707.05589、Henderson 1709.06560、Ferrari Dacrema 1907.06902）、随机基线（Li 1902.07638、Sciuto 1902.08142）、欠定（D'Amour 2011.03395）、捷径解（Geirhos 2004.07780）、基准饱和/天花板（2602.16763、2607.20864、2605.07395）、硬件彩票（2009.06489）、可复现性程序（Pineau JMLR 2021）与 ReScience C。结论：这些一般主张均被覆盖，不可作为贡献；可交付的是小型合成架构对照的资格检查工具箱＋案例链，四项具体条目（填充安全端点、完整无记忆参照、匹配复现推翻探索优势、负向判词对称区间）在本次检索边界内未见直接对应。正面交付物清单见 `research/paper/positive_deliverables_v1.md`。

# Prior-art evidence matrix

2026-09-28 DIR0009：[受保护状态与经典边界](dir0009_protected_memory.md)。核对量子噪声代数、正交不变量、稳定表示/变化动态、Hopfield纠错；手写关系投影与最近经典码字等价，但此局部结论不否决整个可训练论文设计。见`research/design/dir0009_paper_design_v2.md`；没有外部代码复现或新效果结果。

2026-09-28 DIR-0008 增补：[状态更新代数与观测条件审计](dir0008_state_update_research.md)。13项原始研究分层阅读、1项元数据补查、3个作者仓库固定提交核查；新增DeltaNet/DeltaProduct、有限精度状态追踪、部分可见操作和陪集诊断的边界。ORDER-OBS为未注册独立benchmark草案，无模型效果、无新Gate1候选认证。代码来源与许可见`research/evidence/DIR-0008_sources.json`，代数验收只证明六状态构造。

2026-09-26 DIR-0007 增补：[软等变先例核查](dir0007_soft_equivariance.md)。Ravanbakhsh 2017 参数共享、Finzi 2021 RPP、Kim 2023 等变正则已覆盖候选核心直觉。ORBIT-SOFT 仅为同函数类/同谱先验的受控实验设计，不称新机制；未复现外部数值，未复用代码。待独立设计审查，不计 Gate 1。

初次审计2026-09-23；2026-09-24 增补 PF-RNN 等直接的多粒子信念先例，并以 `fixed_slot_hypothesis_retention.md` 定向核查弱候选保留/后证据修正；同日新增 EviTrack 延迟消歧轨迹选择预印本。范围是用户指定的八类必查家族/代码及后续针对性发现；不是系统综述或穷尽式新颖性检索。下文把论文作者报告、源码可见事实、以及本次没有复现的结果分开。

本轮仅阅读论文、仓库元数据、固定提交中的源码和测试文件；没有安装或运行这些项目，没有重现论文数值，也没有把外部代码带入 NeuroTopo。

SCR-0002 只读复核 PF-RNN 固定 `PFGRUCell` 与作者论文后确认：代码每步调用重采样，论文 Table 3 的 `NoResample` 是 PF-LSTM；抽象无重采样加权粒子与 H3-SLOT 共享低权重恢复的代数，不构成作者 PF-GRU 已实现固定身份槽的证据。具体状态/评分/读出/噪声/BN 边界与非等价反例见 `research/evidence/SCR-0002_noresample_slot_equivalence.md`。本表原有 PF-RNN 行的作者结果和源码审计等级不升级为复现。

| 家族 | 直接证据与已确认重叠 | 与本项目问题的边界 | 源码审计与维护/测试信号 |
|---|---|---|---|
| [Transient / TRNN](https://doi.org/10.1038/s42003-024-07282-3) | Liu 等（*Communications Biology*, 2025）比较瞬态轨迹 RNN 与持续活动的 vanilla RNN；论文任务含嗅觉延迟配对关联（ODPA）、delayed choice 和空间工作记忆。TRNN 加入自抑制、稀疏连接和 sensory–association–motor 层级拓扑。作者报告两类模型可有相同可训练参数数目，并称 TRNN 在部分延迟记忆任务上表现更好。 | 这是“延迟任务中轨迹型动态可改善记忆”的直接重叠。论文任务主要要求保持/匹配记忆内容；本轮未见其将后续相反证据导致的显式假设修订作为目标，也未见同时匹配训练计算与推理计算。因此不能把“轨迹动态能改善延迟记忆”当作新主张。 | [固定提交 81eee36](https://github.com/jiashuncheng/Transient/tree/81eee3611b6f89b2eecb9ffac3cfabb2b7525691)，最后提交 2024-11-08。检查 `ODPA/new/model_torch.py`、`RL/examples/trnn.py`、`README.md`、`RL/test.py`。ODPA 模型逐时步更新活动、阈值/抑制状态并通过 PyTorch 优化器训练；`RL/test.py` 是依赖本机日志路径的分析脚本，不是 ODPA 单元测试。GitHub API 未报告许可证，仓库没有看到面向 ODPA 模型的正式测试套件。**只读源码审计，未复现。** |
| [RIMs](https://arxiv.org/abs/1909.10893) | Goyal 等提出多个近似独立的 recurrent mechanisms，以注意力瓶颈稀疏通信，并仅在相关时刻更新被选中的模块；论文关注环境变化下的泛化。 | 与“多个独立循环分组”最接近。它不是显式的假设/信念修订架构，也不是本项目计划中的“所有分组均运行 + 联合线性读出”：RIMs 另含输入路由、top-k 选择、稀疏更新和模块间注意力。分组 GRU 应称为模块化基线，不应借此声称有 RIM 式选择性机制或语义分工。 | [固定提交 aa61773](https://github.com/anirudh9119/RIMs/tree/aa61773a949c1e3f499d4c05dd77fcd2adbffa19)，最后提交 2022-03-24。检查 `event_based/blocks_core.py`、`event_based/BlockGRU.py`、`event_based/rnn_models.py`、`README.md`。有 copying/adding/Atari 实验代码；未找到仓库许可证或正式单元测试。**论文和源码均只读，未复现。** |
| [Particle Filter RNNs / PF-RNN](https://ojs.aaai.org/index.php/AAAI/article/view/5952) | Ma 等（AAAI 2020）用带权粒子集合近似循环信念状态，PF-GRU/PF-LSTM 以学习到的观测评分、随机转移和软重采样更新；作者报告相对常规门控 RNN 的任务收益。原论文还有一条 PF-LSTM **整体信念**从错误位置恢复的示例及 **PF-LSTM** `NoResample` 消融。 | “多个候选潜在状态、概率竞争和证据更新”已有直接架构先例，比当前全组同输入 M3 更接近功能性假设轨迹。示例不证明同一粒子身份存活，平均任务消融也不能替代晚到反证的预注册测试；论文未报告 PF-GRU 的 `NoResample` 数值。本项目未匹配粒子容量/重采样预算或复现性能。 | [固定作者提交 0a76d52](https://github.com/Yusufma03/pfrnns/tree/0a76d52a857f06d4128bbc7601c616d714b25734)，可见最后提交2024-06-22。检查 `pfrnns.py`、`model.py`、`README.md`、`LICENSE`、`requirements.txt`；GPL-3.0，固定旧 `torch==1.3.1`，未见正式测试目录。**只读源码审计，未安装/运行。** 详见 [定向记录](particle_filter_rnns.md)与 [弱候选审计](fixed_slot_hypothesis_retention.md)。 |
| [EviTrack](https://arxiv.org/html/2605.19283) | Haq（arXiv 2026 v1）在测试时保持有分数的潜在轨迹，局部扩展/选择并可全局剪枝；作者在已知世界模型的双势阱延迟消歧任务中报告相同推理候选预算下优于 SIS/Bootstrap PF 的恢复和预测。 | “歧义保留→后证据选择→更快恢复”及局部选择/全局剪枝权衡已有直接先例。作者实验使用真实模型，不是训练 recurrent 骨干与强单流总状态/参数/训练计算匹配，也不测试生物拓扑、适应、重组或语言迁移。结果未由本项目复现。 | 只读 [arXiv HTML v1](https://arxiv.org/html/2605.19283) 的方法、实验、消融和限制；未核到作者代码仓库、许可或测试，**未运行**。见 [定向记录](evitrack_delayed_disambiguation.md)。 |
| [conn2res](https://doi.org/10.1038/s41467-024-44900-4) | Suarez 等（*Nature Communications*, 2024）发布固定 reservoir + 训练线性读出的工具；支持多种网络结构/节点动力学，并包含 NeuroGym 延迟比较、延迟配对关联、延迟匹配类别等任务。 | 任务家族与延迟证据保持/比较高度相关，是重要边界。其主方法固定循环网络，只训练线性读出；这与要求端到端 BPTT、训练可塑循环状态的主实验不同，不应当作同训练协议的替代基线，也不宜直接作为项目依赖。 | [固定提交 3ccb707](https://github.com/netneurolab/conn2res/tree/3ccb7074261c910847dcd0164b00ff3b02bade90)，最后提交 2024-12-20。BSD-3-Clause。检查 `conn2res/reservoir.py`、`conn2res/readout.py`、`conn2res/tasks.py`、`conn2res/tests/test_reservoir.py`、`test_tasks.py`。`EchoStateNetwork.simulate` 前向递推固定矩阵，`Readout.train` 调用 sklearn 拟合；测试目录存在，但 pinned `test_reservoir.py` 只有说明字符串，任务测试主要检查数据获取。README 列出较旧的 Gym/NeuroGym 依赖。**读码未安装、未运行测试。** |
| [Drosophila connectome reservoir](https://doi.org/10.3390/biomimetics10050341) | Costi 与 Hadjiivanov（*Biomimetics*, 2025）以果蝇连接组构造 ESN reservoir，预测三体问题轨迹；比较时匹配 reservoir 大小、稀疏度、谱半径、输入尺度和读出参数量，并报告连接组结构/权重在所测条件下与过拟合差异相关。 | 展示“生物连接组可作为时间序列 reservoir”与公平匹配控制，任务为混沌时间序列预测，不是可训练 BPTT 或证据冲突后的假设修订。 | 论文关联的 [ESA GitLab 仓库](https://gitlab.com/EuropeanSpaceAgency/fly_connectome) 固定提交 `90679d672cd25905758b3abc6674fa6a28525d52`（2025-05-04）。检查 `src/utils.py`、`src/main.ipynb`、`README.md`、`environment.yaml`。代码实现 ESN 的 tanh/leak 递推、洗出阶段和回归读出；主实验在 notebook 中组织。仓库 API 未报告许可证，未见自动测试目录/CI。**只读检查 notebook 源，不等同重跑实验。** |
| [TuragaLab/flyvis](https://doi.org/10.1038/s41586-024-07939-3) | Lappalainen 等（*Nature*, 2024）以果蝇视觉通路实测连接为结构约束，端到端 BPTT 训练机制网络完成 optic-flow 任务，并用神经活动数据检验预测。训练参数包括细胞型时间常数/静息电位和类型间突触强度缩放；其稀疏连接约束大幅减少自由参数。 | 连接结构、神经动态和任务约束共同塑形轨迹的成熟先例；研究目标是视觉通路机制与神经活动预测，不是通用分组 latent 或假设修订。无需将生物学解释迁移到 NeuroTopo。 | [固定提交 92b3845](https://github.com/TuragaLab/flyvis/tree/92b3845cc426dd309a1a0e1b3890156c42e14021)，最后提交 2026-08-06。MIT。检查 `flyvis/network/dynamics.py`、`network/initialization.py`、`network/network.py`、`solver.py`、`tests/test_dynamics.py`、`tests/test_network.py` 和 `.github/workflows/auto_test.yml`。仓库有多版本 Python CI 和较完整测试目录。**未运行。** |
| [Continuous Thought Machine (CTM)](https://arxiv.org/abs/2505.05522) | Darlow 等（2025 技术报告）将内部迭代、每神经元历史模型与神经活动同步作为表示；论文覆盖图像、迷宫、排序、奇偶、问答和强化学习，并描述依任务难度提前停止的自适应推理计算。 | 对“时间过程可作为内部表示”及推理计算控制有概念重叠；其同步表示、神经元私有历史模型与多组独立 GRU 不同，未见具体延迟冲突证据修订任务。 | [固定提交 4a6c9c3](https://github.com/SakanaAI/continuous-thought-machines/tree/4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466)，最后提交 2025-12-29。Apache-2.0。检查 `models/ctm.py`、`models/modules.py`、`tests/tests.py`、`tests/README.md`。有黄金输出/形状等测试。**未运行或复现。** |
| [ncps / NCP, LTC, CfC](https://github.com/mlech26l/ncps) | 稀疏结构化 recurrent wiring、连续时间常数与 CfC 等模型，是可训练循环动态的实现先例。 | 可以作为连续时间或稀疏 wiring 的参考；与多独立 latent 轨迹、明确假设修订任务并不相同。主线已有 sparse tanh baseline，不需要为回答首个问题引入此依赖。 | [固定提交 695ec22](https://github.com/mlech26l/ncps/tree/695ec22d7d1831f2ced67346ef7d2f08c525bd47)，最后提交 2024-08-14。Apache-2.0。检查 `ncps/torch/ltc_cell.py`、`ncps/torch/cfc_cell.py`、`ncps/wirings/wirings.py`、`ncps/tests/test_torch.py`。PyTorch/TensorFlow 两类测试存在。**未运行。** |
| [Biological Processing Units (BPU)](https://arxiv.org/abs/2507.10951) | Yu 等（arXiv 2025；arXiv 页面标注“accepted to AGI 2025”）把完整果蝇幼虫连接组作为固定 recurrent core，只训练输入/输出投影；报告图像与棋类任务表现。 | 固定连接组 + 训练投影的架构相邻，但任务并不直接测试逐步证据修订；而且这与 BPTT 训练可塑循环状态有明显方法差异。 | 核对 arXiv 摘要、HTML 正文的架构/训练方法和代码链接区域；论文正文没有提供清楚的作者实现仓库链接。本审计**没有确认或检查作者代码仓库**，没有复现论文数字。BPU 在此按论文中明确展开的 Biological Processing Unit 解释；其他同缩写实体不视为该论文实现。 |
| [Neuromorphic hierarchical modular reservoirs](https://doi.org/10.1038/s41467-026-74466-2) | Milisav 等（*Nature Communications*, 2026）以嵌套层级模块构成 recurrent reservoir；论文报告层级组织带来更丰富的神经时间尺度，并改善 memory capacity、multitasking 和 temporal processing。 | 近期的“模块层级→时间尺度池→记忆能力”直接先例，因此“多模块本身提高延迟记忆”不能作为新颖性主张。它使用固定 reservoir + ridge 读出；本轮未确认它测试了后续冲突证据导致的假设修订，也未见 BPTT 训练循环核。 | [代码仓库固定提交 28c1327](https://github.com/fmilisav/milisav_hierarchical_modularity/tree/28c132738775dc0b72802dfb20fa346320ef08ec)，最后提交 2026-04-07。BSD-3-Clause。检查 `network.py`、`task.py`、`run.py`、`README.md`、`LICENSE`。代码使用 `conn2res.EchoStateNetwork` 和 sklearn Ridge；仓库无测试文件/自动测试工作流。**只读检查，未运行。** |
| [Reid multiple-hypothesis tracking](https://graphics.stanford.edu/courses/cs428-03-spring/Papers/readings/CollaborativeProcessing/Reid_MHT_ieee_trans_ac_1979.pdf) | 1979 原论文保持多个数据关联分支，让后续观测重新提高**仍保留**的早期假设概率；也按概率剪枝、按效果合并。 | 多解释与后续修订是直接行为先例；已删分支并未以原身份等待恢复，亦非固定命名的可训练 recurrent 槽。 | 原论文 PDF 已读；作者实现、许可与测试未核，未复现。 |
| [Belief Net differentiable HMM filtering](https://proceedings.mlr.press/v331/chen26c.html) | Chen 等（L4DC/PMLR 2026）将 HMM forward filter 写成可微结构化网络，学习初始/转移/发射概率并递推隐藏状态后验。 | 对有限已知假设，显式后验更新是强简单基线；HMM 状态会转移，不是固定身份轨迹或 H3-SLOT 每槽 GRU，也未核查晚到反证任务。 | 原论文明确关联 [作者仓库](https://github.com/Rockostoneo/BeliefNet-HMM)，本轮只核对 README 与项目关联；未固定提交、核许可证/测试或运行。 |
| [Namikawa–Tani RNN experts](https://arxiv.org/abs/0706.1317) | 2007/2008 论文以多个 recurrent 专家和门控/权重处理切换的时间序列，加入自适应方差学习。 | 多循环状态和加权专家切换已有直接架构相邻工作；专家代表序列原语，不等同于显式假设后验或低权重固定槽恢复。 | 只读原论文 HTML/摘要；作者代码、许可和测试未核，未复现。 |
| [MH-iSAM2 multi-hypothesis inference](https://www.cs.cmu.edu/~kaess/pub/Hsiao19icra.html) | Hsiao 与 Kaess（ICRA 2019）以 Bayes tree/Hypo-tree 追踪歧义变量的多个解，通过剪枝控制组合增长。 | 复杂任务中多解释并存是工程先例；其 SLAM 图优化/剪枝不等于固定 recurrent 槽或本任务的修订收益。 | 只读作者页面摘要；论文 PDF、源码、许可和测试未核，未复现。 |
| [Unified PF-LSTM（2026 预印本）](https://arxiv.org/abs/2609.01967) | Malekzadeh 等在部分观测事件日志中维护带权 recurrent-state hypotheses，并用粒子加权均值和 MGF 特征预测下一活动与停留时间；作者报告三组急诊数据上的效果。 | 近期文献已直接使用“weighted recurrent-state hypotheses”描述多解释序列建模，进一步压缩宽泛语义的新颖性空间；该方法沿 PF-LSTM 粒子路线，未核到同一固定身份槽晚到反证测试。 | 只读 2026-09-02 arXiv v1 的摘要与方法；未见正文明确作者代码链接，未核许可/测试、未复现。 |

## 证据等级

- “论文报告”只表示来源论文如此陈述；不表示本审计重新验证其数据或数值。
- “源码已检查”表示读取了上表列出的固定提交文件；没有检查整个历史、依赖闭包或所有数据文件。
- “未复现”适用于本表所有论文和代码仓库。本审计没有生成任何新的性能证据。

## Structured fields
[evidence_matrix.json](evidence_matrix.json) explicitly tracks Project/Paper, Mechanism, Biological source, Topology role, Dynamics role, Task, Open-source status, What it establishes, What it does NOT establish, Overlap with NeuroTopo, Can reuse code, Mandatory baseline. The detailed source-backed explanations above govern interpretation; machine fields do not upgrade source reports into reproduced evidence.

## EXP0022定向先例补充（2026-09-28）

PSR未来tests与RNN动力学对抗扰动已有直接先例。等当前输出的一次冻结hidden干预、独立未来后缀与同幅控制用于局部机制量化，不宣称新原理或最小半径。范围及原始来源见`exp0022_equal_output_stress.md`。

## EXP0023组合一致性先例（2026-09-28）

稳定/状态正则RNN、组合一致性诊断均已有原始文献；本轮检验一个已知操作关系下的局部输出正则，并保留同操作标签监督对照。诊断有效与训练修复有效分别检验，不宣称首次正则或普遍因果。见`exp0023_composition_consistency.md`。

## EXP0024状态重整

监督原型投影与有限状态正则已有先例；本轮只检验冻结病例能否被自己预测驱动的codebook重整稳定化，额外先验和校准成本显式披露。见exp0024_state_refresh.md。

## EXP0025新seed与缩放对照

有限态码本不是新原理；当前检验新训练seed复现，以及原型形状/原输出与原形状/原型输出两条条件缩放路径。额外监督与不可行事件全部披露；参考exp0025_refresh_replication.md。

## 2026-09-28 跨领域研究设计审计

见 [crossfield_research_practice_audit.md](../evidence/crossfield_research_practice_audit.md)。Mante/Sussillo–Barak 的固定点与局部动力学方法、Littman–Sutton–Singh 的未来条件测试、Geiger 等的表示互换干预均是分析既有 recurrent 状态的直接方法先例；Wang–Niepert 的有限状态正则是监督/可学习码本的最近架构重叠。它们给出未来可测分歧，不证明本项目状态机制或方法新颖性。原论文已读，代码/许可/维护/测试未检查、未运行，论文效果未复现。原始项目 EXP0025 的 R/M 阴性判词及 Gate1 不变。
