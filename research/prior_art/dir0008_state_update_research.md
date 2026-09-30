# DIR-0008：从记忆容量转向状态更新代数与观测条件

同日后续：EXP0017v2已完成本报告所引草案的控制资格，固定四条件未同时取得学习与余量，计划关闭；见`research/evidence/EXP-0017_order_observation_results.md`及决定0087。以下为该实验之前的文献与设计审计，不追认候选效果。

2026-09-28；研究报告，非系统综述。事前范围：`research/design/dir0008_research_protocol.md`。结论是 **DESIGN_DIRECTION_IDENTIFIED / NO_GATE1_CANDIDATE_CERTIFIED**。最值得保留的是“操作组合 × 观测可见性”的判别设计，不是新增一种已证明有效的 NeuroTopo 模型。

## 1. 对当前实验的诊断

本轮实读 git、stage gates、EXP0016注册与 live loop：ITER-0011 / EXP-0016 v1，DECIDED / INCONCLUSIVE / CONTROL_CEILING。强GRU的固定OOD平均KL为0.01698163；任何非负KL模型相对该批预测的平均改进至多为此数，小于原δ=.02。该界不覆盖其他任务、预算或未来模型分布。ORBIT候选A未训练，不能说其失败。当前固定配置关闭。

原设计反复把“历史有信息”当作“结构有可利用的控制余量”的前置线索，但两者相差很远。有限HMM的完整后验可被强单流近似；噪声、状态数或延迟增加，并不必然产生组织优势。**新的设计必须先指出哪种更新操作在实际对照中受限，随后允许GRU否决它的项目价值。** 原有G1-RS、互反图搜索、ORBIT配置不因此重开。

## 2. 检索与证据范围

检索关键词簇：recurrent independent mechanisms / modular generalization；soft equivariance / mixed symmetries；fast weight / delta rule / memory interference；HiPPO / nonnormal memory；state tracking / negative eigenvalues / Householder products / noncommutative；probabilistic automata / state reveals / cosets。通过web搜索、arXiv、会议论文、作者GitHub取证。非英语及全部2026论文未穷尽；不能用未命中证明新颖性。旧FlyVis、conn2res、TRNN、CTM、ncps、BPU沿用仓库历史审计，本轮未重新检查它们的当前代码/维护状态。

下表“报告”均指作者陈述；没有本地复现论文效果。正文范围为定向阅读，不表示读完全部附录。理论使用限定假设，不把任意长度表达性下界当作有限长度训练必败。

| 原始来源／本轮阅读范围 | 对设计有用的证据 | 边界或反例 |
|---|---|---|
| [RIMs，2019/ICLR2021](https://arxiv.org/abs/1909.10893)，摘要；仓库已有方法审计 | 独立循环组、稀疏通信、选择性更新已经组成已有机制。 | “多模块”本身不新；完整RIM包含当前阶段未开放的通信，不能直接整体搬入。 |
| [HiPPO，NeurIPS2020](https://proceedings.neurips.cc/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html)，会议摘要 | 历史压缩可从指定测度的多项式投影导出。 | 更丰富时间尺度须面对这种算法基线；摘要不能支持任意任务优越性。 |
| [Schlag等，ICML2021](https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf)，§3–4 | 线性注意力与fast weights关系、容量限制及delta纠错写入已有直接先例。 | 局部写入/更新目标关联不是新原理；正交键能消除许多人工串扰。 |
| [Kim等，ICML2023](https://proceedings.mlr.press/v202/kim23p/kim23p.pdf)，摘要、§1–2 | 近似/混合对称性下，硬等变约束可失配；软正则已有方法。 | 支持研究先验失配边界，不支持回调旧ORBIT噪声或预算。 |
| [Mamba，2023/2024](https://arxiv.org/html/2312.00752v2)，摘要、选择性SSM方法入口 | 输入依赖更新用于筛选信息。 | “输入决定何时保留/更新”已有先例；不能以此声称假设竞争。 |
| [Illusion of State，ICML2024](https://arxiv.org/html/2404.08819v2)，§2–5 | 区分记忆与操作组合，给出受限SSM表达性分析。 | 有界深度/精度等假设及复杂度类分离前提必须保留；不适用于任意GRU，也不证明本稿S3有限任务有理论难度分离。 |
| [Unlocking State-Tracking，ICLR2025](https://arxiv.org/html/2411.12537v4)，§4、有限精度附录；先读v2后核v4 | 负/复特征结构影响计数与状态追踪能力。 | 负特征值不是一般充分条件；论文使用特定cast/运算模型，不能当作逐步IEEE浮点执行的普遍定理。 |
| [Gated DeltaNet，ICLR2025](https://arxiv.org/html/2412.06464v3)，§2–3与消融入口 | 全局遗忘和定向delta更新互补。 | 属现有模型包；LM收益不能推出当前项目门通过。 |
| [DeltaProduct，NeurIPS2025](https://arxiv.org/html/2502.10297v6)，§3–6；先读v3后核v6 | 多个Householder更新提高表达性，作者也明确增加参数/计算；重复相同键会退化。 | 因此“多做几步胜单步”有计算解释；组词任务及反射乘积已有近邻，不能宣称新机制。 |
| [Learning State-Tracking from Code，2026-02预印本](https://arxiv.org/html/2602.14814v1)，§5及附录B | 部分可见操作和排除式观测引出信念状态；特定线性表示会有范数衰减。 | 这是最重要反例：确定性追踪成功不能外推到信念修订。衰减不等于信息必然丢失；归一化/log-domain滤波须纳入。 |
| [Held-Out Transition-Pair Falsifier，2026-06预印本](https://arxiv.org/html/2606.07254v1)，协议与§6.5 | 把局部转移模板留出作为抗记忆检验；附录保留同因子非交换对。 | 主划分跨因子操作本来可交换；不能将任意相邻pair留出叫非交换检验。本稿不引用其长程性能为已验证事实。 |
| [Mamba-3低秩非交换更新，2026-09-23预印本](https://arxiv.org/html/2609.28273v1)，§III–IV及限制 | 显式推导低秩反射更新的交换子；本文也强调转移性质不保证学习成功。 | 与拟议方向高度重叠；是刚出现的未复现预印本，不能据其优于Mamba的作者报告推断胜GRU。 |
| [Tracking States or Tracking Cosets，2026-09-24预印本](https://arxiv.org/html/2609.29951v1)，摘要、干预协议定向阅读 | 部分准确率可能来自陪集解；作者用子空间交换和共享后缀研究行为。 | accuracy或状态距离不足以说明完整状态追踪；其均匀全群输入定理不直接适用于本稿生成元输入，更不授予Gate3语义结论。 |

非正规动态线补查了 [Ganguli等，2008原论文元数据/摘要](https://pubmed.ncbi.nlm.nih.gov/19020074/)；PMC正文被验证码阻断，故本轮不据此提出具体非正规性定理或推进非正规图候选。

## 3. 三条路线的取舍

**优先：操作的相对几何与有序组合。** 对线性转移 `A_a,A_b`，`[A_a,A_b]=A_a A_b−A_b A_a`刻画该转移部分的顺序敏感性。各矩阵单独同谱，不保证它们的乘积或交换子相同。它给出了比“模块数更多”更具体的操纵对象。链条是：连接/共享子空间 → 输入条件转移组合 → 有序状态表示 → 操作组合 → 状态追踪。若含输入注入或状态依赖门控，不能只看A的交换子就判全模型是否顺序敏感。

**第二：选择性写入与干扰。** delta-rule、GDN、RIMs已直接覆盖。若重新研究，必须在固定键Gram矩阵、相同总存储下比较本地/分布编码，并用正交键控制换基等价；只有相关键下胜出不能证明拓扑原则。学习路由、在线适应、模块通信不作为当前实验的附加补丁。此线本轮不形成候选。

**第三：近似对称性或时间尺度。** 可检验结构先验与失配程度的交互，但软等变/RPP/HiPPO已有强先例。原ORBIT无强控制余量，不能通过改终点为早期学习速度、换λ、加难度把关闭结果重新包装。本轮保留为背景，优先级低于操作组成问题。

## 4. 已落实的设计优化

具体草案见 `research/design/dir0008_order_observation_v0.md`。它把六状态C3×C2与S3任务配成一对：状态数、生成元阶数和单个生成元循环型相同，而两个操作能否交换不同；再交叉完全可见/随机遮蔽操作，并增加相同可见观测。这样能分别检查操作表达、信念修订和有限预算学习。

最强常规控制仍是全输入/全读出的GRU；先例控制为允许负值的DeltaNet和DeltaProduct；算法基线是六状态精确滤波及可学习转移滤波。没有任何一项已被本轮证明有模型余量。**这个任务问的是已知更新家族何时有用，失败也能决定停止哪类机制；它不是H3多候选的改名续跑。** 拟定轨道为独立benchmark，项目计分不作请求。

可操作的突破是把实验从“哪种组织更好”改为“什么更新结构能表达任务、观察不确定后优势是否保留”。最可能有价值的结果包括GRU全面胜出、先例只在完全可见条件有益、或两者都封顶；均不能解释为NeuroTopo获得正结果。

## 5. 代码可复用性核对

固定提交及逐文件SHA见 `research/evidence/DIR-0008_sources.json`。源码只读，未安装依赖、未执行作者代码。

| 仓库与固定提交 | 已看文件、许可、测试与维护证据 | 本轮决策 |
|---|---|---|
| [automl/DeltaProduct](https://github.com/automl/DeltaProduct/tree/d62241a81d07aa32b1b65e7d17377f6a7cd0a5d8) | 根与state_tracking LICENSE均MIT；generate_data.py与gated_deltaproduct.py定向核对；实际有每步独立投影及allow_neg_eigval配置。树中有vendored FLA tests，但未核专用DeltaProduct覆盖或测试是否通过；最新提交2025-10-13，README称后续维护在FLA。 | 方法/生成器参考。现成训练依赖Triton/GPU，不直接用于Mac CPU；未来采用经验证CPU递推包装。 |
| [automl/unlocking_state_tracking](https://github.com/automl/unlocking_state_tracking/tree/68f8e5fc6d25043d4e6a1fc169b58281c9e01d12) | 阅读delta_net_no_triton.py的beta缩放及模式分派；该vendored子树MIT，非整个仓库统一许可判断。存在naive/triton测试路径，未运行；最新提交2025-03-15。 | 文件名含no_triton仍导入相关算子，不能宣称开箱即用CPU。 |
| [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet/tree/b53d6d3a161267432a79c1c04af69fa52bddc921) | LICENSE为Nvidia Source Code License-NC；看lit_gpt/gated_delta_net.py的beta、状态及chunk路径。树路径筛查未发现名字含test的文件，不等于不存在所有验证。最新提交2026-03-13。 | 不复制；不能因另一FLA实现MIT就将此仓库标MIT。 |

## 6. 完成范围与限制

完成当前证据复核、13项原始研究的分层阅读、另1项元数据级补查、3个作者仓库的固定提交定向检查、设计草案及代数核查。未训练、未生成随机研究样本、未改实验注册或Gate状态。外部临时venv已不存在，本轮以`PYTHONPATH=src`和系统Python3.13运行纯标准库loop/status及审计检查；这不验证Torch训练环境。

这不是覆盖全部领域的穷尽综述，尤其没有完整重推所有表达性定理。完成后的独立反思及修订记录在 `research/reflections/DIR-0008/`；以其中corrections为本报告解释的最终限定。
