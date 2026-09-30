# Stage gates

## 现行架构研究阶段门 `ARCH-GATES-v1`（2026-09-29 起，仅向前适用）

**目标**：从可检验的拓扑与动力学原则出发，设计、训练和检验完整的人工计算架构。阶段门限制的是**能够宣称什么**，不是能否提出候选、写原型或做有边界的诊断。旧的 Gate 0–9 和 `G1-MATCH-v3` 保留为历史实验的判据；下面的 A0–A6 不重判任何旧实验，也不自动让旧 Gate 1 通过。重写前的权威文件已[逐字节归档](archive/stage_gates_pre_arch_gates_v1.md)，来源与哈希见[归档记录](archive/stage_gates_pre_arch_gates_v1.provenance.json)。规则变更与适用范围见[决定0101](decisions/0101_architecture_stage_gates_v1.md)。

| 门 | 可作出的主张和通过条件 | 当前状态 |
|---|---|---|
| A0 候选定义 | 冻结一个**完整架构候选**的状态、更新、读出、训练方式和资源路径；说明区别于简单循环网络及相关先例的可证伪预测、最小消融、预期失败情形。单个重整算子或一项微任务诊断不算完整候选。 | `SC-REL-v3: A0_CONTRACT_FROZEN / DESIGN_ONLY`（[合同](design/arch0001_successor_compatible_core_v3.md)、[决定0102](decisions/0102_arch0001_a0_candidate_contract.md)）；只确认候选定义完整，尚无候选实现或学习证据。 |
| A1 可运行计算 | 候选与强控制在预定任务上达到基本学习资格；端到端训练、状态更新和测试可复现；任务有信息与非封顶余量，记录实测资源和工程失败。通过只说明候选能计算。 | `INCONCLUSIVE / NO_QUALIFIED_CONDITION`（[EXP-0028 v1](../experiments/EXP-0028/v1.md)、[报告](evidence/EXP-0028/report.md)、[决定0103](decisions/0103_exp0028_a1_qualification.md)）：工程8项与独立重放通过，IID8 前测 `CAND` 2/3 seed、`INV_COMP` 3/3 seed；控制侧 8 个预注册条件无一致命合格（σ≤.15 全封顶或近封顶，σ=.4 时 N5 保持均值 .5989 差 .0011、N8 随机水平），且显式 Gram 重编码参照在 σ≤.15 达 100%。A1 未通过。学习资格另经 [EXP-0029 v1](../experiments/EXP-0029/v1.md) 12 新 seed+前 3 seed 复现为 `PARTIALLY_SUPPORTED / LEARNING_QUALIFICATION_AT_BUDGET`（三臂均 12/12；`CAND` 合计 14/15，唯一失败为 `HOLD_COLLAPSE`；[决定0104](decisions/0104_exp0029_learning_qualification.md)），任务非封顶余量缺口仍未解决。 |
| A2 架构效应 | 在预注册任务、独立训练实例和有效强控制上，**完整候选**达到预写的最小有意义收益及不确定性标准；声明主要资源预算并在比较中控制或剖析，披露其余资源比值。可先得到任务与预算内的**整套架构效果**，无须预先证明每个组件独有贡献；该结果不能直接归因于某个组件。 | `FALSIFIED`（ARCH-0001 在已测范围内）：[EXP-0033](../experiments/EXP-0033/v1.md) 探索性 +11.26pp（控制算子为候选 1.82 倍，非匹配）未在 [EXP-0034](../experiments/EXP-0034/v1.md) 的合同口径控制（算子比 0.912）与全新种子上复现——候选 0.6637 对控制 0.7261（−6.24pp，95% CI [−15.04,+4.10]pp），同监督无修正臂 0.6717 与候选持平（[决定0109](decisions/0109_exp0034_falsification_and_route_closure.md)）。候选 `SC-REL-v3/v4` 在已测任务与预算内无匹配计算优势；其它候选仍有各自独立门。旧 EXP0004 的负效与 EXP0027 的局部反例不能计作通过。 |
| A3 计算机制 | 用候选的定义性消融及同一完整状态的配对干预，检验关键状态或运算对已见效应的因果作用；检查未来命令响应、任务答案及简单替代解释。没有 A2 效应时可报告局部机制，但不能声称它解释架构优势。 | `NOT_EVALUATED`；EXP0027 仅提供已暴露 REL 状态的局部兼容性反例。 |
| A4 泛化与资源 | 冻结候选和选择规则，在未见训练实例、至少一个未见任务条件或任务族、较长序列和资源预算变化下验证效应与失败边界；效率或扩展性各自有独立基线和端点。 | `NOT_EVALUATED`。 |
| A5 生物原则归因 | 若主张源于生物拓扑，比较生物、生成、保度/模块/模体匹配和可学习拓扑等适当对照，验证抽象原则对 A2–A4 效应的因果贡献。人工架构的局部收益本身不授予此门。 | `NOT_EVALUATED`。 |
| A6 自然序列与规模竞争 | 在 A1–A4 已有对应证据的前提下，分别注册自然序列建模、语言模型、扩展效率及与 Transformer/Mamba 的竞争主张；使用相应有力模型、数据、质量与实际计算/时延对照。任何前门通过均不自动授予本门。 | `NOT_EVALUATED`。 |

**A2 资源口径 `ARCH-MATCH-v1`。** 在看候选结果前声明主要预算轴，以 pilot 实测核验并在正式比较中匹配；`SUPPORTED` 还须匹配或作算子级实际计算剖析。状态、参数、样本、更新、内部步数和时延全部披露。单点比较只支持该预算附近的结论；跨预算效率须比较质量与资源曲线。整套架构可以获得限定范围的 A2 效果判词，但组件因果归因须另过 A3。

**实验准入与判词。** 允许并行开展有边界的架构设计、组件诊断、领域基准和生物资料审计；进入某条证据轨时标明目标门、候选版本、先例、最强反驳、对照、资源预算、停止条件和成功/失败各会改变什么。只有 A2–A6 的正式效应或归因主张需要相应预注册和独立复核；A0/A1 的设计及工程通过不能冒充性能结果。既有 `gate1_candidate_first_v2` ResearchLoop 入口仍服务历史 `project_gate1` 合同；新门不能用该字段伪装为已通过旧 Gate 1，直到另有显式迁移实现。冻结后更换任务、候选、指标或匹配口径须新版本；失败与不确定结果保留。

**推进规则。** 可以先构造适应、通信、竞争或重整的最小原型，前提是说明它检验哪条预测并给出同预算简单对照。不能因微任务局部成功跳过完整架构比较，也不能因一个候选失败封锁所有后续架构设计。A3 诊断可先于 A2 进行，但只能获得局部机制判词；A5/A6 的强主张仍需各自所列的先行效应与独立验证。任何阶段门的通过都按候选、任务、资源和版本限定，不是整个 NeuroTopo 假说的永久通过。

**历史文本提示。** 下方旧表中的“Gate 1 不过则停止其他机制”的语句只解释当时的执行规则；自 `ARCH-GATES-v1` 生效后，它不再阻止 A0/A1 设计或有界组件研究。旧实验的资格、匹配和判词仍按其注册时规则裁决。

## 历史 Gate 0–9 与实验记录（保留原判词）

2026-09-29 EXP0032v2/0033/0034完成 A2 通道收尾：v2 扫完控制菜单（`DELAY-2` 上 w32b2 精确解出、多项落带；预注册匹配选中的 w48b2 5 seed 均值 0.6455<0.65，QUALIFICATION_NOT_REPRODUCED；v1 因未注入 width/blocks 的实现缺陷作废）；EXP0033 fresh seed 确认载体 w96b2（0.6978）并得探索性优势 +11.26pp（95% CI [+0.35,+23.69]，控制算子 1.82 倍候选，非 A2）；EXP0034 用合同口径控制 w48b2（算子比 0.912）与全新种子复现得 −6.24pp（候选 0.6637 对控制 0.7261，同监督无修正 0.6717）→ **ARCH-0001 在已测范围内判 FALSIFIED 并关闭**（决定0109）。A1 仍未通过，旧 Gate1 不变。

2026-09-29 EXP0031 v1完成 DELAY-d 任务资格的修正端点复核：同任务/同训练/同种子重训 12 个运行，与 EXP0030 归档哈希 12/12 逐位一致；修正端点（r∈[d+1,T]）与扩充无记忆参照下，d=1 封顶 1.0000、d=2 0.5991、d=3 0.6109（门槛 0.65），参照≈0.50、oracle=1.0，菜单内无合格条件。判词 INCONCLUSIVE / NO_QUALIFIED_MEMORY_CONDITION，取代 EXP0030 的 SUPPORTED/A2_TASK_QUALIFIED（决定0106）；A2 仍未评估、A1 未通过、旧 Gate1 不变。

2026-09-29 EXP0030 v1完成 A2 记忆寄存器任务族注册与控制侧资格：`DELAY-d`（读出头输出 `u_{r−1−d}`，命令 i.i.d.）6 配置×2 seed、每条件 2000 更新（共 30000 更新/15 拟合）。d=1 两监督均封顶 1.0000；**d=2 PLAIN 0.6916**、d=3 0.7306 可学非封顶；无记忆参照≈0.54、oracle=1.0。取首个合格 `D2_PLAIN`，独立 3 seed 确认 0.6970（0.6922–0.7009），干净副本重放 2 运行差 0。判词 SUPPORTED / A2_TASK_QUALIFIED（限任务/预算/种子），无候选训练、无 A2 效果、A1 与旧 Gate1 不变。见报告与决定0105。

2026-09-29 EXP0029 v1完成 ARCH-0001 学习资格 12 seed 复现：`CAND`/`EXTRA_SUP`/`INV_COMP` 三臂均 12/12 通过，共 36000 更新、36 拟合；合并前 3 seed 后 `CAND` 14/15（唯一失败为 HOLD_COLLAPSE）、`INV_COMP` 15/15；干净副本重放 3 运行哈希/指标差 0。判词 PARTIALLY_SUPPORTED / LEARNING_QUALIFICATION_AT_BUDGET，A1 仍因缺少非封顶合格任务而未通过，A2–A6 未评估，旧 Gate1 不变。见报告与决定0104。

2026-09-29 EXP0028 v1完成 ARCH-0001 A1 实现与任务资格：工程8项全通过、干净副本重训3运行哈希/指标差0；IID8 前测 `CAND` 2/3 seed、`INV_COMP` 3/3 seed；控制侧8条件无一合格（σ≤.15 封顶或近封顶，σ=.4 N5 保持均值 .5989<.60、N8 随机），显式 Gram 重编码在 σ≤.15 全时点100%。判词 INCONCLUSIVE / NO_QUALIFIED_CONDITION，A1 未通过，A2–A6 未评估，旧 Gate1 不变。见报告与决定0103。

2026-09-29 ITER0021 / EXP0026v2完成1196主轨迹+104独立τ轨迹。PARTIALLY_SUPPORTED / SINGLE_EXPOSED_SCHEDULE_WINDOW：主条件失败25002/25003/25008中仅25002通过完整窗口；k2/4/8/16 H/U在256/512/1024均100%，k32/64 HOLD50%，R/I同窗口。25003 k2/4恢复HOLD但长期UPDATE未达标。50%经逐类核查源于单类读出收敛；原始审计1300/1300、预定复演200/200精确。见EXP-0026_schedule_drift报告与决定0099，Gate1不变。

2026-09-28 DIR-0010完成用户提出的重整间隔×漂移时间×子空间机制设计先行审计：leak0时共同正交旋转对REL/INV的Gram与读出为精确gauge；26条输入、13,644,800个float32矩阵的最大正交残差1.430043e−7，仅是数值输入性质，不排除动态放大。读出核平面不等于慢变量；H1–H4保留为待测假说。DYN0002 v0矩阵和独立反思已归档，0研究模型执行/训练，决定0098；EXP0025/DYN0001/Gate1不变。

2026-09-28 DYN-0001 v3完成等当前logit的t8单次重整长期轨迹诊断：原256步前缀加冻结后缀至512/1024步，26模型×2任务×5条件。新REL25002 HOLD的NONE/FULL/EQUAL三个视界均50%；EXP0025每8步刷新在原256步为100%，因此早期单次干预不够，仍未区分重复与晚期单次。REL25003 UPDATE随长度衰减。v2数值批次复演失败（0完成单元）与26后缀保留，v3原256步NONE精确重放。报告DYN-0001_equal_logit_future_observability.md，决定0097；EXP0025与Gate1判词不变。

2026-09-28 ITER0020 / EXP0025v2完成12新seed×REL/INV及2旧参考。R_NOT_SUPPORTED（FULL8修复1/5），M_NOT_SUPPORTED（EQUAL8有效子组1/2），总体INCONCLUSIVE。新25002固定logit重整75%→100%，仅幅度调节仍75%，保留局部机制见证；其余种子损伤与不可行回退完整报告。报告EXP-0025_refresh_replication.md，决定0096；Gate1不变。

2026-09-28 ITER0019 / EXP0024v1完成冻结状态原型重整，主预设REL23005病例SUPPORTED / FROZEN_CASE_STATE_REFRESH_RESCUE：LONG_H末步50%→100%，H/U平均75%→100%；GAUGE8仍75%。12监督codebook拟合、0神经更新。其余训练seed已暴露，群体区间跨0，REL23004存在损伤，经典有限态投影解释保留。报告EXP-0024_state_refresh.md，决定0095；Gate1不变。

2026-09-28 ITER-0018 / EXP-0023 v2完成15旧诊断＋六新seed四臂训练，24000更新。INCONCLUSIVE / LEARNING_GUARD_FAILURE：CYCLE/EXTRA共同IID仅3/6，P学习资格不足、M未支持、R不成立；CYCLE未稳定优于同操作EXTRA，也平均劣于原REL，INV六seed长程均100%。报告`research/evidence/EXP-0023_composition_repair.md`，决定0094。此固定输出一致性候选未获支持，Goal以完整限定结果结案；Gate1不变。

2026-09-28 ITER-0017 / EXP-0022 v2完成15冻结模型×3半径等logit压力测试，研究更新0。INCONCLUSIVE / NO_REGISTERED_WITNESS_FOUND：约束/baseline全通过，但INV四seed未见注册未来损伤；REL20004有半径相关次要损伤且定向≈随机。候选族额外保总范数，有限搜索阴性不证明全等输出鲁棒。报告`research/evidence/EXP-0022_equal_output_stress.md`，决定0093；Gate1不变。

2026-09-28 ITER-0016 / EXP-0021 v1完成四新seed重复与冻结状态干预：12000更新，INCONCLUSIVE / LEARNING_OR_ENGINEERING_UNRESOLVED（R=false、M=false；无工程失败，REL20005长程/NONE不合格）。INV四新seed全部8原cell末步100%；REL存在长程保持/更新失败。即时模长/方向交叉未重复，次要后续更新敏感性不能替代主终点。报告`research/evidence/EXP-0021_replication_intervention.md`，决定0092。Gate1不变，本轮无新任务/调参。

2026-09-28 ITER-0015 / EXP-0020 v2 completed one-seed RAW/INV/REL learning (3000 updates), SUPPORTED / SINGLE_SEED_LEARNABILITY_ONLY (decision0091). All three learn IID retention/update; INV and REL retain performance under stronger shared rotations while RAW degrades. This is exploratory learned-memory evidence, not REL superiority or a Gate1 result. Strong INV results remain primary context; Gate1 stays NOT_PASSED.

2026-09-28 ITER-0014 / EXP-0019 v2 completed frozen paired-history dynamics: D64 float32 has 0/1152 new exact opposite-b state mergers through128 shared suffix operations; INCONCLUSIVE / NO_NEW_EXACT_MERGER_OBSERVED (decision0089). D16/float64 and shrinking distances remain descriptive, not topology or global information claims. 12 conditions replayed exactly, no GRU/probe updates, Gate1 remains NOT_PASSED.

2026-09-28 ITER-0013 / EXP-0018 v1 completed frozen CO binary linear probes: 6 parent checkpoints, 12 converged probes, 24 Newton updates, 0 GRU updates. INCONCLUSIVE / NO_USEFUL_LINEAR_READOUT (decision0088): D64 IID/OOD test accuracy50.0326%/49.4222%, similar to shuffled-fit-label controls. Independent single-process replay passed in4.17s. A negative fixed-class probe is not absence of information; no architecture or causal specialization result. Gate1 remains NOT_PASSED.

2026-09-28 ITER-0012 / EXP-0017 v2 completed ORDER-OBS control qualification: 24 models, 18000 updates, 75.19s run+analysis. INCONCLUSIVE / LEARNING_FAILURE (decision0087): CO/CU/NO fail D64 IID learning; NU learns but mean OOD KL=.01967336 fails .02 headroom (seed CI crosses threshold). Close this fixed 2x2/750-update effect plan, no A1/A2 training. Supplemental full checkpoint replay validates numerics after an explicitly retained floating-point readout-shape bug; cumulative replay timing is UNVERIFIABLE. Gate1 NOT_PASSED, no new mechanism result.

2026-09-26 ITER-0011 / EXP-0016 v1 completed actual staged control training: 12 models,3 seeds,36000 research updates,92.31s. Local verdict INCONCLUSIVE / CONTROL_CEILING (decision0086): B/U/D pass IID learning but fail frozen OOD headroom windows; H64 GRU mean OOD KL=.01698163. Close this fixed ORBIT task configuration; candidate A was not trained, no general architecture falsification and no Gate1 passage. Independent checkpoint/oracle/score recomputation matched within2.22e−15. EXP0015's task-information result remains valid.

2026-09-26 AutoResearch ITER-0010 / EXP-0015 v1 completed a registered **zero-training task-information screen**, with local verdict SUPPORTED / TASK_INFORMATION_ONLY (decision0085). All three frozen information gates passed; suffix8 KL mean0.06012534,95% CI[0.05590587,0.06450655], lower bound>.02. Independent matrix recomputation matched within1.58e−14. This is neither actual-control headroom nor model learning/organization evidence: ORBIT model research stays HOLD and Gate1 stays NOT_PASSED. Code/ledger entry-policy v2 synchronization only implements the already-effective0080.

2026-09-26 design-only continuation: decision0083 corrects the DIR-0006 graph-audit scope; DIR-0007 ORBIT-SOFT is a concrete, unregistered soft-organization-prior design with algebra/conditional-power checks only. No task samples or training occurred. Its local estimand, project eligibility and empirical qualification are distinct; Gate 1 remains NOT_PASSED. See `research/loop/current_frontier.md` for the current snapshot rather than earlier dated navigation.

EXP-0013 v3 / ITER-0009 completed the control-only qualification and is `INCONCLUSIVE / LEARNING_FAILURE`: analytic task-information and delayed-evidence/shortcut guards passed, but SC64 and D64_full failed the pre-registered IID32 learning gate at both learning rates after all 28,800 control updates. Independent raw-array/source review accepted the result; RS4 received zero task updates. The conditional EXP-0014 formal comparison was never run. Gate 1 remains `NOT_PASSED`; see `research/iterations/ITER-0009/review.md` and `research/negative_results/exp0013_control_learning_failure.md`.

Decision0043 prospectively permits a **bounded architectural** Gate 1 test of explicit independent recurrent organization against resource-matched sparse and strong dense single-GRU controls. A strong control's ability to internally form slots does not disqualify it; observed benefits identify finite-budget organization/connectivity bias, not pure K count or Gate 3 hypothesis semantics. The decision does not admit old candidates or change the current `NOT_PASSED` state; see `research/evidence/STR-0004_gate1_architecture_estimand.md` and `docs/gate1_k_factor_qualification_template.md`.

Post-task reflection added decision0044: independent pre-run review must explicitly establish that matched and dense single-stream controls retain input/output capacity, fair training and meaningful IID learning; weakened or unresolved controls stop a project effect run. G1-RS v0 remains closed before training.

EXP-0012's frozen four-angle, zero-sample matrix screen ended `NO_MATRIX` (all D32 values above its registered ceiling; no H/cue/u/y samples or models). Independent recomputation and packet-only reflection passed. This is a task-parameter qualification failure, not a multi-trajectory effect; Gate 1 remains `NOT_PASSED`. See `research/decisions/0047_exp0012_no_matrix.md`.

EXP-0011 v6 / ITER-0008 stopped at the registered resource gate: K4_score/S68_sparse p50 latency ratios were 1.4811–1.7629 in all nine timing comparisons against the 1.12 failure line. Engineering preflight was valid, but no task information or research models ran; the field-reference result is `INCONCLUSIVE / RESOURCE_MISMATCH` and cannot advance project Gate 1. See `research/iterations/ITER-0008/decision.md` and `research/negative_results/exp0011_v6_resource_mismatch.md`.

Prospective FR-0002 eligibility audit decision0041 is `FIELD_ONLY`: the scored K4 versus unscored single GRU compares a PF-style package, not an isolated recurrent K-update factor. It does not revise EXP-0011 or Gate 1, and project-motivated v7 score optimization is stopped. See `docs/FR-0002_gate1_eligibility_review.md`.

EXP-0009 v1 / ITER-0007 completed as **INCONCLUSIVE** for the registered >0.05-nat block advantage because its d=4 matched control lacked the preregistered oracle-relative headroom. All ten paired seeds and thirty model records completed; independent science review and seed9401 clean-checkout reproduction passed. Gate 1 remains closed. Earlier EXP-0008 was INCONCLUSIVE; EXP-0004/0006 scoped predictions remain FALSIFIED, and EXP-0007's fixed-ridge transfer direction remains PARTIALLY_SUPPORTED.

| Gate | Requirement | State |
|---|---|---|
| 0 | Locked environment, registry, runner, tests, resource accounts, prior art | PASSED; 177 tests + all-model overfit at EXP-0011 engineering freeze; EXP-0004 frozen-commit seed9101 all5 models exactly reproduced |
| 1 | Generic trajectory benefit after matching the declared consumed axis (compute matched-or-profiled for SUPPORTED; all other axes disclosed as measured ratios) | NOT_PASSED; EXP-0004 primary prediction FALSIFIED; independently reviewed |
| 2 | Adaptation exceeds noise/compute controls | NOT_STARTED |
| 3 | Functional hypothesis-specific causal specialization | NOT_STARTED |
| 4 | SAE without relevance collapse | NOT_STARTED |
| 5 | Biological topology exceeds strong matched nulls | NOT_STARTED |
| 6 | Compact synthetic principles recover effect | NOT_STARTED |
| 7 | Transfer to controlled natural sequence tasks | NOT_STARTED |
| 8 | Independent LM | NOT_STARTED |
| 9 | Scaling after replication | NOT_STARTED |

No adaptation, communication, DARC, biological graph or LM is authorized by evidence before its prerequisite. Failure to establish Gate 1 keeps the program here; do not add mechanisms to rescue a negative comparison.

Earlier evidence remains valid for its original direct-training conditions. Current evidence is `research/evidence/auto_research_cycles.md`, both iteration reviews and `docs/EXP-0004_reproduction.md`. The two matched primary models learn IID8 perfectly but independent blocks lose at primary OOD32. This does not falsify every multi-trajectory architecture.

Benchmark/baseline audit: `research/evidence/gate1_benchmark_baseline_audit.md` separates the short-delay ceiling from the non-ceiling matched primary comparison, and records why this lookup task does not establish competing-hypothesis trajectories. EXP-0004's frozen decision and Gate 1 state are unchanged.

ITER-0006 post-task reflection: `research/reflections/ITER-0006/` records a fresh-context packet-only critique, grounded review, main-agent reaction and the compact ledger event. It found the minimal action packet omitted visibility into already stored run details and later review/reproduction status. The packet builder now supports selected action-output bundles; no EXP-0008 or Gate 1 scientific decision changed.

CAL-0001/0002 benchmark qualification: both synthetic Bayesian evidence tasks defeated one-cue shortcuts but a dense H64 GRU reached their exact posteriors under the fixed calibration budgets. Neither was promoted to an architecture comparison. `research/decisions/0002_gate1_benchmark_strategy.md` stops further task-variable inflation pending a justified computational bottleneck. CAL-0002 received a separate fresh-context post-task reflection in `research/reflections/CAL-0002/`; Gate 1 is unchanged.

CAL-0003/0004 training-condition calibration: fixed d=1 training failed badly at the first predeclared d=4 nuisance-duration transfer, with all three seeds below qualification; the full CAL-0003 run was exactly reproduced in a clean checkout after a test-import repair. An equal-token, equal-update paired comparison then found varied d=0/1/2 training yields d=4 final joint KL 0.018–0.023 versus fixed d=1's 1.65–4.05. `research/evidence/CAL-0004_cadence_qualification.md` and its independent pairing audit record the scope. This qualifies a **training distribution** for further benchmark design, not a multi-trajectory effect or Gate 1 passage.

CAL-0005 was a design-only audit of a proposed fixed d=2 control. Its precommitted constraint test found the desired contrast cannot simultaneously match examples, updates, tokens and forward-linear proxy while preserving a variable schedule with maximum d=2. The independent post-task reflection corrected the proxy-percent error and narrowed the decision; no new model training occurred. `research/decisions/0018_cal0005_fixed_d2_contrast.md` records the exact NO-GO scope. Gate 1 remains NOT_PASSED.

STR-0001 audited whether project Gate 1 had become a self-blocking novelty gate. Independent pre-decision criticism reduced the initial `PROJECT_RULE_OVERCONSTRAINED` claim to `INCONCLUSIVE`: current decisions exclude pure prior-art replication, but do not classify a new controlled K-factor experiment inside a prior-art backbone. Packet-only reflection then found that the suggested field benchmark qualification had no concrete task, purpose or bounded budget; decision0020 marks it `NOT_REGISTERED`. No Gate 1 rule, experiment or downstream authorization changed.

CAL-0006 then registered a one-shot field-level benchmark screen with explicit task, oracle, controls, seeds, budget and failure rules, without claiming project Gate 1 eligibility. v1/v2 engineering omissions were prospectively corrected in v3 before outcomes. The formal six-arm run from `f170e9b` passed oracle information and IID learning guards but failed its all-seed OOD12 headroom condition: dense seed603 KL0.064968<0.08, while five other records were in-window. Independent science review and frozen-checkpoint/data-stream reproduction confirmed the narrow `CEILING` result. Its fresh-context task reflection separated the deterministic all-seed rule from post-hoc evaluation-sample uncertainty and noted the revision slice KL0.09819 is not covered by the overall 0.08 threshold; `research/reflections/CAL-0006/` preserves the reaction. Decision0021 closes this fixed task configuration; Gate 1 remains NOT_PASSED and no new architecture phase begins.

STR-0002 reflected on the repeated task-first Gate 1 search after CAL-0006. Its independently corrected decision0022 applies `CANDIDATE_FIRST_FOR_GATE1` only to **future project Gate 1-motivated qualification training**: name an H3 candidate or reference K factor, falsifiable ablation and actual matched primary control before selecting task headroom. This preserves separate, bounded domain benchmark research and does not change any old CAL/EXP or Gate 1 `NOT_PASSED`.

Its packet-only reflection found five follow-through gaps: field benchmark decisions remain possible, reference K project eligibility is unresolved, document edits do not test the entry, action ranking needed bounded alternatives, and candidate-first could select favorable tasks. The implemented `gate1_candidate_first_v1` structural QUESTION gate now rejects missing-factor project entries, admits bounded independent/field entries, records state/parameter/compute plans and freezes task-search limits. Four dry-run cases,141 tests, and a cloned66-event live-ledger migration passed independent review; live event-000067 enabled the policy while preserving ITER-0007 DECIDED, reflection and prior events. `research/reflections/STR-0002/` contains the main-agent reaction; this is no scientific gate passage.

EXP-0009 compared independent recurrent blocks with a degree/self-loop/effective-parameter/compute-matched random-mask GRU under that varied schedule. Independent pre-run review corrected its >0.05-nat CI threshold and barred Gate 1 passage from this comparison alone. The preflight and run provenance checks passed, but C0 d=4 KL averaged only0.012463 against a0.07 headroom minimum; the registered M3 benefit was not discriminable. `research/evidence/exp0009_posterior_transfer.md` preserves all results and the protocol-limited decision. Its fresh-context post-task reflection, grounded main-agent reaction and compact ledger event are complete in `research/reflections/ITER-0007/`; no later phase gate changed.

Further block-only difficulty search is closed by `research/decisions/0003_exp0009_headroom_and_h3.md`. The next H3 design must satisfy `research/design/h3_candidate_trajectory_contract.md` and the newly audited PF-RNN/RIMs baseline boundary before any mechanism implementation or new architecture experiment. This is a research design requirement, not Gate 1 passage.

`research/decisions/0006_gate1_baseline_dependency.md` separates the unchanged Gate 1 generic matched-effect rule from any later claim of candidate-specific advantage over PF-GRU/RIMs. It orders zero-training mechanism definition, actual primary-control qualification and only the relevant baseline behavior preflight before a new formal comparison. A draft that made beating all strong prior-art baselines a Gate 1 condition was corrected after independent review; no past decision or gate status was revised.

The first zero-training candidate screen, SCR-0001 (`research/decisions/0007_h3_slot_no_spec.md`), gave an identity-conditioned weighted-slot model **NO_SPEC** under its prespecified design test: no distinct, numerically falsifiable task prediction or complete candidate contract. An independent critic accepted that narrow label while identifying a rare-hypothesis persistence counterfactual still lacking controls and thresholds. No model was implemented; Gate 1 remains NOT_PASSED and H3 is not generally falsified.

SCR-0002 followed the candidate-first rule without training or a new benchmark. Primary PF-RNN paper/code review and independent mathematical criticism found that low-weight recovery under unpruned additive log weights is shared with no-resampling weighted-particle and exact-filter variants. The author PF-GRU code still resamples; the paper's `NoResample` table is PF-LSTM, so no exact author-code equivalence was claimed. Packet-only reflection further showed index-weight recovery does not prove stable hypothesis semantics, split three distinct PF reference objects and pinned CAL-0001/0002's auxiliary near-oracle values; see decision0025. Current H3-SLOT remains `NO_DISTINCT_SPEC` for lack of a new quantitative task prediction or fair ablation; decision0024 does not falsify all H3 or change Gate 1 `NOT_PASSED`.

GOV-0002 audited only the currently named candidate/task/control combinations under the active entry policy. Its independently corrected trace matrix finds `NO_REGISTERABLE_PROJECT_GATE1_EXP_NOW`: the old matched M3 effect is negative, H3-SLOT remains underspecified, posterior/dynamic CAL configurations fail relevant main-control qualification or lack a candidate, and reference K project eligibility is undecided. This is a present portfolio/registration state, not a generalized architecture falsification or ban on separate field benchmark/memory questions. `research/loop/current_frontier.md` records objective re-entry triggers; decision0026 preserves Gate 1 `NOT_PASSED` and the ITER-0007 reflected DECIDED ledger.

GOV-0002's packet-only reflection clarified that a new syntactically complete H3 contract may enter `QUESTION` **before** control headroom is known; independent design review and a separately frozen finite control qualification follow, and only an actual qualified control plus full resources/ablation permits formal architecture-effect EXP registration. Four copied-ledger dry runs admitted synthetic project-design, independent-memory and field-reference questions only to QUESTION, while project reference K scoring remained unresolved. `research/decisions/0027_gov0002_stage_boundary_correction.md` and `research/reflections/GOV-0002/` preserve this scope without starting a new live EXP or changing Gate 1.

PA-0001 (`research/decisions/0008_fixed_slot_prior_art_boundary.md`) found direct prior-art overlap for the broad behavior of retaining alternatives and revising a dominant interpretation with later evidence, but only partial overlap for the exact same low-weight fixed recurrent slot surviving and recovering. The PF-RNN paper already reports a PF-LSTM NoResample ablation, not a PF-GRU no-resampling comparison on this task. This narrows novelty and control requirements, not Gate 1 status or H3-SLOT's NO_SPEC.

Its packet-only post-task reflection (`research/reflections/PA-0001/`) required a two-claim verdict and exposed incomplete initial coverage of no-resampling-method and author-code sources. A post-task search log records the deviation and adds the 2026 Unified PF-LSTM preprint plus the author-linked Belief Net repository without treating either paper's outcomes as reproduced. The remaining fixed-identity question has no generator, numeric effect threshold or matched-control headroom; it is not an EXP registration. Gate 1 remains NOT_PASSED.

MON-0001 (`research/decisions/0009_exp0009_monitor_strategy.md`) reanalyzed only EXP-0009's already stored d=1 monitors to test a possible sample-efficiency reinterpretation. The matched C0 had mean KL0.077642 at300 updates but M3's paired direction crossed zero (6/10 positive); by900 updates C0 was below the prespecified0.01 ceiling line and the M3 direction was negative. The strategy label is UNINFORMATIVE_CEILING, with the separate factual finding of no consistent early signal. No new training or Gate 1 revision followed.

Decision `research/decisions/0010_pf_preflight_branch_stop.md` closes only the strategy of using an early PF-GRU engineering smoke test to rescue current H3-SLOT/old M3 without a discriminating candidate and qualified control. An independent critic found it low information and contrary to the existing dependency order, though not categorically forbidden by Stage Gates. No new PF code or training was started; Gate 1 remains NOT_PASSED.

Its packet-only post-task reflection led to `research/decisions/0011_pf_engineering_scope_correction.md`: a truly separate PF software fault can be debugged without changing Gate 1 research dependencies, while promoting such a test into early research preflight requires a versioned protocol change. Current repo has no PF implementation or concrete PF fault. The decision records five observable candidate-contract requirements for reopening H3 design review; none have been met. Gate 1 remains NOT_PASSED.

GOV-0001 (`research/decisions/0012_reference_effect_vs_gate1.md`) audits the original charter against current project rules. A future local, registered and resource-matched PF/RIM replication could support the **field-level generic multi-trajectory question**, but under present decision0006 a pure prior-art effect cannot by itself pass **NeuroTopo's project Gate 1**. Treating the former as the latter would require a prospective versioned rule change; no experiment, PF implementation or gate advancement is authorized by this interpretation. The current PF smoke-test STOP remains limited and in force.

Its post-task reflection produced `research/decisions/0013_gate1_causal_estimand_scope.md`: a whole PF-GRU versus GRU performance gap is an architecture-package effect, not an isolated causal effect of trajectory count. Any future field-level study must predefine the manipulated factor and matched ablations, or report only the package effect; no task screening or PF implementation has begun. Gate 1 remains NOT_PASSED.

DIR-0001 (`research/decisions/0014_h1_range_no_go_now.md`) independently reviewed whether a wider graph-only reciprocity search should start after the local 10/10 witness. It found no present decision that such a search could change under the current H1/Gate 1 constraints; no graph search or training was run. This is a sequencing NO_GO_NOW, not a falsification of H1 or graph-space reachability.

DIR-0001's packet-only reflection corrected an overbroad implication that every H1 method audit was prohibited (`research/decisions/0015_h1_method_scope_correction.md`). The preregistered no-new-graph BND-0001 count-bound audit (`research/decisions/0016_h1_count_bound_nondecisive.md`) found 0/10 graphs where a loose necessary bound excluded a +0.10 reciprocity target, and stopped as `BOUND_NONDECISIVE`; it did not run graph search or advance H1/H2/Gate 1.

TH-0001 (`research/decisions/0017_gate1_capacity_argument.md`) shows the existing CAL-0001/0002/EXP-0009 posterior tasks admit 4/8-log-weight exact filters, far smaller than H64. Under explicit smooth full-posterior assumptions, the same state-dimension lower bound applies to single and grouped H64 states; increasing K alone gives no multitrajectory capacity argument. Independent math review also identified a shared, non-feedback pre-query output cache omitted from the old recurrent-state-only resource account; future `resource_account` now reports it separately, while frozen EXP artifacts and Gate 1 status remain unchanged.

The conditional H3 identifiability NO-GO (`research/design/h3_identifiability_audit.md`, independently reviewed in `docs/h3_identifiability_predecision_review.md`) applies only to implementing the current underspecified candidate model; it does not falsify H3 generally. A read-only H1/H2 graph audit found exact `Q=0.75−crossEdges/1024` coupling and other co-varying graph properties; a graph-only reciprocal-edge witness exists for10/10 C0 seeds but is not a null model. `docs/h1_h2_evidence_independent_review.md` independently replayed all ten witnesses after an environment-path repair. `research/decisions/0004_h1_h2_evidence_scope.md` keeps H1/H2 UNTESTED and authorizes no new model run. Gate 1 remains NOT_PASSED.

ITER-0003 diagnostic: blanking distractor payload reduced both block and random-model performance, and the random control lost26.21pp against a registered maximum allowed loss of3pp. The primary gap interaction CI crosses zero. This does not revise EXP-0004's FALSIFIED scoped prediction or permit mechanism escalation. See `research/evidence/exp0005_payload_diagnostic.md`.

ITER-0004 evidence: `research/evidence/exp0006_uncued_linear_access.md`. A later distinct observational question may examine whether a linear decoder fitted within the trained delay range transfers to longer full-noise sequences, but it cannot alter EXP-0004/0006 decisions or advance Gate1 by itself.

ITER-0005 evidence: `research/evidence/exp0007_cross_delay_transfer.md`. The transfer observation concerns a newly fitted linear readout, not a causal explanation of the pretrained task head. EXP-0004/0006 prior decisions are unchanged.

ITER-0006 evidence: `research/evidence/exp0008_final_revise_access.md`. The final-state probe uses newly labelled data and does not identify a causal failure of the pretrained shared task head. The registered branch is STOP_MECHANISM_ESCALATION.

DIR-0002 (`research/decisions/0054_dir0002_direction_portfolio.md`) audited the current direction portfolio read-only under the active entry policy and decisions 0043/0049/0053. Its corrected label is `RULE_CHANGE_FIRST`: no direction is registerable before two prospective rulings — whether a structure-mismatch (cross-coupled) condition counts as "another task family" under decision0049, and whether Gate 1 permits a learning-efficiency endpoint (updates-to-ε / curve area) as primary. It also records that TH-0001 blocks manufacturing a capacity separation by making the task harder at matched total state, and that the targeted prior-art search could not run (retrieval unavailable), leaving D3/D4 unassessable. No experiment was registered, no gate state changed; Gate 1 remains `NOT_PASSED`.

**Gate 1 匹配口径：判据版本 `G1-MATCH-v3`（决定0069，2026-09-24 生效；取代 `G1-MATCH-v2`/决定0068）。** **单层：无通用/严层之分，无自动降级，无数值上限。**

- **必须匹配的只有一条轴**：机制实际消耗的那条轴，注册时声明，并经 **pilot 实测的超出消耗 argmax 验证**。
- **`state` 匹配不再是必须的**（此为对 v2 的实质放宽）：`state` 与其余各轴（分配/有效参数、样本、更新、内部步数、时延）一律在 manifest 中按**实测比值披露**，匹配与否均须披露。
- **取消数值上限与超限自动降级**：v2 的默认 `1.15×` 上限与降级机制作废，代之以披露。
- **`compute`**：给 `SUPPORTED` 判词须 **matched-or-operator-profiled**（保留 v2 的收紧，不退回"仅披露"）。
- **时延**：受计费报告指标；**除非机制把时延声明为消耗轴，否则不单独阻断 Gate 1 判词**。
- **判词仅就所声明的轴条件化成立**；`state` 未匹配的比较**不得**被读作分离出机制的组织结构。
- 整包比较仍走带**资源未匹配标志**的 `PACKAGE_EFFECT` 轨，**禁止聚合或作为 Gate 1 证据引用**。
- 决定0067 §5 的**日落条款继续有效**。

**变更日志（决定0069）**：**本次为净放宽**，五项动作——(1) `state` 由"必须匹配"改为"披露"；(2) 取消各轴数值上限（默认 `1.15×`）；(3) 取消超限自动降级；(4) 两层并为单层，严层不可达时的"不可测声明"义务随之作废；(5) 时延不再单独阻断判词。**同时保留**"声明消耗轴须匹配（经 pilot argmax 验证）"与"`SUPPORTED` 须 matched-or-profiled compute"。Gate 1 要求文本已**同批**改为 `Generic trajectory benefit after matching the declared consumed axis`，以消除 0067/0068 期间"行文点名 `state` 而判据不要求 `state`"的**同一类不一致**。

**代价已记录（不得隐去）**：`state` 未匹配的比较**不能再被读作**分离出机制的组织结构；v3 下判词必须条件化；**v3 与 v2 科学上不等价，v3 更弱**。本决定**不是**"澄清"或"等价改写"。

**生效依据与审查状态（必须与判据本身分开读）**：本判据由**用户明确直接授权**生效（"别审查了，我允许你放宽"），**独立前瞻审查被豁免、未执行**——已冻结送审 packet（`research/reflections/DECISION-0069/packet.json`）但**无** `review.json`/`reaction.json`，**未运行** `reflection check`。故本判据是**经用户授权的规则变更**，而非**经独立审查确认的规则变更**；不得被引述为"已通过审查"。

**追溯性**：0068 之后**零注册**（`HEAD == 427f7f5`，其后零提交，`experiments/` 与 `research/entries/` 无变更）→ **无需追溯改判**。早于 v3 的、以 Gate 1 为目的的注册按 v1/v2 判据裁决或正式重新注册，**不得**按 v3 追溯。**非回归清单**（候选级 9 条 + 任务/组合级 3 条）**全部保持被否、无一重开**；放宽**不构成**重开理由，重开须走新的注册。每个注册须存储 `matching_criterion_version`。**决定0068 保留在库**并已标注为被本决定取代（其全文见 `research/decisions/0068_gate1_matching_clause.md`）。Gate 1 状态仍 `NOT_PASSED`，无 EXP 注册，既有判词全部不变。
