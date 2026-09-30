# Registry

2026-09-28：EXP0020v1为未运行的手写算子草案，已在实现/采样前撤回直接执行定位；不是失败研究结果。随后EXP0020v2已注册并完成三模型可学习性pilot；活动合同为`research/entries/EXP-0020-v2.json`，单seed结果不作确认性方法排序。

| ID | Version | Question | Status / decision |
|---|---|---|---|
| EXP-0001 | v1 | Matched block vs random recurrence | SUPERSEDED_BEFORE_RUN: diagonal confound |
| EXP-0001 | v2 | Same question, self-feedback matched | COMPLETED / INCONCLUSIVE |
| EXP-0002 | v1 | Longer-budget dense GRU learnability | COMPLETED / PARTIALLY_SUPPORTED (delay0 only) |
| EXP-0003 | v1 | Payload x curriculum diagnostic | SUPERSEDED_BEFORE_RUN: pre-run review corrections |
| EXP-0003 | v2 | Same diagnostic, corrected evidence/branch rules | COMPLETED / SUPPORTED (baseline curriculum only) |
| EXP-0004 | v1 | Matched architecture long-delay transfer | COMPLETED / FALSIFIED (scoped >=3pp primary gain) |
| EXP-0005 | v1 | Frozen-model payload sensitivity | SUPERSEDED_BEFORE_RUN: decision ambiguity and validity guards |
| EXP-0005 | v2 | Same diagnostic with mutually exclusive rules | REGISTERED |
| EXP-0005 | v1 | Frozen-checkpoint payload diagnostic | SUPERSEDED_BEFORE_RUN: ambiguous decision and validity guards |
| EXP-0005 | v2 | Same diagnostic with mutually exclusive decision rules | COMPLETED / INCONCLUSIVE |
| EXP-0006 | v1 | Frozen-state linear probe | SUPERSEDED_BEFORE_RUN: corrected CI interpretation |
| EXP-0006 | v2 | Uncued-candidate linear accessibility | COMPLETED / FALSIFIED (scoped >=5pp linear-access deficit) |
| EXP-0007 | v1 | Frozen readout cross-delay transfer | SUPERSEDED_BEFORE_RUN: added H8B source refit control |
| EXP-0007 | v2 | Source/target refit-controlled transfer | COMPLETED / PARTIALLY_SUPPORTED (direction only) |
| EXP-0008 | v1 | Frozen final-state revise-answer access | COMPLETED / INCONCLUSIVE |
| EXP-0009 | v1 | Matched recurrent organization on hidden-source posterior transfer | COMPLETED / INCONCLUSIVE (primary-control headroom failed) |
| EXP-0010 | v1 | Two-module soft-update temperature on old retrieval task | CLOSED_BEFORE_RUN / independent draft had no Gate 1 decision value; no training |
| EXP-0011 | v6 | PF-style four-candidate package vs resource-conservative single GRU on fixed 8D mode dynamics | RESOURCE_STOP / INCONCLUSIVE (`RESOURCE_MISMATCH`), 0 research updates; field_reference, no Gate 1 credit |
| EXP-0012 | matrix screen v3 | Fixed four-stream linear-Gaussian delayed-evidence candidate generator | CLOSED_BEFORE_TASK_DATA / `NO_MATRIX` (4/4 analytic candidates failed); no model or Gate 1 result |
| EXP-0013 | v3 | Fixed phase-reveal task/history/shortcut and SC64/D64_full control qualification | ITER-0009 COMPLETED / INCONCLUSIVE (`LEARNING_FAILURE`): task information passed, both controls failed IID32, RS4 0 updates |
| EXP-0014 | v3 | Same frozen task, RS4 vs matched SC64 and strong D64_full formal Gate 1 effect | CLOSED_BEFORE_FORMAL (决定0070): conditional qualification failed on EXP-0013 `LEARNING_FAILURE`; formal/replication roots never read, no Gate 1 effect |
| EXP-0015 | v1 | Fixed ORBIT-SOFT HMM full-history versus suffix information screen | ITER-0010 COMPLETED / SUPPORTED (`TASK_INFORMATION_ONLY`): 3/3 information gates pass, independently recomputed, 0 research updates; ORBIT model research HOLD and Gate1 unchanged |
| EXP-0016 | v1 | Learned HMM ceiling, otherwise B/U/D control-learning pilot on the fixed ORBIT task | ITER-0011 COMPLETED / INCONCLUSIVE (`CONTROL_CEILING`): 12 models/36000 updates; GRUs learn but insufficient headroom, fixed configuration closed; A not trained, Gate1 unchanged |

| EXP-0017 | v1 | ORDER-OBS four-cell control qualification | SUPERSEDED_BEFORE_RUN: explicit seed aggregation in v2; zero research samples/updates |
| EXP-0017 | v2 | Same fixed four-cell D16/D64 pilot; D64 gates | ITER-0012 COMPLETED / INCONCLUSIVE (`LEARNING_FAILURE`): 24 models/18000 updates; CO/CU/NO learning fail, NU headroom fail; fixed plan closed, no candidate training |

| EXP-0018 | v1 | Frozen CO hidden-state binary linear readout versus shuffled-fit-label control | ITER-0013 COMPLETED / INCONCLUSIVE (`NO_USEFUL_LINEAR_READOUT`): 12 converged probes, 24 Newton updates, 0 GRU updates; independent replay passed |

| EXP-0019 | v1 | Frozen CO paired-history dynamics | SUPERSEDED_BEFORE_RUN: batch position/size numerical-path alignment in v2 |
| EXP-0019 | v2 | Exact state merger under shared suffix, same-b history and float64 controls | ITER-0014 COMPLETED / INCONCLUSIVE (`NO_NEW_EXACT_MERGER_OBSERVED`): main D64 0/1152 mergers; D16/float64 descriptive, no updates |

| EXP-0020 | v1 | Hand-written protected operator contract | WITHDRAWN_BEFORE_IMPLEMENTATION: paper reframed toward learning, no data/run |
| EXP-0020 | v2 | One-seed learned RAW/INV/REL memory transitions with shared supervision and fixed curriculum | ITER-0015 COMPLETED / SUPPORTED (`SINGLE_SEED_LEARNABILITY_ONLY`): all three learn IID, INV/REL retain strong-rotation performance; no method-superiority claim |
| EXP-0027 | v2 | Exposed-state computation compatibility under reset | COMPLETED / SUPPORTED (`EXPOSED_STATE_COMPUTATION_INCOMPATIBILITY`): controlled next-command witnesses; 0 training updates |
| EXP-0028 | v1 | ARCH-0001 A1 implementation, engineering checks and learnable/non-ceiling task qualification | COMPLETED / INCONCLUSIVE (`NO_QUALIFIED_CONDITION`): 22000 updates; engineering 8/8 and clean-copy replay exact; no menu condition met the frozen non-ceiling band; A1 not passed |
| EXP-0029 | v1 | ARCH-0001 learning-qualification replication over 12 fresh seeds | COMPLETED / PARTIALLY_SUPPORTED (`LEARNING_QUALIFICATION_AT_BUDGET`): 36000 updates, 36 fits; CAND/EXTRA_SUP/INV_COMP all 12/12; combined with EXP0028 CAND 14/15, INV 15/15; A1 still not passed |
| EXP-0030 | v1 | DELAY-d command-memory task family and control-side qualification | COMPLETED / SUPPORTED (`A2_TASK_QUALIFIED`): 30000 updates, 15 fits; d=2 control 0.6916 scan / 0.6970 confirm vs oracle 1.0 and memoryless 0.54; d=1 ceiling; no candidate run |
| EXP-0031 | v1 | Corrected-endpoint re-qualification of the DELAY-d task family | COMPLETED / INCONCLUSIVE (`NO_QUALIFIED_MEMORY_CONDITION`): 24000 updates, 12 fits, 12/12 parent-hash reproduction; meaningful accuracy d=1 1.0000, d=2 0.5991, d=3 0.6109 vs band 0.65; supersedes EXP0030 qualification |
| EXP-0032 | v2 | Control-menu and budget sweep on the corrected DELAY-d endpoint | v2 COMPLETED / INCONCLUSIVE (`QUALIFICATION_NOT_REPRODUCED`); v1 INVALIDATED by an implementation defect (menu not injected). d=2 learnable by several configs (w32b2 exact 1.0000); matched-rule pick w48b2 5-seed mean 0.6455 < 0.65 |
| EXP-0033 | v1 | DELAY-2 carrier confirmation and exploratory SC-REL-v4 probe | COMPLETED / SUPPORTED (`EXPLORATORY_ARCHITECTURE_ADVANTAGE`): carrier w96b2; cand 0.7848 vs control 0.6723 vs EXTRA 0.7056, +11.26pp (95% CI [+0.35,+23.69]); control compute 1.82x candidate -> not A2 |
| EXP-0034 | v1 | Independent matched-compute replication (ARCH-MATCH-v1 carrier) | COMPLETED / FALSIFIED (`CONTROL_BETTER`): cand 0.6637 vs control (w48b2, ops ratio 0.912) 0.7261 vs EXTRA 0.6717, effect -6.24pp (95% CI [-15.04,+4.10]); closes ARCH-0001 in tested scope |

## EXP-0021 v1（完成）

四新seed RAW/INV/REL重复与t32冻结状态干预；12000更新；INCONCLUSIVE，预注册R/M均false。INV稳定成功、REL长程不稳定、即时编码交叉未重复。协议`EXP-0021/v1.md`，配置`configs/exp0021_v1.json`，原始数据`artifacts/EXP-0021/v1`，决定0092。

## EXP-0022 v2（完成）

当前等logit与等幅状态压力测试，15冻结模型×3半径，0研究更新。v1可达性问题在实现前修订并保留；v2无INV主见证，INCONCLUSIVE。REL20004局部效应保留为次要观察。见决定0093、EXP0022报告与原始artifacts/EXP-0022/v2。

## EXP-0023 v2（完成）

HOLD/FLIP输出组合一致性诊断与最小训练修复，六新seed×4臂、24000更新。INCONCLUSIVE / LEARNING_GUARD_FAILURE；全部种子与强INV保留，R/M未支持。v1零漂移floor计数问题在实现前修订，无v1运行。协议EXP-0023/v2.md；报告EXP-0023_composition_repair.md；决定0094。

## EXP0024v1（完成）

监督原型状态重整；12冻结模型、192评估、12原型拟合、0神经更新。SUPPORTED仅指预设REL23005病例，非群体效果/新算法。其他模型和损伤保留。见决定0095与EXP-0024_state_refresh报告。

## EXP0025v2（完成）

2026-09-28 ITER0020 / EXP0025v2完成12新seed×REL/INV及2旧参考。R_NOT_SUPPORTED（FULL8修复1/5），M_NOT_SUPPORTED（EQUAL8有效子组1/2），总体INCONCLUSIVE。新25002固定logit重整75%→100%，仅幅度调节仍75%，保留局部机制见证；其余种子损伤与不可行回退完整报告。报告EXP-0025_refresh_replication.md，决定0096；Gate1不变。 v1因M的GAUGE资格缺口在实现前修订，未运行。24新训练模型、26原型拟合、624评估、独立完整复演通过。

## EXP0026v2（完成）

2026-09-29 ITER0021 / EXP0026v2完成1196主轨迹+104独立τ轨迹。PARTIALLY_SUPPORTED / SINGLE_EXPOSED_SCHEDULE_WINDOW：主条件失败25002/25003/25008中仅25002通过完整窗口；k2/4/8/16 H/U在256/512/1024均100%，k32/64 HOLD50%，R/I同窗口。25003 k2/4恢复HOLD但长期UPDATE未达标。50%经逐类核查源于单类读出收敛；原始审计1300/1300、预定复演200/200精确。见EXP-0026_schedule_drift报告与决定0099，Gate1不变。 v1运行前判词歧义在实现前明确为v2，未运行v1。研究训练0、原型拟合0，三个视界同轨迹。

## EXP0027v2（完成，手动执行）

2026-09-29 EXP0027v2已完成（直接手动执行，旧自动ResearchLoop仍止于EXP0026）。主线转为重整后的计算兼容性：REL25003有当前logit完全保持、GAUGE通过而下一FLIP出错的受控反例；原UPDATE中四FLIP也有反例。FULL原型k2/k4区块全通过，但25003/25008初始化并非全通过，EQUAL保留连续尺度。次要k3使25003 H/U在已测视界均过95%，说明旧六个二次幂间隔结果不能当单调规律。328测试、104短反例0误差重放，结果SUPPORTED仅限具体暴露状态，见EXP-0027/report.md与决定0100。

## EXP0028v1（完成，ARCH-GATES-v1 A1，手动执行）

2026-09-29 EXP0028v1 完成 ARCH-0001 A1 实现与任务资格：`SC-REL-v3` 与 `INV-COMP` 实现后通过工程8项检查（含共同旋转规范、两步梯度规则、单递归状态、前缀一致性、`σ=0` 与 EXP0020 生成器逐位一致、tiny-fit）及独立干净副本重训重放（3运行参数/数据/预测哈希与指标差0）。IID8 前测 `INV_COMP` 3/3 seed、`CAND` 2/3 seed（seed28103 未学会翻转）。控制侧8个预注册条件（长度64/256×leak0/.4×σ0/.05/.15/.4）无一致命合格：σ≤.15 全为100%或近封顶（≥.9902），σ=.4 时 N5 翻转均值 .6218 在带内但保持均值 .5989<.60，N8 随机水平；简单 Gram 重编码参照在 σ≤.15 全时点100%。判词 **INCONCLUSIVE / NO_QUALIFIED_CONDITION**，A1 未通过，A2–A6 未评估，旧 Gate1 不变。协议 `experiments/EXP-0028/v1.md`、配置 `configs/exp0028_v1.json`、产物 `artifacts/EXP-0028/v1`、报告 `research/evidence/EXP-0028/report.md`、决定0103；22000 研究更新、运行约120秒、manifest 记录 `git_dirty=true`（沙箱不可写活动 git 元数据，未能提交）。

## EXP0029v1（完成，ARCH-GATES-v1 A1 学习资格，手动执行）

2026-09-29 EXP0029v1 用 12 个新 seed（28201–28212）复现 `SC-REL-v3`／`EXTRA_SUP`／`INV_COMP(w32,b1)` 在冻结 IID8 课程/预算下的学习资格：三臂均 **12/12** 通过 EXP0020 学习旗标（36000 更新、36 次拟合、217.7 秒）；合并 EXP0028 的 3 个 seed 后 `CAND` 14/15（唯一失败 28103 为 `HOLD_COLLAPSE`）、`INV_COMP` 15/15，独立干净副本重放 3 个归档运行哈希与指标差 0。判词 **PARTIALLY_SUPPORTED / LEARNING_QUALIFICATION_AT_BUDGET**；A1 门仍因缺少非封顶合格任务而未通过，A2–A6 未评估，旧 Gate1 不变。协议 `experiments/EXP-0029/v1.md`、配置 `configs/exp0029_v1.json`、产物 `artifacts/EXP-0029/v1`、报告 `research/evidence/EXP-0029/report.md`、决定0104。运行清单记录 `git_dirty=true`（沙箱不可写活动 git 元数据，未能提交）。

## EXP0030v1（完成，ARCH-GATES-v1 A2 任务资格，手动执行）

2026-09-29 EXP0030v1 注册并校准 `DELAY-d` 命令记忆任务族：读出头须输出 `u_{r−1−d}`；命令 i.i.d. Bernoulli(0.5)；无记忆直通参照≈0.54、显式移位寄存器 oracle=1.0。6 配置（d∈{1,2,3}×监督{PLAIN,BRANCH}）×2 扫描 seed、每条件 2000 次更新（共 30000 更新、15 次拟合）：d=1 两变体均 1.0000（封顶）；**d=2（PLAIN）0.6916**、d=3 0.7306 均满足可学非封顶。按预注册顺序取 `D2_PLAIN`，独立 3 seed 确认 0.6922/0.6979/0.7009（均值 0.6970），干净副本重放 2 运行哈希与指标差 0。判词 **SUPPORTED / A2_TASK_QUALIFIED**（限任务/预算/种子）：只补上决定0103 第 2 条的控制侧缺口，**无候选训练、无 A2 效果、A1 与旧 Gate1 不变**。**该判词已被 EXP0031 v1 的修正端点复核取代**（见下）。候选侧比较须先把反事实分支监督推广并命名 v4。协议 `experiments/EXP-0030/v1.md`、配置 `configs/exp0030_v1.json`、产物 `artifacts/EXP-0030/v1`、报告 `research/evidence/EXP-0030/report.md`、决定0105。

## EXP0031v1（完成，EXP0030 反思触发的修正端点复核，手动执行）

2026-09-29 EXP0031v1 复核 DELAY-d 任务资格：同任务、同训练、同种子重训 12 个运行，与 EXP0030 归档 `final_hash`/训练数据哈希 **12/12 逐位一致**；把主端点改为只统计真实命令位置的 `meaningful_accuracy`（`r∈[d+1,T]`），并把无记忆参照扩为常数零、常数一、直通、逐位置多数（独立校准流）。结果：d=1 封顶 1.0000；**d=2 0.6058/0.5924（均值 0.5991）**、**d=3 0.6162/0.6056（0.6109）**，均低于可学门槛 0.65；参照全部≈0.50、oracle=1.0 → 菜单内无合格条件，判词 **INCONCLUSIVE / NO_QUALIFIED_MEMORY_CONDITION**，取代 EXP0030 的 `SUPPORTED / A2_TASK_QUALIFIED`（决定0106）。逐位置曲线显示 d=2 控制只在前 1–2 个真实位置高于随机。干净副本重放 2 运行（含单列分支哈希）全部为 0 误差。协议 `experiments/EXP-0031/v1.md`、配置 `configs/exp0031_v1.json`、产物 `artifacts/EXP-0031/v1`、报告 `research/evidence/EXP-0031/report.md`；运行清单 `git_dirty=true`（沙箱不可写活动 git 元数据，未能提交）。

## EXP0032v2 / EXP0033v1 / EXP0034v1（完成，A2 通道收尾）

2026-09-29 三段收尾：EXP0032 v1 因实现缺陷（未把 width/blocks 注入 `fit_memory` 配置，12 个"配置"实为同一模型）作废，v2 重扫确认 `DELAY-2` 上 `w32b2` 精确解出、多项配置落于合格带；预注册匹配规则选中的 `w48b2` 5 seed 均值 0.6455 未达 0.65 → `INCONCLUSIVE / QUALIFICATION_NOT_REPRODUCED`（决定0107）。EXP0033 用 fresh seed 独立确认载体（`w96b2`，均值 0.6978）并做探索性对照：`SC-REL-v4` 0.7848 对控制 0.6723、效应 +11.26pp（95% CI [+0.35,+23.69]），但控制算子量是候选 1.82 倍，判 `SUPPORTED / EXPLORATORY_ARCHITECTURE_ADVANTAGE`（明确非 A2，决定0108）。EXP0034 用合同口径控制 `w48b2`（算子比 0.912）与全新 seed 复现：候选 0.6637 对控制 0.7261，效应 −6.24pp → `FALSIFIED / CONTROL_BETTER`（决定0109），**关闭 ARCH-0001 候选路线**（已测范围内无匹配计算优势；同监督无修正臂与候选持平）。全部记录与负结果保留。
