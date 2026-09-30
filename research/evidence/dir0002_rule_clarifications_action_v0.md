# DIR-0002 规则裁定草案：action 摘要

核对日期 2026-09-24；本任务为规则文本审议，不读取新模型结果、不训练、不注册实验。

## 当前权威状态

- 决定0054为 `RULE_CHANGE_FIRST`，明确留下两个须结果前裁定的问题：交叉耦合是决定0049所说的“另一个任务族”吗？Gate 1是否允许 learning-efficiency 作为主终点、是否需版本化？
- 决定0049将现行 G1-RS 阶段显露 generator 设为该方向最后一次参数化任务尝试，并保留“新的独立架构机制/研究问题通过事前批评后，可另提任务族”的路径。
- TH-0001已明示另行注册、任务有根据、资源匹配、主控制有余量的 K×model 学习效率问题逻辑上可测；这不支持用 K 增加容量论证。
- 决定0043将可过 Gate 1 的对象限于有状态／计算匹配的有限预算组织／连接偏置，不支持纯 K 数或候选语义。当前 Gate 1 仍 `NOT_PASSED`。

## 拟议的前瞻裁定

1. 任务对现已固定为：`z∈{-1,+1}^4`，无外场 Ising `p_J(z)=Z_J^-1 exp(Σ_(i,j)∈E J_ij z_i z_j)`，四环边E={(1,2),(2,3),(3,4),(4,1)}；F的J=0，X四边J=0.25；每步`y_t=z+N(0,I4)`，T=16。F/X是同一新 generator 的两个条件，只耦合矩阵不同；先验joint entropy差为处理内容。预写主效应是`D_{k,c}=mean_s[AUC_B(random_sparse)-AUC_B(block)]`，`β_c=(D_{4,c}-D_{2,c})/log(4/2)`，`I=β_F−β_X>0`。这相对EXP-0013的phase-reveal控制资格具有不同问题与终点，但须经独立评阅确认，不能自动绕过0049。若矩阵外改生成、试探其他矩阵、或资格失败后换generator，本解释失效并须新审。
2. Gate 1 可前瞻采用有限预算效率效应，但每个EXP只能有一个预先冻结的primary endpoint。当前KSI选`AUC_B=(1/B)Σ_jΔu_j(L_j+L_{j+1})/2`，KL按`log|Z|`归一；updates-to-ε只作次要，未达记`>B`与达标率，不插补/不替代主终点。实际主控制random-sparse(K=2/4)及dense H64 guard须在F/X各自独立资格流过共同ε与预写动态范围、难度和资源标准。任何资格失败不换主终点，不追认旧曲线；不作渐近／普遍compute结论。数字门槛和算法尚未冻结，当前不进入资格或正式轮。

KSI estimator detail: `L_j=mean_e[KL(p_e||q_{e,j})/log16]`, `u_j` are cumulative optimizer updates on the fixed B budget, `u_0=0`, `u_M=B`; `AUC_B=(1/B)Σ_j(u_{j+1}-u_j)(L_j+L_{j+1})/2`. `D_{k,c,s}=AUC_B(random_sparse)-AUC_B(block)`, `D_{k,c}=mean_s D_{k,c,s}`, `β_c=(D_{4,c}-D_{2,c})/log2`, `I=β_F-β_X`. One prelocked δ_min decision: support if lower95CI>δ_min; falsify minimum useful effect if upper95CI<δ_min; else inconclusive. Any control qualification cell failure stops before formal effect; no endpoint substitution.

## KSI终点与episode复算

每个episode先抽一个`z~p_J`；给定z，16个`y_t`条件独立且均服从`N(z,I4)`。模型看到整个`y_1…y_16`序列，预测16态联合posterior；训练使用oracle posterior soft-target cross-entropy `−Σ_z p(z|y)log q(z|y)`。 held-out stream独立于训练／资格调参；同一seed/cell的block与random sparse共用流。

定义 `L_j=mean_e[KL(p_oracle,e||q_model,e)/log16]`，`u_j`为累计optimizer updates，`u_0=0,u_M=B`；checkpoint网格及held-out episodes在资格数据揭示前冻结。唯一primary为 `AUC_B=(1/B)Σ_{j=0}^{M-1}(u_{j+1}-u_j)(L_j+L_{j+1})/2`。每种子 `D_{k,c,s}=AUC_B(random_sparse)-AUC_B(block)`，跨seed均值 `D_{k,c}`；`β_c=(D_{4,c}-D_{2,c})/log(4/2)`，`I=β_F-β_X`。预写三分支：95% CI下界>`δ_min`为支持；上界<`δ_min`为否定最小有用预测（另报可能的小正向）；CI跨`δ_min`为不确定。任何primary-control或dense-guard资格单元失败都停止正式效果轮。B、checkpoint网格、episode数、共同ε、δ_min、nonreach统计、种子、CI算法及资格容差尚未冻结，当前只能审规则／设计，不能执行资格运行。

## 方向审查依据与边界

- G1-KSI 草案尚未有具体 generator、oracle、数值门槛、预算 B、正式种子或数据流，仍不是注册协议。
- D1最接近的资源约束模块 RNN先例已改变任务变量covariance与模块结构并测module ablation；D4的matched-random组件消融与逐样本损伤也已有 attention-head closure先例。新方向不得作“首次发现模块化／匹配随机干预”的主张。固定提交、代码与有限来源差异见 `research/prior_art/dir0002_direction_prior_art.md`。
- G1-M3-SHAM-0001是零训练的旧 M3 方法草案；其十个权重无候选映射，不支持 H3 语义或 Gate 1 能力主张。

## 决策价值

若两项解释通过独立评阅，当前阻塞从“规则边界未裁定”转为“候选设计和数值资格未完成”；若任一不通过，则仅该项目 Gate 1 方向保持关闭／需要改走独立 benchmark。无论何种结果，既有实验标签、Gate 1状态和预算条件均不变。
