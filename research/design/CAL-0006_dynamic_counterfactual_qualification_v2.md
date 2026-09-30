# CAL-0006 v2：动态反事实假设任务的一次性资格筛查

v2 在任何本任务数据/训练结果之前取代 v1；独立预运行审查发现 v1 的随机流、配对与超时口径不够明确，见 `docs/CAL-0006_pre_run_review.md`。v1 不覆盖、不追认；以下未改的公式、问题、阈值和失败分支仍按原登记执行。

2026-09-24。**唯一问题：**一个需要跟踪四种候选动力学状态的延迟证据修订任务，在固定训练预算下是否让强单流 GRU 保有可学习、非封顶的 OOD 后验拟合余量？这仍是**基准资格**，不是 NeuroTopo 架构实验、H3 语义验证或项目 Gate 1。它测试任务中的“动态状态 → 后验表征 → 后验计算”是否足以避免 CAL-0001/0002 的近 oracle 天花板；没有测试生物拓扑。先例与局限见 `research/prior_art/cal0006_dynamic_filter_scope.md`，用户原始问题见 `research/design/gate1_user_charter_evidence.md`。最强拒绝理由：四条模式轨迹只需8个状态数加4个 log 权重，H64 单流有足够表示容量；任何效果可能只是优化或 OOD 长度而非多轨迹必要性。

## 任务和解析基线（在数据之前固定）

每例采静态隐藏模式 `H∈{0,1,2,3}`，均匀先验。每种模式各有二维**确定性**状态 `z_t^k`，`z_0^k=0`：

```text
theta = (-0.85, -0.30, +0.30, +0.85) radians
A_k = 0.9 * rotation(theta_k)
B = (1.0, 0.5), C = (1.0, -0.3), sigma = 0.8
u_t ~ Normal(0,1), z_t^k = A_k z_(t-1)^k + B u_t
y_t = C · z_t^H + Normal(0,sigma^2)
logw_t(k) = logw_(t-1)(k) - (y_t - C · z_t^k)^2/(2*sigma^2)
posterior_t = softmax_k(logw_t(k))
```

模型只见按时序输入的 `(u_t,y_t,OBS=1,QUERY=0)`，在前4条观测后见 `(0,0,0,1)` 早期查询，再见4条（训练/IID）或8条（唯一 OOD 主条件）观测及最终查询。目标为两个查询时的**完整四类解析后验**；loss 为早末等权 soft-target cross-entropy，主指标为最终 posterior KL-to-oracle（nat，越低越好）。固定模式矩阵**不作为输入**；参数由训练学习。oracle 以全部已见 `u,y` 和上述已知生成律更新。四条二维状态+四个 log 权重是显式充分状态上界；共同平移后权重可少一个，不声称最小维数。均匀预测的 KL 和 oracle 的 MAP 真值准确率只用于说明任务信息量，不作为模型余量替身。

训练/IID 共有8条观测，OOD 唯一主条件共有12条观测，早期查询始终在4条之后。任务 seed 派生相互独立且命名的 H、控制 u、观测噪声 RNG 子流，生成器不读取模型或全局 RNG。训练与 IID/OOD 流严格独立；评估同 seed 的8/12观测共享前8条 `u,y`，因此不会把难度差误认成更换了早期证据。分析须报告自然发生的早/末 oracle-MAP 修订频率、模型在该切片的 KL 和总 KL，但修订切片是次级描述，不改变主判据。不得按结果改 `sigma`、角度、长度、损失或主端点。

## 用途、模型、预算与一次性分支

独立用途是为**领域通用问题**筛掉一个可能仍可由单流轻易解出的动态基准；筛查不决定参考 K 因素实验能否计入 NeuroTopo 项目 Gate 1，也不自动许可 PF/RIM 训练。两个事前固定的强单流对照：`dense` H64 GRU 和同 H64/4组度数及自环约束的 `random` 稀疏 GRU，均用 `RecurrentModel`，共享每种子逐步训练 batch；后者是未来任何相同4组块候选最接近的现有主控制，**但尚不能代替未定义新候选的实际匹配控制**。保留稠密臂是防止只因稀疏控制偏弱而误称任务困难。每臂种子601–603，最多1200 updates、batch128、Adam0.003、梯度裁剪1、单线程 CPU，不早停也不择优；共最多6×1200更新、921,600训练样本。每种子在300/1200监测单独 IID 验证流；最终 eval 各条件8192例，评估种子与训练/验证互不重合。总墙钟上限900秒；超时仅标 `INCOMPLETE_BUDGET`，不可据部分结果更换规格。保留全部 checkpoint、数据/源码哈希、参数/状态/执行代理、每臂训练及推理时间；前向线性 FLOP代理不冒称完整训练 FLOPs。

任务信息护栏：解析 oracle 的 IID 最终真值 MAP 准确率须≥0.55，OOD12 早末 MAP 变化率须≥0.08；若不足，标 `TASK_UNINFORMATIVE` 并停止。模型学习护栏：两臂每一种子 IID 最终 KL<0.10 nat；任一不满足为 `LEARNING_FLOOR`。唯一 OOD 主余量窗口：两臂每一种子最终 KL∈[0.08,0.50] nat，其中 `δ=0.05` 与 `ε=0.03` 给出最低可判别余量0.08；低于0.08为 `CEILING`，高于0.50为 `OOD_FLOOR`。仅全部条件均满足时标 `SCREEN_PASSED_FOR_DESIGN_REVIEW`；这不构成正式对照资格、Gate 1 通过或后续模型运行许可。若混合失败，记录所有原始数值并按上述优先顺序给主失败标签；不从8/12之外找新的主长度。

实现前验证：生成器与独立暴力枚举似然 oracle 一致、固定 seed 字节确定、早期共享前缀/目标、隐藏 H 不出现在输入且观测噪声仅经 y 可见，观察切片正确；模型形状/梯度、32例 tiny overfit、loss/资源公式 sanity。正式执行须从提交的干净源码和冻结配置启动，保存每步/监控汇总与最终全结果；若验证失败先修复工程且保留偏离记录，不调科学阈值。执行后独立科学审查和 packet-only 反思，逐条 react。成功仍仅允许**另行**设计有明确 K 因素、匹配状态/参数/计算与实际主对照的研究；失败则关闭这一任务配置，不把设计失败升级为 H3 否证。

## v2 固定随机流、配对和执行边界

种子601/602/603为模型种子。每种子、每训练步 `step∈{1..1200}` 的数据根 seed 精确为 `6_000_000 + model_seed*10_000 + step`；两个模型在同一步使用**同一个已生成的 batch**。固定验证根 seed 为 `6_300_000 + model_seed`，各1000例、8条观测，用于300/1200监测，不选 checkpoint。最终 IID8/OOD12 共用评估根 seed `6_400_001`、各8192例，所有模型/种子共用该评估流。上述根 seed 互异。每次生成都以 `np.random.SeedSequence(root).spawn(3)` 依次得到 H、u、观测噪声子流；H 用 `integers(0,4,size=examples)`，u 与标准正态噪声各用**时间在前**的 `standard_normal((observations,examples))`，随后按公式乘 `sigma`。相同根 seed 的8/12条流必须逐例共享 H、前8条 u、噪声、y，且早期后验与前8条观测后的后验逐元素相同；记录前缀摘要。没有跨长度重抽样或按模型改变评估数据。

两臂由 `RecurrentModel(input_size=4, hidden=64, classes=4)` 构造；dense 为 `groups=1, topology="dense"`，random 为 `groups=4, topology="random"`，其图 seed 为 `6_500_000 + model_seed`。每臂构造前都调用 `configure(model_seed,1)`，但 random 与 dense 因循环度数不同而采用不同的 `weight_hh` 初始化刻度，**不能声称初始参数完全相同**；须记录各臂初值、mask 和有效/分配参数。两者总递推 H64、batch/update/token 与现有稠密执行实现的线性代理相同，但有效参数不同；本筛查不把它们当正式匹配架构因果对照。模型输出查询 token 后的 posterior logits，query 对解析 oracle 不更新。

900秒总墙钟从最终评估数据生成前开始，覆盖模型构造、训练、每次验证、最终评估、checkpoint 与报告写入；在任务护栏后和每一步更新/评估/保存阶段边界检查。若越界，立即以独立 `INCOMPLETE_BUDGET` 记录保存已完成臂/步数/耗时与错误，不作资格判定；已写完整 checkpoint 不删。输出为新的 `artifacts/calibration/CAL-0006_v2.json` 与 `artifacts/calibration/checkpoints/CAL-0006_v2/`，存在则报错。任务护栏先用冻结评估流计算，失败时不训练并记录全部护栏数值及 `TASK_UNINFORMATIVE`。运行环境固定 `/private/tmp/neurotopo-venv/bin/python`（锁版本见 `uv.lock`），单线程确定性 CPU；脚本、任务源码、配置和模型源码哈希入报告。工程预检可在独立微型 seed 上运行，但不得改变上面任何正式流或科学阈值。
