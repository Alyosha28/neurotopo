# EXP-0015 v1：ORBIT-SOFT 固定任务的信息筛查

2026-09-26。新的 AutoResearch 请求下登记的**零优化更新、单配置、仅任务信息**研究。不是 ORBIT-SOFT 效果实验或项目 Gate1 准入。此前 DIR-0007 的 HOLD 与0084结论保留；本轮以新问题/协议处理一个可直接停止该候选配置的经验缺口，不自动重新审查整份v1，也不启动控制或候选训练。

## 问题、用途、候选与最强拒绝

问题：DIR-0007 固定的16状态环形HMM，在其原样冻结的T/观测噪声/64步条件下，是否同时通过full-vs-uniform信息量及最后1/8帧捷径差阈值？

具名候选 ORBIT-SOFT：完整H16 GRU上对G循环群权重投影残差加固定软正则；实际拟议主控制B使用共轭G'同强度同谱正则，U/R/D为强保护。其函数类相同、1904参数相同；所有资源仍须未来实测。方向预测是先验对齐降低固定预算后的完整历史后验KL。本筛查不实例化这些模型、不读其分数，不把oracle差称为模型headroom。

本问题的因果链位置仅是 Computation→Capability 测量任务是否要求超越suffix8的历史整合，**不检验Topology→Dynamics或组织效果**。最强拒绝：这是无助项目过门的易跑旁支（0029），或绕过0084/0049。回应是明确的可改变决定：只要原信息门失败，**关闭ORBIT-SOFT这个固定任务配置，取消其144000资格更新及后续效果计划**；成功仅保留这个任务信息条件，不获得任何训练/项目资格；任务选择数严格1，失败后不改T/噪声/长度/阈值重跑。不重开G1-RS、EXP0013生成器或其他历史候选。

## 精确生成器（首次样本前冻结）

n=16384，N=16，L=64；z0∼Uniform{0..15}。列随机T=.7I+.2S+.1Sᵀ，S把状态j送到(j+1) mod16。
每步根据u：u<.7则不动，.7≤u<.9则+1，其余−1；z_t=(z_(t−1)+increment) mod16。
发射v<.6时y=z_t，否则j=floor((v−.6)/.4*15)，y=(z_t+1+j) mod16。若浮点取值产生j=15则工程中止，不裁剪到有效值。
NumPy2.3.3 PCG64；SeedSequence(77000007).spawn(3)依序用于z0、transition、emission；仅调用initial.integers(0,16,size=n)、transition.random((n,64))、emission.random((n,64))各一次。保存z0/z/y与数组hash，不保存为模型训练集。

q0=uniform；q_t=normalize(L(y_t)⊙Tq_(t−1))，L(y)_j=.6 if j=y else .4/15。float64。主实现T作用用局部shift加权，独立复核用显式16×16矩阵乘法；另用4状态短序列枚举隐藏路径检验算法，不使用研究随机根。所有后验有限、非负、和1误差≤1e−12；正式复核两实现全序列最终后验误差≤1e−10。

q_suffix1/q_suffix8均从对应后缀之前的uniform开始，仅过滤y64或y57:64。uniform是该固定T的平稳分布，因此这是仅见后缀的正确后验，不把完整前史偷偷带入捷径。
逐例存 a=KL(q64||uniform)、b=KL(q64||q_suffix1)、c=KL(q64||q_suffix8)；自然对数，不clip、fillna或改负数；低至−1e−12的数值舍入允许记录原值，超出则工程中止。

## 冻结判定与不确定性

继承DIR-0007原阈值：mean(a)≥.20nat；b的样本bootstrap双侧95% percentile区间下界≥.05nat；c同下界≥.02nat。10000 bootstrap，PCG64(77100007)，每次独立有放回抽n个索引，同一索引用于b/c，按先resample再row的C-order生成；np.quantile(method='linear')。按块生成索引不改变flatten后的随机调用顺序；保存10000×2个bootstrap均值及最终区间。

统计单位是独立生成的序列，此CI只表示**本固定生成器的采样不确定性**，不是训练seed/泛化到任务族的不确定性。mean(a)按原设计为点值门，附分布描述但不擅改为CI门。不利用测试样本挑任务；该批样本今后标为已暴露，不能再作独立效果评估。

一次运行须输出全部三个数、区间、失败项、每序列值、oracle后验与全部原数组；即使第一个阈值失败也完成其余信息量计算（预算内）。互斥决策：
- 工程、来源、预算或复核失败：INCONCLUSIVE / ENGINEERING_ABORT、INCOMPLETE_BUDGET或REPRODUCTION_UNRESOLVED；保留输出，不按科学失败改任务。
- 工程与复核通过、任一信息门失败：INCONCLUSIVE / TASK_INFORMATION_FAILURE；关闭这个固定配置，未测机制不判FALSIFIED。
- 全门通过：SUPPORTED / TASK_INFORMATION_ONLY；只支持这组信息要求，不放行控制训练、Gate1或后续门。项目因子资格、规则范围、控制学习/余量仍HOLD。
接近阈值仍按冻结下界判；不补样本。保留上界以区分明显不足与精度不足，但不改分支。

## 预算、来源、独立复核

工作流orbit_information，worker硬限300秒，0优化更新，1配置，1个研究根；包含生成、oracle、bootstrap、分析，不包含事前工程测试。工程限60秒、只用枚举/手写观测，不调用77000007/77100007。分析只读原数组，不能再次抽任务。
source/config/protocol/entry/uv.lock SHA、clean git commit、Python可执行文件hash、Python/NumPy版本、硬件、wall/CPU秒、数组shape/dtype/hash写manifest。现有路径不可覆盖。正式执行须先独立前瞻审查、注册、工程验证、干净源码冻结；从VALIDATED由loop_worker启动。
独立原数组复核预算300秒、0更新、**不生成新的任务样本**：显式矩阵filter重算全部最终和后缀后验，按冻结同一根再生bootstrap索引验证10000对均值/区间/判词，检查原数组/来源hash。这个复核是算法第二实现，不是独立样本复制。总研究执行+复核墙钟≤600秒、优化更新0。源码审查和packet反思时间另记录，不冒充worker算力。

只有独立结果审查后才在账本REVIEWED/DECIDED，随后packet-only任务反思和主代理reaction。任何成功也不自动开始EXP0016；失败不再以换噪声救此配置。Gate1保持NOT_PASSED。
