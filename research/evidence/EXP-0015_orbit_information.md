# EXP-0015：固定 ORBIT-SOFT 任务通过信息门

2026-09-26。ITER-0010，冻结源码 `e8b4d3c258efcd544f67ea4a0288f54a693b31a0`。16,384条长度64的序列，一套事前固定任务参数、一个研究根；0候选/控制研究更新。**SUPPORTED / TASK_INFORMATION_ONLY** 已获独立科学审查接受，任务后packet反思0项问题，主代理reaction及账本反思归档完成；权威记录为ITER-0010/review.json、decision.md与research/reflections/ITER-0010/。

| 冻结指标 | 实测 | 95% bootstrap区间 | 原门槛 | 结果 |
|---|---:|---|---|---|
| KL(full posterior ∥ uniform) | 2.0991721195 nat | 原协议为均值点值门 | 均值≥0.20 | PASS |
| KL(full ∥ suffix1 posterior) | 1.0875542070 nat | [1.0727697612, 1.1018843362] | 下界≥0.05 | PASS |
| KL(full ∥ suffix8 posterior) | 0.0601253407 nat | [0.0559058729, 0.0645065477] | 下界≥0.02 | PASS |

解释：在这个固定生成器上，只见最后8帧的正确Bayes后验仍与完整历史后验有非零、超过原阈值的信息差。没有测试更长截断窗；不说明完整64帧全部必要或必须增加状态维数。已知精确forward filter仅需16类后验向量；本结果不否定简单单流解。

## 执行与复核

主worker（生成+oracle+10,000次序列bootstrap+分析）3.380530791秒；独立原数组复算0.755529250秒，均在各300秒硬限内。复算改用显式转移矩阵和不同bootstrap块大小，覆盖全部后验、逐例KL、bootstrap均值、完整区间、顶层判词及冻结来源；最大数值误差1.576516695e−14，小于1e−10。它是同一批数组的第二算法复算，不是独立样本研究复制。

250项工程测试与既有tiny-overfit通过；这是工程证据，不是ORBIT模型性能。仓库工程套件确有旧模型合成优化检查，研究更新0只指EXP0015；不得扩大为整个会话无任何优化器调用。

原始数据与后验：`artifacts/EXP-0015/v1/raw.npz`、`oracle.npz`；analysis、manifest、independent_reproduction.json记录SHA与计算环境。研究根77000007、bootstrap根77100007现已暴露，不得重新用于独立效果评估。

## 对研究决定的实际影响

原筛查的关闭分支未触发；这个固定任务的信息条件保留。**尚无任何实际GRU主控制的headroom/学习结果，未测ORBIT组织收益或候选方差。** 项目因子资格、0049/Generic范围、模型/资源/功效的剩余条件不变；ORBIT效果研究仍HOLD，Gate1仍NOT_PASSED。

本轮另修复v2入口代码与v1账本之间的断连：追加event-000088，原87条事件字节前缀和旧current均保留；这只是落实0080，不是新的标准放宽。此次科研推进是取得任务信息证据，不把流程修复或设计文字当研究成功。
