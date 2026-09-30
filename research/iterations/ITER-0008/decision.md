# ITER-0008 / EXP-0011 v6 决定

**INCONCLUSIVE；子状态 `RESOURCE_MISMATCH`。** 冻结 v6 的候选 `K4_score` 相对真实单流主对照 `S68_sparse`，在三组训练16、推理16/32共九个实测 p50 比值上为 1.4811–1.7629，均超过注册的 1.12 资源停止线。工程预检有效，静态参数、状态及线性 MAC 范围通过，但不能抵消实际时延超标。ResearchLoop 在 `RESOURCE_STOP` 正常结束；没有任务信息读取、资格/正式/复核模型、研究更新或任务效果判词。

最强无趣解释是 PyTorch eager 小算子与逐 token Gaussian 评分路径的实现成本，而非多轨迹在任务上的计算能力。这一解释有 profile 和诊断臂时延的间接支持，尚未由等价实现的前瞻干预证实。此轮只否定 **v6 在指定 CPU/实现/预算下符合已注册资源约束**；不否定多轨迹效应，也不构成 Gate 1 证据。`field_reference` 本来也不计项目 Gate 1。

下一行动为事前注册单独的等价评分实现与资源检查，限定在工程输入上核前后向数值、状态/参数与原阈值；不改写 v6、重抽任务矩阵或偷看研究根。若优化后仍超资源门，终止这一低信息架构包路线，先重定义有科学判别力的 K 因素和对照，而不是继续寻找刚好过关的任务。Gate 1 保持 `NOT_PASSED`。

依据：`experiments/EXP-0011/v6.md`、`artifacts/validation/exp0011_preflight.json` 及 receipt、`artifacts/EXP-0011/v6/manifest.json` 与 `analysis.json`、`research/iterations/ITER-0008/review.md`。
