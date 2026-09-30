# EXP-0011 v3 第三轮独立预运行审查

2026-09-24，独立 `gpt-6-sol` / high 只读提交 `49db848` 的v3、v2审查/回应、未执行矩阵脚本和loop源码；未修改、冻结矩阵、采样或训练。裁定 **`READY_FOR_MATRIX_FREEZE`**，范围仅是**先实现计时包装器，再由它唯一执行矩阵脚本，把数组/SHA写入v4**；不许可任务信息检查、工程/资格模型训练或正式实验。

v3在协议层解决v2三项阻断：preflight≤300、worker含analysis≤3000、零更新重放≤300且总和≤3600；测量有效但资源不合格可携原始数值进入VALIDATED，再由RUNNING零研究训练和ANALYZED(INCONCLUSIVE)正常闭环；主效应资源措辞已收窄为分配/有效参数及前向线性MAC保守、正式batch128路径三组p50差≤8%、其余算子完整披露。图定义、Gaussian/CE/KL、成对统计及98,000更新上限未见新硬矛盾。

**冻结前置条件：**当前 `scripts/freeze_exp0011_matrices.py` 只排他写矩阵及其数组SHA，本身没有单调计时、脚本SHA、退出码或不可覆盖执行报告。直接裸跑会永久缺失唯一冻结步骤的preflight耗时；须先提交、审查一个独占计时包装器，由其记录这些字段并唯一调用脚本。v3中“数组和SHA写入v2”是版本笔误，v4须更正为v4。

**训练前置条件另存：**现有 `loop_worker`还没有`pfstyle_field`，`resource_account`也未覆盖Gaussian/评分与完整执行操作，异常会留RUNNING。v4/实现阶段必须补工作流、失败闭环回归测试和严格涵盖analysis/写盘的worker时限；`READY_FOR_MATRIX_FREEZE`不是这些功能已存在的证明。
