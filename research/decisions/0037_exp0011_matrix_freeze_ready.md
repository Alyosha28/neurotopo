# EXP-0011：仅固定矩阵步骤获得事前许可

按已提交v3与 `docs/EXP-0011_v3_review.md` 独立审查，判 **`READY_FOR_MATRIX_FREEZE`**。具体动作仅限：先提交并检查独占计时包装器；包装器以固定脚本/seed执行一次矩阵冻结，保存原数组、little-endian float64 SHA、脚本SHA、提交、退出码和单调耗时的不可覆盖报告；随后在v4正式协议/配置中填入该数组和SHA。不能看数组后改任务阈值、A生成规则、主长度或seed。

当前未产生矩阵、任务数据或训练结果，ResearchLoop仍是ITER-0007 DECIDED，Gate1仍NOT_PASSED。EXP-0011是`field_reference`，未来任何领域正结果也不自动计项目门；代码实现、资源工程验证、控制资格和正式训练都需v4注册及独立检查之后按顺序进行。
