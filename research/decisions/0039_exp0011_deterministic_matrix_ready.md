# EXP-0011：固定任务矩阵设计步骤获得有限许可

依据v4和`docs/EXP-0011_v4_review.md`，判 **`READY_FOR_DETERMINISTIC_MATRIX_FREEZE`**。可执行范围只有：从已经提交的固定seed/QR生成脚本创建`configs/exp0011_matrices.json`，用事先提交的独立只读校验器重推并核实际数组/SHA/有限性/正交性，留命令/依赖/耗时，提交数组，v5只补值与SHA。生成或校验失败保留原样、停止；不得改seed/阈值/任务配置后称同一次尝试。

决定0037的计时包装器条件经独立审查后由v4**前瞻替代**，没有执行旧包装器；其失败记录和源码历史保留，当前代码可移除。此许可**不**涵盖任何任务信息数值、控制资格、模型实现训练或Gate1判定。ResearchLoop仍ITER-0007 DECIDED，项目Gate1仍NOT_PASSED。
