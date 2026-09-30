# CAL-0006 决定：关闭这个动态反事实滤波任务配置

2026-09-24。冻结协议 v3 与运行提交 `f170e9b`、原始记录及独立复核分别见 `research/design/CAL-0006_dynamic_counterfactual_qualification_v3.md`、`artifacts/calibration/CAL-0006_v3.json`、`artifacts/calibration/CAL-0006_v3_publication.json`、`docs/CAL-0006_reproduction.md`。独立科学审查只要求补报预写的自然修订切片 KL，未否决正式判词；完整解释见 `research/evidence/CAL-0006_dynamic_filter_qualification.md`。

解析 oracle 的任务信息护栏通过，dense/random H64 GRU 的三种子 IID8 末次后验 KL 全部<0.10 nat。唯一正式 OOD12 主条件要求**六个臂×种子记录全部**在 `[0.08,0.50]`；dense seed603 为0.0649678，其余五项为0.08210–0.11529。严格按预写优先级，标签 **`CEILING`**，`qualified_for_design_review=false`。这是本配置的一个近阈资格失败，不能称所有模型完全饱和，也不能由 dense/random 的非等有效参数差异归因拓扑或轨迹优势。

按一次性分支关闭该固定任务规格的架构设计入口，不将0.08改小、不移除 dense 护栏、不选其他长度/噪声/角度补救。它不改变 STR-0001 资格歧义、H3-SLOT `NO_SPEC`、旧 EXP/CAL 或 Gate 1 `NOT_PASSED`；没有实施 PF/RIM、适应、通信、生物图或语言模型。此决定是**基准工程资格**标签，不是项目四分类架构效应结论。
