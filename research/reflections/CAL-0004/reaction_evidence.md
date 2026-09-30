# CAL-0004 任务后反思与主模型反应

2026-09-24。独立 `gpt-6-luna` / max 以 `fork_turns=none` 仅审查设计/结果 packet；三条逐字证据锚点经 `neurotopo.reflection` 校验通过。审查确认预定校准护栏成立，同时限制解释范围。主模型逐项核查如下。

**pairing_audit_detail — 接受并已补证。** 原结果中的 `training_pairing_and_block_compute_verified=true` 是执行断言摘要，单独不足以让另一上下文重算。独立脚本 `scripts/audit_cadence_pairing.py` 重新生成全部三种子×1500步的固定/可变任务输入：每种子五块的 d0/d1/d2 均为100/100/100；每臂每块691,200 token，线性运算代理相同；每步 H/R、cue/source、早末目标逐元素相等且两臂的累计摘要相同。`artifacts/calibration/gate1_cadence_pair_audit.json` 保存逐块数值与哈希，干净 checkout 完整重放的摘要也精确一致，见 `docs/CAL-0004_reproduction.md`。

**distribution_bundle_interpretation — 接受解释边界。** 可变臂同时见过 d=2 和多样节律；当前证据只支持“这整套训练分布在同样预算下改善本任务 d=4 后验拟合”。不能把效果唯一归于打破位置捷径，也不能声称特定神经机制。反思者提出的分块集中排序控制可用于**将来若要主张节律机制**的专门预注册实验；本轮不为通过现有校准而追加该控制。正式架构比较只把已资格校准的 varied 日程公平地施加于所有模型，不以 CAL-0004 为架构证据。

**resource_matching_boundary — 接受。** 两臂的样本数、参数与状态、每块及总 token、解析 dense 线性前向运算代理完全相同，累计训练步骤墙钟约3.4秒/臂/种子；报告没有测完整 forward/backward/optimizer FLOPs 或进程峰值内存。`research/evidence/CAL-0004_cadence_qualification.md` 和复现文档均只称代理预算匹配，不称完整实际算力相等。

反思没有改写任何 EXP 结果或 Gate 1。CAL-0004 保持工程资格结论；后续架构主张需独立预注册、使用新种子、相同 varied 日程及匹配资源。
