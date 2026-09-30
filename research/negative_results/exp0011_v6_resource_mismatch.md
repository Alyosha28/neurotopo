# EXP-0011 v6：资源门失败，无架构效果数据

ITER-0008 的 `field_reference` 架构包比较在工程预检被注册资源规则停止。K4_score 与 S68_sparse 的动态状态均为68，静态参数与线性MAC代理落在冻结范围内；但三个计时区组的训练16、推理16/32共九个 K4/S68_sparse p50 比值全部为1.4811–1.7629，高于1.12失败线。工程预检有效、530次工程更新，正式研究更新为0，未生成任务信息、资格、正式或复核结果。主决定 `INCONCLUSIVE / RESOURCE_MISMATCH`，不构成多轨迹效果的正面或负面证据，也不计项目Gate1。

这暴露了设计时只看静态参数与线性MAC而低估执行成本的问题。算子profile和两个诊断臂时延提示逐token Gaussian评分与小算子调度可能是主要开销，但未做等价实现干预，不能宣称确切因果份额。v6结果和原始计时永久保留；不能重测择优或放宽阈值追认PASS。

原始证据：`artifacts/validation/exp0011_preflight.json`及`_receipt.json`；`artifacts/EXP-0011/v6/manifest.json`、`analysis.json`；独立审查`research/iterations/ITER-0008/review.md`。
