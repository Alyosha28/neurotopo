# ITER-0006 独立反思后的核验与行动

2026-09-24。独立 `gpt-6-luna` / max / `fork_turns=none` 仅收到 `packet.json` 的路径及审查格式说明，按要求未接收历史对话。审查初稿把两处证据摘录拼接成新句子，`validate_review` 拒绝；让同一反思者仅据原 packet 修复原文锚点后，2 条 issue 通过校验。这是输出约束的真实失败与修复，不算新增科学证据。

## I1：分析摘要不足以独立核验执行细节

接受**packet 可见性问题**，不接受“原实验缺少这些记录”的推断。`analysis.json` 只有 `provenance_and_matching_pass: true` 汇总；`artifacts/EXP-0008/v1/manifest.json` 为 `REGISTERED/COMPLETED` 且含提交、配置、锁和源码 SHA-256 及墙钟。30 份 `seed-*/<model>/result.json` 均存在、`source_replay=EXACT`；每份按延迟保存检查点、初始参数、掩码、数据和系数哈希、A/B/验证/测试拆分计数、参数与计算代理、阶段耗时。抽查 seed9101 M3 delay32：A/B 各4096 pair，validation1024，test2048；分配/有效参数15428/6212，64 状态值，每例稠密线性 MAC 代理621440。独立结果审查也逐项核验，不能把摘要布尔值单独当作这些细节的证明。

行动：`src/neurotopo/reflection.py` 现在允许把 `manifest`、`verification`、`decision` 作为 action 产物装入同一个最小 packet；`src/neurotopo/loop.py` 对后两者核对账本哈希。相应测试验证 bundle 的可见性、来源匹配和证据引用。下一轮构包时按问题加入必要产物，而非只给汇总布尔值。

## I2：分析结束与任务完成的时间点混淆

接受**packet 时序表达问题**，原实验的结束前置条件实际已完成。账本记录：`ANALYZED` 为2026-09-23 16:04:49 UTC，`REVIEWED` 为16:12:27，`DECIDED` 也为16:12:27且序号在后；`research/iterations/ITER-0006/verification.json` 包含独立科学审查和 seed9101 clean-checkout 复现文档的 SHA-256，后者记录 PASS。`analysis.json` 的 `complete:true` 表示分析产物完成，并不能单独证明审查和复现结束；其中的 `next_branch` 是注册分析给出的分支，正式采用见后续 DECIDED 记录。

行动：action bundle 可附 `verification` 与 `decision`，让新上下文看到上述前置步骤是否有完成记录；`docs/post_task_reflection.md` 明确其各自来源和局限。历史 EXP-0008 的 `INCONCLUSIVE` 与 Gate 1 状态均不改写。

本反思审计未重跑30份结果，也未把独立反思者的疑点当作原实验失败。它改变的是后续反思 packet 的可核验性和下一轮研究的输入质量。
