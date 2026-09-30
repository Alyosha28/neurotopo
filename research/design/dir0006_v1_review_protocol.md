# DIR-0006 一次性更正复核

2026-09-26，v1 投递前保存。问题：v1 是否已将逐操纵标签、旧任务余量、新候选未测余量、谱协变量与能力通道、0014历史理由、合法只读取证边界分开，并如实维持“不搜索”的当前决定？

输入：v1正文、原始事前设计、v0及原始来源（内嵌）。最强拒绝：主代理只是把所有未知都叫 NO_GO，从而掩盖不同失败原因；或更正又构成无来源的新禁令。标签只表当前执行停止，不得被读成类别被证伪。

判词：CORRECTION_ACCEPTED → 留档更正并关闭此次审计，不授权搜索/训练；CORRECTION_NEEDS_NARROWING / CORRECTION_REJECTED → 记录残余、维持停止，不再进行自动文本循环。不得用审查投票产生科学证据。请用固定 reflection JSON schema；证据逐字取 design/action；overall_assessment 附 FILES_READ。禁止 packet 外读文件、联网或运行研究代码。
