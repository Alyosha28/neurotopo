# EXP-0015 源码整改定向复核

2026-09-26，研究根未调用。问题只限于四项已定位实现缺口是否闭合：迟到analysis恢复；独立来源复核；300秒中途deadline及失败回执；完整区间/decision一致性。也检查修复没有改原任务/阈值或绕过zero-research-update。

最强拒绝：测试只验证错误消息，真实源与运行路径仍存在跳过入口。范围为冻结整改摘要、当前源/测试/config/protocol/run公共辅助及工程报告。只读packet，禁包外读取、联网与运行代码。

判词FIXES_ACCEPTED或FIXES_INCOMPLETE；前者允许主代理在已验证且干净提交中运行EXP0015；后者按具体错误修复，不能带病发科学判词。此处不重审项目因子资格、不改Gate1。review按reflection schema，引用逐字锚定，附FILES_READ。
