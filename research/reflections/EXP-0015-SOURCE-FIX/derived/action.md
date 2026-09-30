# EXP-0015 源码审查整改（首次正式样本前）

2026-09-26。SOURCE_NEEDS_FIX四项全部接受；任务生成器、阈值、样本量、bootstrap和预算未改。

SRC-001：worker现在把迟到analysis原子移动到analysis_aborted_before_fallback.json，再写规范INCONCLUSIVE分析。工程夹具模拟分析写完后301秒超时，确认旧字节保留、INCOMPLETE_BUDGET、账本进入ANALYZED。

SRC-002：独立矩阵复核移至独立模块orbit_information_reproduce，仍不导入主oracle/生成器。复核干净源码HEAD、全部src fingerprints及其git提交字节、配置/协议/入口/锁文件哈希、配置内容、完整validation回执和source表、Python/NumPy版本与解释器hash、正在执行的审计模块hash。夹具在临时Git仓库制造来源声明和各冻结文件改变，全部拒绝；无研究样本。

SRC-003：复核用SIGALRM/setitimer限制300秒，异常统一写pass_=false/REPRODUCTION_UNRESOLVED的独占回执。超时与hash失败注入验证回执持久化、alarm解除、不覆盖旧输出。正式调用将使用此有界入口。

SRC-004：所有7个指标（两均值、四上下界、uniform均值）、失败项/substatus及顶层decision均独立核对；改变上界/均值/decision的夹具均拒绝。

28项整改定向测试通过。随后完整工程验证的计数/源码hash见当前suite.json；包内包含其快照。此前228项版本已被此次实现更正取代，不用旧sourcehash报告放行新代码。

工程成本区分：按experiment-runner要求，scripts/verify.py还执行仓库既有合成tiny-overfit（validate.py共300工程更新）；pytest也含历史工程夹具。它们没有调用EXP0015研究根、没有训练ORBIT候选/控制，不能将“研究更新0”扩写成“整个工程测试没有任何优化器调用”。EXP0015特有工程检查只使用手写/枚举数据和非研究根；研究worker预算不包含仓库回归测试。

下一步只在此整改被独立接受、全套验证和干净源码冻结后执行已登记的零训练筛查。原ORBIT效果资格HOLD不变。
