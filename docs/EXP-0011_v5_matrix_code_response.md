# EXP-0011 v5 对矩阵代码审查的回应

2026-09-24，仍未生成矩阵或任务数据。独立审查 `docs/EXP-0011_matrix_verifier_review.md` 判代码REVISE，主代理接受三项阻断；修订后的脚本与v5须再次只读复核才执行。

生成器现在在最终文件`open("x")`前核raw/QR/det/最终数组有限性、正交和det，JSON禁NaN/Inf；保存自身字节SHA、当前提交、NumPy构建/BLAS输出、Python解释器及uv.lock SHA。校验器现在把Git/源码读取放进可写FAIL报告的异常保护，记录`sys.executable/argv/cwd`及脚本/工作树状态；从同seed独立逐列纠正QR符号并字节重算。v5明确环境版本/构建/lock不同为`ENVIRONMENT_REPLAY_MISMATCH`：保留原矩阵及错误，不容差放行、换矩阵或重抽；未来跨环境使用以本次提交的数组字节/SHA为权威任务定义。

修订只改变防无效浮点与来源校验，不变A算法/seed、任务、模型、阈值或预算；还没有任何任务/模型表现可用于选择。旧v4与否决审查保留。本次不修改未通过审查的一次性包装器，它已从现行源码删除。
