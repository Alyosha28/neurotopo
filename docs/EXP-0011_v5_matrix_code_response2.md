# EXP-0011 v5 矩阵代码第二次修订回应

矩阵仍不存在，任务/模型未运行。主代理接受第二次独立代码复核：生成器现在同步写入并读回JSON，再核数组SHA；校验器以`parse_constant`拒绝全文件任何NaN/Infinity（含不参与计算的字段），核四条`checks`有限性、bit generator、40位提交及该提交中生成器文件SHA；报告实际解释器/argv/cwd与工作树快照。生成时记录运行线程环境变量，校验时与NumPy/BLAS构建/lock一并比较；已记录环境一致但重推字节不同单列`RECOMPUTE_ENV_UNDETERMINED`，保留权威原矩阵，不容差放行或重抽。

这些是实现v5既定“标准JSON、来源、冻结环境、逐字节重算”要求的代码修正，未改seed/QR/decay、任务和模型阈值。需第三次独立静态复核明确`READY_TO_GENERATE`后才运行一次固定脚本。
