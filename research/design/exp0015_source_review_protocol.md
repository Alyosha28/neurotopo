# EXP-0015 源码与恢复流程审查

2026-09-26，正式样本前冻结。前瞻协议已判READY_FOR_IMPLEMENTATION、零issue；本次检查实现是否兑现固定协议，并检查入口账本迁移是否仅落实既有0080，不覆盖旧记录。

输入只有实现摘要及其source/config/protocol/tests/0080原件包。最强拒绝：oracle或bootstrap虽跑通却实现了不同问题；失败/脏树/无效验证被误放行；迁移偷改旧决策。请静态检查生成器次序、column转移、suffix初始化、样本bootstrap索引配对、固定阈值、失败保存、zero-update、worker限时与源码hash、迁移单向/幂等/反思门。无需修改代码或运行任务。

判词SOURCE_ACCEPTED（只认可源码可进入正式VALIDATED审查）；SOURCE_NEEDS_FIX（列可定位错误，修复并工程验证后才运行）；SOURCE_REJECTED（协议级问题，停）。不依据预想结果调参数。review按标准schema并附FILES_READ，只读packet派生文件，不能读取包外文件/网络或执行研究代码。
