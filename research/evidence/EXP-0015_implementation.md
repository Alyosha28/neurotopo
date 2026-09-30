# EXP-0015 实现与工程状态（首次研究样本之前）

2026-09-26。已实现orbit_information及只读分析，接入受监督loop_worker；代码只生成固定HMM任务与oracle/区间，没有模型实例或优化器。预审协议READY_FOR_IMPLEMENTATION，所有原参数/阈值不动。

源、配置、协议、独立原数组复核脚本与测试在附包。已运行24项定向测试，覆盖隐路径枚举oracle、suffix条件化、确定性、bootstrap块大小一致性、阈值/NaN、配置改动拒绝、worker路由与入口迁移。尚未调用正式根77000007或bootstrap根77100007。

迁移新增显式migrate-entry-policy接口，仅接受v1到v2，必须引用非空授权证据，并要求当前DECIDED且反思完成；追加事件，不重写旧事件。已在live复制品测试：原字节前缀、旧current与iteration保持，迁移后EXP0015可进入QUESTION。原live未因该干运行而变动。

独立复核脚本不导入主filter/生成器，以显式转移矩阵及不同bootstrap分块重算原数组，不增加新任务样本。计划正式执行前完成全套工程验证、冻结提交、干净源码checkout，再以主账本VALIDATED启动；实际是否可运行以源码审查/工程证据为准。本摘要不代表正式运行已获通过。
