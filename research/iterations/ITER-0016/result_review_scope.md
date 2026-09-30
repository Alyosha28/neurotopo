# EXP0021结果独立审查范围

问题：执行/数值/判词是否遵循冻结v1，是否完整披露强控制和负种子。
输入：冻结提交153c076的协议、源码、配置、验证；全部artifacts/EXP-0021/v1。
最强反驳：单seed编码分工不重复、共同rotation结构先验、干预OOD非特异性；不许审查后换阈值。
验收：从独立干净checkout单次<=180s复演所有15最终模型、156新监测checkpoint、120原测试模型条件、75干预模型条件；独立重算XOR/metrics/R/M判词、检查原始哈希及不变量。共用forward/intervention实现这一边界必须披露；无重训/重试/新条件。保存review.json、reproduction.json、audit_execution.json到本iteration。若数值失败保留失败，不重跑到成功后隐去失败。
