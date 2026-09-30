# EXP-0013 三件工程预检证据独立复核

独立`gpt-6-sol`/high只读已提交报告、receipt、terminal seal、配置/共同锁、源码/验证套件与ResearchLoop状态；未调用注册任务信息或训练根。裁定 **`READY_FOR_VALIDATED`**，只许可记录工程VALIDATED，不是任务/控制资格。

三件各为PASS，SHA依次`7bcc10b6712db3f2a9ace1ed0a1284d4eb5278b3b4c0e24855b741eccb930f21`、`0adf8643832d0ee93aaade5c9366f91a9030fe5159c0977e51a7742c5285c5ae`、`0520cb9dd57b50060e9c6bcf726fca4e35d7e90c490df1e5901b8ca0c48383c4`，互引及当前后继Git提交文件逐字节吻合；工程源码提交`c42ed52`为当前运行准备提交真祖先，config/lock/pair及六个源码哈希一致。SC64/D64_full各100工程更新、合计200；末/初损失比0.760739/0.753911≤0.8。工程根42的T48主/保留两switch中full/last/truncated独立oracle最大误差≤`9.99e−16`。报告1.009226秒、receipt1.009785秒、seal记录receipt fsync1.010010秒，时间递增且≤300秒。既有工程套件PASS、218 tests与旧模型tiny-overfit也已封存。

这些数值只覆盖随机张量与小工程oracle；不代表任务信息、强单流学习、OOD余量、正式效果或Gate1。`ITER-0009`在审查时仍REGISTERED。
