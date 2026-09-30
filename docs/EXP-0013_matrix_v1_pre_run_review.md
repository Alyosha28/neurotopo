# EXP-0013 双矩阵/γ冻结 v1 源码预审

独立`gpt-6-sol`/high只读提交`2613876`的任务/矩阵协议和冻结脚本；没有计算Q、γ、任务样本或模型结果。裁定`REVISE_BEFORE_FREEZE`。正常路径的两个SeedSequence子序、PCG64/QR列符号及行列式修正、D_base(48)的l=0..47/t≥17累计、γ公式与`<f8`哈希均一致；干净checkout/原子链接/receipt门存在。

硬偏差在明订的`NO_TASK`失败分支：非有限D_base仍写作严格JSON数值，`allow_nan=False`会报错而不产生报告/receipt；非有限γ只写null而无位模式或失败原因。v2在任何运行前改为合法JSON的null+`<f8`位模式/文字分类，并增加目录fsync，科学菜单/阈值不变。主工作区仍有用户未跟踪文件，实际执行须用干净checkout。此预审不授权任何任务样本或训练。
