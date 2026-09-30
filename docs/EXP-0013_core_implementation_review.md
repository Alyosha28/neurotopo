# 阶段显露任务、三臂GRU与资源账独立工程审查

独立`gpt-6-sol`/high只读提交`917fc64`的`phase_reveal_task.py`、`phase_reveal_models.py`、`phase_reveal_resources.py`及专项测试，对照EXP-0013/0014 v3；仅用工程root42，未调用冻结任务信息/训练根或研究更新。裁定 **`READY_FOR_RUNNER_INTEGRATION`**，不等于放行任务数据。

审查核SeedSequence四子流/cue错误映射、t>switch相位、full四H枚举、末帧历史边缘化与截断u1:16协方差；GRU门/分路输入与循环mask、RS/SC同种子可训练初值逐字节相同、D_full全输入、16004/4484及16004/16004参数账、前缀/后缀状态重放。12项原专项测试通过；另独立以工程root42验证T48 switch16和switch20的full/last/truncated oracle最大误差≤1e−15。主代理将这一T48两switch检验补成专项测试，现9项任务测试通过。

进入运行器仍须逐seed核RS/SC初值、SC度/自环/交换/图SHA及持久化输入通道；资源账仅为线性MAC代理，须实测全算子/时延。`step(recurrent_weight=...)`可接受外部权重，正式训练/操作须固定使用内部已掩码路径。独立核心审查未评价资格runner、研究预算或Gate1效果。
