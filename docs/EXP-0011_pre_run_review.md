# EXP-0011 v1 独立预运行批评

2026-09-24，独立 `gpt-6-sol` / high 只读提交 `2e2c258` 的v1、FR-0002设计/反思、未执行矩阵脚本、loop/模型/资源代码、CAL-0006与PF-RNN/RIMs/EviTrack先例；没有修改文件、执行矩阵、采样或训练。裁定 **REVISE**：该 `field_reference` 架构包问题可继续，不应因概念边界直接NO-GO；v1尚不能冻结矩阵或注册运行。

1. `loop_worker`尚无 EXP-0011 workflow，旧runner失败会停在RUNNING。任务信息/真实主控制余量/资源失败应正常产出可分析终止报告，主四分类为`INCONCLUSIVE`、原因作子状态，**零正式训练**且能完成ANALYZED→REVIEWED→DECIDED与packet反思；不能将资格失败写成架构效应FALSIFIED。
2. 静态参数预估正确：K4 13,200/3,984，S68_sparse 14,964/4,152（分配/有效）。现有资源函数漏Gaussian头/评分/反传等；须在候选研究训练前用未训练实例和隔离工程输入核全算子与计时，预写warmup及延迟/操作超限分支。
3. S68新15度环须验证行列度、自环、强连通、成功交换及哈希；S64_score 16度随机图的具体生成路径原稿未锁，现有`recurrent_mask`对groups=1 random会拒绝。图失败不得换seed。
4. v1已把98,000更新分完，额外clean-checkout**重训**一对主臂至少2,400更新，超过100,000；应把末段复现定义为零更新权重/前向重放，或前瞻修改预算。3600秒包含矩阵冻结却由VALIDATED后的worker单独计时，范围不一致，须写可执行的总时限核算。
5. 锁定Gaussian似然、早晚soft CE与 `KL(p_oracle‖p_model)` 的精确归约/数值稳定；明确正式IID护栏只管主三臂、稠密差用同一seed bootstrap索引、复核4/5各对照的计数和失败优先级。

矩阵脚本的种子与QR方向已有固定算法；在独立接受v2后才生成实值并写入v3数组/SHA，不得看实值后调任务阈值。两诊断臂不能归因纯K/score；任何领域正效应均不计项目Gate1。CAL-0006旧2D配置的CEILING不决定本8D配置资格。
