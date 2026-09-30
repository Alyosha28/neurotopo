# EXP-0007 v2 实现静态独立审查

日期：2026-09-23。对照 `experiments/EXP-0007/v2.md` 与 `configs/exp0007_v2.json`，只读检查 `transfer_probe.py`、`transfer_probe_analysis.py`、`loop_worker.py` 和两份新增测试。未执行 EXP-0007 正式数据流、未读取模型结果、未修改源码；已有工程 smoke 的报告不是本审查独立复现。

## 判定

**静态代码对照通过，待完整测试、源码哈希验证及干净提交；尚无正式科学结果。** 初审曾发现分析器把 H8A/B 源头与 H32A/B 目标头的测试准确率平均绝对差都接到 `maximum_source_refit_variation`，漏读登记的目标头键 `maximum_refit_variation`。主代理在正式运行前修复：现在 `source_test_mean_abs` 读取源头键，`target_test_mean_abs` 读取目标键；新增两项把这两个阈值故意设成不同值的合成边界测试。已静态复核代码和测试条件；主代理报告 11 项 transfer 决策测试通过，本审查没有独立执行完整测试。v2 登记的两个值均为 0.02，此修正没有改变预注册数值规则。

## 对照通过的部分

- 数据生成用登记的 A8/B8/V8、A32/B32/V32/T32、A64/B64/V64/T64 各自 RNG。`validate_splits` 对所有单份 pre-query 前缀检查重复；各数据集只取 counterfactual pair 的偶数行，提取状态的循环在 QUERY 后停止，未消费 EVIDENCE 或后缀。完整 pair 只用于原任务行为读取。每个提取前缀都按批与原模型 pre logits 做 `torch.equal`，覆盖全部拆分。
- 唯一均值和总体标准差由 **A8 训练状态**计算，下限 0.001。H8A/H8B/H32A/H32B/H64A/H64B 的训练、验证、测试共用这一尺度；B8 没有暗中重新拟合 scaler。每头 4,096 个标签为 `pre_target[::2]`，64 特征加截距、四类 one-hot、float64 正规方程、λ=1 仅罚斜率，系数形状 65×4，即 260 个参数。H8A 与 H32A 在同一 T32 隐状态上计准确率，主差值方向与协议一致。
- `load_verified_model` 复核复制检查点字节、旧初始参数与掩码，重放原实验测试集；新 runner 还比较 EXP-0004 的模型、任务和资源源码哈希，要求干净工作树及通过的源码哈希验证。manifest 固定配置、检查点清单、来源 manifest、锁文件和源码摘要。逐臂记录保存数据/目标摘要、A8 scaler 与各头系数/文件哈希、完整原任务资源、状态提取 FLOP 代理和各阶段实际时间。
- 分析器要求完整十个登记种子/三模型、相同数据摘要、M3/C0 初始参数配对及主要资源匹配、六头各 260 参数及同一 scaler 文件摘要。它以十个模型种子内的 `D=(H32A−H8A)_{M3}−(H32A−H8A)_{C0}` 做 20,000 次种子层 bootstrap。M3/C0 四个同域验证分数逐种子 ≥0.90，M1 对应四项按种子均值 ≥0.90；原 pre 准确率差距和绝对下限先行。SUPPORTED、PARTIALLY_SUPPORTED、FALSIFIED、INCONCLUSIVE 的有序条件与 v2 一致；delay64、宽 GRU 及 B 头不替代主终点。

## 运行后仍需独立核对的限制

A/B 护栏用的是各模型十种子**准确率差的绝对值均值**，不是逐实例预测一致率，也可能遮蔽个别种子的较大差异；协议如此登记，正式报告须逐种子披露。分析器从保存的准确率重算 D，并核查系数/scaler 文件哈希，但不从原始逐例 logits 独立重算所有探针准确率；计划中的单种子 clean-checkout 重放和结果审查应补这一层数值核对。所有 CI 条件于一次固定新测试流和同一批既有训练种子。即使决策为 SUPPORTED，也只能描述固定 ridge 头的跨延迟迁移差异，不能将其归因于单纯时间、干扰、语义假设或原 readout 的因果失效；Gate 1 不变。
