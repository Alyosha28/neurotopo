# EXP-0008 v1 实现静态独立审查

日期：2026-09-23。对照 `experiments/EXP-0008/v1.md` 与 `configs/exp0008_v1.json`，只读检查 `final_probe.py`、`final_probe_analysis.py`、`loop_worker.py` 及两份新增测试。未运行 EXP-0008 正式数据流，未读取结果或修改实现；主代理的隔离 RNG smoke 不等于本次独立复现。

## 判定

**静态代码对照通过，待完整测试、源码哈希验证和干净提交；尚无正式科学结果。** 初审发现 `ridge_fit_macs_proxy` 按单头 `2×train_pairs×(65²+65×4)` 计算，却作为每 delay 的 A/B 两头总账目。主代理在正式运行前修为 `4×train_pairs×(65²+65×4)`；现已静态核对 runner 的新表达式，以及分析器对该绝对预算和 M3/C0 相等的硬门。主代理报告八项合成判定测试通过，本审查未独立运行完整套件。测试夹具采用修正后的预算值，但目前没有故意篡改该字段的负例；硬门本身在源码中明确，后续完整验证仍须通过。此次修正不改变主准确率或 CI 规则。

## 符合协议的实现

- `split_batch` 对 A/B 训练、验证、测试使用登记的互不相同 RNG，`validate_splits` 只对每 pair 的偶数行共享 pre-evidence 前缀检查 split 内/跨 split 重复。每个 split 保留相邻的 confirm/revise 两行；`final_hidden` 从零状态运行完整序列至最后 QUERY，训练头时用全部 `final_target`（每 pair 两条），`score_final` 的主 `revise_accuracy` 只读取奇数行。固定生成器把偶数行设为 prior confirm、奇数行为 alternative revise；运行前比较原 `task.py` 的冻结源码哈希。正式实现还应显式断言偶/奇行的前缀相等且 selector 恰为 prior/alternative，以让逐条结果记录的 `pair_split_verified` 不仅依赖固定生成器和外部测试。
- 所有 split 的全部最终隐藏状态均按批与原模型 final readout logits 做精确相等检查。A 头从 `train_a` 的 `2×train_pairs` 状态求唯一均值和总体 SD（0.001 下限），B 头沿用同一尺度；两头的标签均为正确的 final selected label。共用 `fit_prior_head` 使用 H64+截距到四类 one-hot、float64 正规方程、λ=1 仅罚斜率，形状 65×4、260 系数，无原模型参数更新或测试集拟合。
- 来源加载先查复制检查点字节、原始初值与掩码，并重放 EXP-0004 旧流；新运行器核对旧模型/任务/资源源码哈希、干净工作树和当前源码哈希验证。manifest 固定配置、检查点清单、来源 manifest、锁文件、源码和运行环境；逐臂保存输入/目标及共享前缀摘要、A/B 权重与标准化文件哈希、配对数、原任务分数和资源/时间记录。
- 分析器要求完整十种子×三模型记录及相同数据摘要、M3/C0 初值配对和同延迟模型资源匹配，核对两头各 260 系数、共同 scaler、文件哈希与训练 pair/example 计数。它以十个 seed 内的 **M3−C0 test revise A** 差值进行 20,000 次配对种子 bootstrap；C0/宽 GRU 的 A/B 验证均值与种子计数、M3/C0 的 A/B test revise 平均绝对差、原模型 revise 行为差距及非地板门槛均先于决策。`CI_upper<−0.03`→SUPPORTED，余下 `CI_upper<0`→PARTIALLY_SUPPORTED，余下 `CI_lower>−0.03`→FALSIFIED，否则 INCONCLUSIVE，与 v1 顺序一致。B 头没有被择优替代 A 主终点。

## 复核重点与科学边界

现有单元测试覆盖成对状态/读出、跨 split 重复、四类 ridge 及各主分支/护栏的合成记录；正式运行后仍须独立复算逐种子差值、区间、A/B 和原行为护栏，并在 clean checkout 重放完整 seed9101 子集。分析器从记录中的准确率重算统计量，尚未从保存的逐例探针 logits 独立重算准确率；需要在独立复现中核对这条数值链。每个 pair 只有一条 revise 主测试行，10 个原训练/图种子才是统计重复。高/低线性读出表现都不能单独判定信息是否被晚到更新保留或销毁，也不能把新监督头等同原读出；Gate 1 不变。
