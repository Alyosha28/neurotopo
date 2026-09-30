# EXP-0003 v1 运行前独立核对（2026-09-23）

范围：只读核对 `experiments/EXP-0003/v1.md`、`configs/exp0003_v1.json`、`src/neurotopo/study.py`、`src/neurotopo/study_analysis.py`、`tests/test_study.py` 及现有 EXP-0002 结果；未运行实验。本文件记录冻结和正式运行前的审查意见，不是结果或预注册的事后改动。

## 一致之处

- 四臂均为 H64 dense GRU、6000 updates、同一优化器/批量/损失。直接组始终 delay8；课程组边界为 1–2000: delay0、2001–3000: delay2、3001–4000: delay4、4001–6000: delay8。监测点按当前训练 delay 生成；最终测试只在最终权重上进行。
- `variant_batch` 先调用原生成器，再将所有 `DISTRACTOR` 事件的候选/数值八维清零；delay8 包括查询前八个和证据后两个，类型、长度、目标、其余事件及 RNG 消耗保持。测试覆盖该不变量。噪声/空白同日程、同种子共享初始化与原始样本抽样；不同日程因长度改变而不逐样本配对。
- 监测流 `520000+delay` 与最终流 `530000+delay` 分开，最终覆盖 noise/blank × delay8/16。直接组每条样本 18 个事件；课程总事件数按每序列 82000，对比直接 108000。代码记录实际 train tokens 和线性 FLOP 代理，故可披露计算差异。延迟 0 在旧检查点 400 仍近随机、2000 已满分；首段改成 2000 有既有依据，但其余课程段仍是待检验设定。

## 冻结前需修正

1. **“SUPPORTED = curriculum recovery” 缺少直接组失败条件。** 当前协议和分析器只要求 `noise_curriculum` 达到可学阈值且每种子的 joint 比 `noise_direct` 高至少 0.10。若新种子下 `noise_direct` 也可学，代码仍可能宣布“恢复原任务可学性”；此时应称课程增益，而不能称从不可学中恢复。建议在 SUPPORTED 条件增加 `noise_direct` 不可学，或将该标签明确改为“固定预算课程增益”。保留每种子 0.10 差与三种子可学阈值，不依据结果改变。
2. **资源匹配失败仍可能触发后续架构分支。** `study_analysis.analyze` 在 `matching=False` 时把决定改为 INCONCLUSIVE，却先按 `noise_curriculum` 可学性设置 `MATCHED_ARCHITECTURE_CONFIRMATION` 分支。分支必须在核对失败时退回诊断/修复。当前 `matching` 还只核对同日程臂的 `train_tokens`；为兑现协议中的配对与预算记录，应至少核对四臂 6000 updates/相同 examples、同日程 noise/blank 的初始参数哈希、tokens 和线性 FLOPs，以及同一最终评估键的数据哈希。
3. **“factorial” 的配对效应输出不全。** JSON 仅显式保存 noise 课程减直接的逐种子 joint 差；协议承诺“每种子与配对差”，而问题同时询问 nuisance payload。请预先固定并输出同日程 blank−noise 的逐种子 delay8 pre/joint 差，以及课程效应在两种 payload 下的差。它们可作为描述性次要对比，无需增加显著性检验或把空白条件的成功说成纯语义干扰。

## 解释边界

空白干预同时改变无关输入的信息量和幅度，保留 recurrent 时间步；空白成功、噪声失败只能支持“随机负载相关障碍”。两种直接组都失败也不能单独归因于纯时间跨度。课程组虽与直接组更新数/样本数相等，却只接受 2000 次 delay8 训练且总 token/FLOP 更少；课程失败不排除其他固定课程有效。EXP-0003 仅诊断 dense 基线可学性，不能修改 EXP-0001/0002 的结论或推进 Gate 1。

## 主代理整合记录（正式运行前）
以上三项已在 EXP-0003 v2 中明确并修复：恢复需要直接组不可学；资源/数据/初始化核查失败阻止后续分支；所有公共测试条件输出负载与课程的逐种子差。v1未运行，原始记录保留。课程固定2000/1000/1000/2000，无中途调整。
