# SCR-0002 所援引的旧 CAL 量化范围

两份冻结工程校准均来自源码提交 `fedc4ea4edd3622d68a57e4099c32270376b8da7`，各三种子、H64 **dense GRU**、1500 updates、batch128；它们**没有训练 H3-SLOT**，也没有匹配一个未来 PF-GRU/固定槽的实际主控制。数值为模型输出后验相对同任务**解析 oracle 后验**的末次 `KL(target‖model)`（nat），不是随机真标签到100%准确率的差。

| 校准 | 冻结原始位置与任务 | 三种子末次 KL-to-oracle | 范围 |
|---|---|---|---|
| CAL-0001 | `artifacts/calibration/gate1_belief_v1.json` 的 `model_results[*].calibration.final_kl_to_oracle`；四静态内容假设，已显示可靠度与标记干扰，6条有效线索。 | seed101–103：0.0005421144 / 0.0002879277 / 0.0003387307；均低于事前0.01天花板阈值。 | 四个 log 权重足以作正确递推；这一个配置不留已登记的模型余量。见 `research/negative_results/CAL-0001_belief_benchmark_ceiling.md`。 |
| CAL-0002 | `artifacts/calibration/gate1_source_belief_v1.json` 的 `model_results[*].calibration.final_joint_kl_to_oracle`；四内容假设×两种隐藏来源可靠度，8条有效线索。 | seed201–203：0.0022676745 / 0.0018214203 / 0.0020322541；均低于事前0.01天花板阈值。 | 八个联合 log 权重足够；这一配置同样不留已登记模型余量。见 `research/negative_results/CAL-0002_hidden_source_ceiling.md`。 |

这些事实只约束当时的小静态后验任务、指定训练预算、dense H64及其 oracle-relative KL；它们不能证明所有动态/未建模假设任务封顶，也不能独立否定固定身份 recurrent 槽。SCR-0002 的当前实现 NO-GO 直接来自**候选规格及可区分预测缺失**，不是旧 CAL 的普遍化结论。对 CAL-0006 的动态任务已有单独正式近阈资格失败记录，不能混用三种任务的阈值。
