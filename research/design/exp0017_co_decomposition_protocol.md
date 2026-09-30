# EXP0017-CO-DECOMP：已有预测的输出误差分解

2026-09-28。用户授权执行。**事后假设、分解前规则**：已知CO联合KL约.701接近ln2，本问题由该已暴露结果提出；在读取分量指标之前保存并提交此设计。不是新训练注册，不重开EXP0017，不改变旧判词。采用statistical-review；仅分析输出概率，不能推断hidden是否编码某分量、能力不可能性或Gate3语义。

问题：CO的误差是否主要来自二值分量b，而三值分量j已在输出中预测正确？最强反驳：接近ln2只是误差总和的巧合，可能两个分量都差、是关联错误，或者b本来分布不平衡。

范围固定：EXP0017v2/CO，D64主诊断与D16描述锚点，全部3seed17001–17003、IID32/OOD128、每组4096例，共12组保存的最终logits/target/latent/tokens。只读predictions.npz及对应result.json；核原SHA，不生成研究流、不读取保留namespace、不调用模型、无优化。其它cell不在本次范围。

q=3b+j。用float64 logsumexp归一化z，记log p(b,j)。精确点质量target应与latent一致；j=q%3,b=q//3。分解每例：

`L_joint = -log p(b*,j*) = -log p(j*) - log p(b*|j*)`。

同时计算反向分解`L_joint=L_b+L_j_given_b`，防止把相关性误当分量错误。保存两方向各项均值、准确率、b真实正类率、逐真实(j,b)子组统计、p(b=1|j*)偏离1/2的MAE/RMS、p(b=1)偏离1/2的MAE、给定正确j的二值准确率及其NLL。输出联合预测准确率只能作行为描述，不主张隐状态忘记。

预写描述判据（不是显著性检验或总体等价证明）：对D64每个seed×长度均要求j_NLL<=.05、j_accuracy>=.95、b_given_j_NLL/joint_NLL>=.90、abs(b_given_j_NLL−ln2)<=.03、conditional_b_uniform_MAE<=.05、conditional_b_accuracy∈[.47,.53]、true_b_fraction∈[.47,.53]。全部六组满足→SUPPORTED/POSTHOC_OUTPUT_BINARY_UNRESOLVED；若所有六组j_NLL占比<.5但部分接近性条件不满足→PARTIALLY_SUPPORTED/OUTPUT_ERROR_CONCENTRATED；若任一组j_NLL占比>=.5→INCONCLUSIVE/MIXED_OUTPUT_FAILURE；如果全部六组b_given_j占比<.5→FALSIFIED/LOCAL_BINARY_DOMINANCE（只否证此输出诊断）。先判全六组反向结果，再判全满足，再部分，再其他不确定。阈值是本次事后分析的实用描述约定，无功效或未暴露集支撑，不计Gate1。

seed作为汇总单位，均值、sample SD、df2的95%t区间只作描述；不把4096序列当模型复制。阈值附近不增加样本/种子。D16结果不改变主结论。

验收：手工分布的精确分解（正确j且b均匀→0+ln2；正确b且j均匀→ln3+0；六类均匀→ln3+ln2）；在每个真实数据组逐例验证两个chain-rule恒等式误差<=1e-12、复算联合KL与保存值差<=1e-10；检查真实支持为点质量且合法。另用直接概率归一化及scalar索引独立实现复算，均值误差<=1e-12。异常停止并保留错误，不删除失败文件。

预算：主分析与独立数值重算合计<=60秒，单次受监督命令统一计时；输出exclusive-create，先json.dumps校验再写，失败另存error记录。无模型重放或训练。审计脚本hash、输入hash、运行秒数和原件不变复核全部存档。

分支意义：若二值误差集中成立，只确定下一种解释要围绕该输出分量展开；不决定是表示、优化还是读出失败。若不成立则撤回ln2线索。任何新训练干预需独立定义，不由本轮自动启动。完成后按docs/post_task_reflection.md做新上下文packet-only反思及主代理reaction，作为独立只读研究审计归档，不推进live实验账本。
