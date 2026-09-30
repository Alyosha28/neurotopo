# DIR-0007 终稿审查后的处置

2026-09-26。全新上下文 `orbit_final_review` 只读冻结packet `9c27a3a0…9155`，判 **DESIGN_INCOMPLETE_HOLD**，R1 high、R2 medium。两项全部接受；原v1和脚本保留，不覆写冻结输入。以下是**审查后更正和核验**，不追称审查者已看过，也不把局部修复改写成设计通过。

## R1：接受并更正共轭映射

原v1写 A_F→B_X 和 B_F→A_X；后一项错误。对所有模型作同一个输入/输出坐标变换Q时，正则群依次为 G→QGQᵀ→Q²G(Qᵀ)²。这里Q不满足所需例外条件，不能把第二个映射叫作A_X。

正确实现约定：A_F的pushforward使用QGQᵀ；B_F的pushforward使用Q²G(Qᵀ)²，称其为**B_F的共轭副本**，不另命名为A_X、不增加研究臂、训练预算或复制数。E_F的副本使用G'。X始终仅为实现校验；主要四比较均在F上，不受这个命名修正影响。

主代理对256个矩阵基向量逐一核验，错误的B_F→A_X投影等式最大逐元素误差 **0.0625**；正确的Q²投影等式误差 **0**。结果 `research/design/dir0007_conjugacy_correction.json`，明确标为post-packet。尚未运行完整GRU/Adam工程等变验证，不把矩阵关系冒充完整模型检查。

## R2：接受并隔离修复判词守卫

原冻结的 `scripts/dir0007_power_plan.py` 中classify没有isfinite守卫；有限正下界搭配NaN上界确实可进入SUPPORTED_LOCAL_ONLY。原代码保留作为失败证据，**不得直接作为后续runner判词函数**。

新增 `scripts/dir0007_decision_guard.py`：在分类前检查上下界长度、有限值、顺序；无效输入统一INCONCLUSIVE_EXECUTION，再调用原有限输入规则。保留原8个边界例，新增24个NaN/±Inf位置回归与2个长度/上下界错误例。结果见 `research/design/dir0007_decision_guard.json`。这个修复不重算、不改变条件功效结果，零任务样本、零优化更新。

## 主代理决定

原v1判词 **DESIGN_INCOMPLETE_HOLD** 保留，审查线按事前规则收尾，不发起第二次终稿修订审查。上述已核验差错处理不解决项目因子资格、0049/Generic范围、控制实测余量、候选方差假设、实测资源和完整模型工程等实质条件。

交付地位：一个具名、带比较与停止规则的**未注册方法候选设计**；可供后续有针对性的准入判断，当前不是可运行实验，更不是Gate1证据。没有搜索图、生成任务样本、训练模型或修改历史实验结论。Gate1仍NOT_PASSED。
