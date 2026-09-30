# FR-0002 项目 Gate 1 资格审查：先决定，再优化 v7

2026-09-24，**在任何 EXP-0011 后续任务信息、模型效果或等价评分优化前**固定。本审查不重判 v6 的 `INCONCLUSIVE / RESOURCE_MISMATCH`，不改变 Gate 1，也不使用任何任务样本。它只问：**现有 FR-0002 `K4_score` 对真实单流 `S68_sparse` 的比较，若未来在新协议通过资源、任务资格和正式效果门，是否能作为项目 Gate 1 的“多轨迹因素”证据？** 判断对象是已写明的状态更新、评分和对照，而不是某个假想改良模型。

## 证据范围与最强反驳

审查用户章程 Gate 1 的“state-capacity-matched / compute-matched generic multi-trajectory effect”、§26 功能验证与必查先例；当前 `research/stage_gates.md`、决定0006/0019/0023/0034/0035、`docs/gate1_k_factor_qualification_template.md`、`research/entries/EXP-0011.json`、冻结 `experiments/EXP-0011/v6.md`、`src/neurotopo/pfstyle_models.py`、`research/prior_art/exp0011_scope.md`。核对一手 [PF-RNN论文](https://ojs.aaai.org/index.php/AAAI/article/view/5952)、[作者代码](https://github.com/Yusufma03/pfrnns) 与 [RIMs论文](https://arxiv.org/abs/1909.10893) 的机制覆盖范围。只读，不运行生成器、task root、训练、工程预检或重新计算已有模型效果。

最强反驳：`K4_score` 同时改变独立循环块、四个模式索引 Gaussian 头、四条累计证据分数和输出形成方式；`S68_sparse` 是单个 GRU 加直接四分类头。即便两者总状态/参数/时延相配，也可能只估计**已有 PF 风格完整架构包**对普通 GRU 的效果，而非 K 轨迹因素。`S64_score` 自身也有四条持续评分，仅能作诊断；`K4_unscored` 与主对照的读出又不相同。过度坚持“必须先新颖”亦可能把用户原定的通用效应 Gate 1 偷换成论文贡献门，故需逐项区分实证效应、因素归属、新颖性和 Gate 2 启动。

## 预写决定规则

- `PROJECT_ELIGIBLE_FACTOR`：现有主比较及冻结消融已经让**K=多条递归候选更新**相对功能性 K=1 单流成为可辨识主因素，评分/输出装置差异有预写可检验的分解，且这不是仅把先例包重命名。即使给此标签，仍须在任何效果结果前另外版本化项目计分、入口策略、资源/操作诊断与 Gate 2 启动规则；不得追认 v6 或先例作者结果。
- `FIELD_ONLY`：现有主比较同时操纵 K 与评分/头/输出包，或诊断不能隔离 K；正结果至多报告领域包效果。此分支应停止**以项目过门为理由**继续给同一 FR-0002 做 v7 性能优化；可保留有独立科学问题和预算的领域研究，但不据此开启项目轨。
- `UNRESOLVED`：上述状态/更新或先例证据不足以判其中一支；列出最小缺失证据，保持现行项目入口拒绝，不运行效果。不能把“还没效果数值”当作因素资格不可判断的唯一理由。

审查必须逐表比较 K4_score、S68_sparse、S64_score、K4_unscored 的持续状态、每步更新、评分、读出、训练监督、可观察消融和残余混同；单独写明静态账目通过但 v6 实际时延失败为何与**资格**判定不同。若为 `FIELD_ONLY`，下一设计必须真正改变 K 更新因素或建立可辨识的因素分解，不能只将两臂都加相同四头四分数再换 mask（G1-DD/G1-SB 已否）。本审查本身不计 Gate 1 证据。
