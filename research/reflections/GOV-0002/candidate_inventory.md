# GOV-0002 反思后的非 JSON 候选与任务清单

2026-09-24。响应 packet-only R2。检索范围是本项目 `research/design/`、`research/evidence/`、`research/decisions/`、`experiments/`、`configs/` 与 `src/neurotopo/models.py` 中已具名的机制/任务/因素；使用 `rg --files` 与 `rg -n 'H3-SLOT|DARC|PF-GRU|RIM|multi_gru|candidate|trajectory' ...` 搜候选词，再按冻结协议/决定去重。`research/entries/` 当前不存在，测试/反思 dry-run JSON 均标为合成例，不计真实候选。本清单不声称外部文献或所有可能新机制穷尽。

| 具名材料及一手本地位置 | 因素/候选更新 | 任务及 oracle | 真实主控制/余量 | 数值主张/消融 | 状态、参数、执行计算 |
|---|---|---|---|---|---|
| `src/neurotopo/models.py` M3 block mask；`experiments/EXP-0004/v1.md`、`EXP-0009/v1.md`；决定0003 | **已有**固定四块全输入 GRU；**无**候选身份/权重/路由新更新。 | EXP-0004 查找任务已有非封顶 OOD32，但可缓存；EXP-0009 后验任务有 oracle，唯一 d4 主控制余量不足。 | EXP-0004 M3/C0 旧控制已匹配且 M3−C0−9.10pp；EXP-0009 C0 d4 KL0.012463<0.07。**无新因素的主控制。** | 旧 EXP-0004 ≥3pp 预测已 FALSIFIED；EXP-0009 有用>0.05nat 因余量门 INCONCLUSIVE。**无新注册预测/身份消融。** | **已有旧实验账**，不自动匹配一个未来候选。 |
| `research/evidence/h3_slot_candidate_screen.md`、`SCR-0002_noresample_slot_equivalence.md`、决定0007/0024/0025 | 仅示意 `C,Z^k,l^k,e_k`；身份、初值、完整 f/score/readout 未定；低权重恢复非独有。 | **无**已指派给本候选的生成器/主端点；静态已知 H 的精确滤波为简单解。 | **无**真实匹配 PFGRU-R0/单流主控制与独立余量。 | `NO_SPEC`/`NO_DISTINCT_SPEC`；**无**方向 δ、失败身份消融。 | 仅符号状态 `dim(C)+Kdim(Z)+K`；**无**数值有效/分配参数与执行计算。 |
| `research/prior_art/particle_filter_rnns.md`、`fixed_slot_hypothesis_retention.md`；决定0013/0019/0023 | AUTH-PFGRU/RIM 是既有参考机制；PFGRU-R0 与 PFSTYLE-ID0 尚未在本项目实现；参考 K 项目资格未定。 | **无**对具名参考 K 因子前瞻冻结的新任务/最终流。 | **无**本地 K 因素实际主控制资格。 | PF/RIM 作者结果非本地注册因素效应；**无**项目 K 因子数值消融。 | 已知需核粒子×宽、权重、评分、BN、路由/重采样等；**无**本地等状态/参数/实际计算账。可走 `field_reference`，不能直接过项目门。 |
| `research/hypotheses.md`、`research/program.md` 中 adaptation、communication、DARC | 仅后续机制候选集合，非当前已实现 Q1 因素；Gate2+ 前置条件未满足。 | **无**当前合格 Q1 任务/对照。 | **无。** | **无**当前数值化 Gate1 消融；不能加后续机制补旧失败。 | **无**当前匹配账。 |
| `research/design/h1_reciprocity_feasibility.md`、`h1_method_reentry_table.md`；决定0004/0014/0016 | 局部互反边 swap witness，不是多假设 K 操纵。 | **无**匹配 null 的 H1 能力任务；生物图阶段未进入。 | **无**H1 能力主控制。 | `LOCAL_WITNESS_ONLY`/`BOUND_NONDECISIVE`，不是性能效应。 | 图约束局部检查已有；谱/motif/路径/null 与训练模型资源未齐。 |

任务库存并未被伪装为候选：CAL-0001/0002 是小静态精确后验天花板；CAL-0003/0004 是训练条件校准；CAL-0005 是固定 d2 匹配设计 NO-GO；CAL-0006 是四模式动态任务的一次性 `CEILING` 屏幕。EXP-0005–0008 是冻结 EXP-0004 权重上的干预/探针诊断，不引入新的 Gate 1 候选更新。所有正式 EXP-0001–0009 及对应任务/配置在 `experiments/README.md` 列出；其存在不等于同一新项目入口已有完整五项材料。

这份清单支持的是**当前有名文件中尚无一行完整组合**，不是“任何未想出的模型都不能工作”。若发现一个漏列的、已经版本化且五项齐全的真实组合，GOV-0002 的当前可注册性标签须重新审核，而旧实验判词仍不追改。
