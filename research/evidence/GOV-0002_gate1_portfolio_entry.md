# GOV-0002：目前没有可注册的项目 Gate 1 新实验组合

2026-09-24。按事前提交 `065f3c7` 的只读协议，检查当前仓库而不训练、不调任务、不改历史阈值。标签 **`NO_REGISTERABLE_PROJECT_GATE1_EXP_NOW`** 指**现存材料不足以注册下一项项目 Gate 1 EXP**，不是多轨迹机制一般无效、项目永久停止或独立领域研究被禁。当前 `ResearchLoop` 是 ITER-0007 `DECIDED`、已反思，`gate1_candidate_first_v1` 已启用；`research/entries/` 目前无真实项目合同，测试 fixtures 不计。

| 现存组合 | 可辨操纵与语义 | 任务/oracle 与实际主控制 | 资源、先例和正式结果 | 项目新 EXP 入口 |
|---|---|---|---|---|
| **M3 四个全输入独立 GRU 块** | 代码有 block mask，但各块读同一输入、共享读出，没有候选假设身份、权重或路由；“块即竞争假设”不能推论。 | EXP-0004 的随机掩码控制在 OOD32 有32.96百分点至100%的表面余量，双方状态/度/自环/有效参数/稠密执行代理匹配；但该任务的四绑定+晚期直接 selector 可缓存检索，不要求竞争解释。 | 预写最低+3pp结构转移收益在该任务 **FALSIFIED**：M3 比主控制低9.10pp。决定0003关闭用同一块代理再调任务难度救正效应。 | **否，限旧 M3 的项目 H3 方向。** 旧任务可以在另一个具名记忆/检索问题中重新注册，但不能追认竞争假设语义或改 EXP-0004 原判。 |
| **H3-SLOT 固定身份加权槽** | SCR-0001 `NO_SPEC`、SCR-0002 `NO_DISTINCT_SPEC`；身份/初值/完整更新、语义保持与任务输出/失败消融尚缺。槽索引存活不等于同一解释存活。 | 无该候选实际主匹配控制、独立 headroom 或合格任务；静态已知假设的精确滤波是简单解。 | PF-RNN 带权粒子、RIMs 模块及显式滤波已覆盖宽机制/行为；待建 `PFGRU-R0` 与自有 `PFSTYLE-ID0` 均尚未实现验证，不能作为已匹配数值基线。 | **否。** 零训练候选合同未满足，不能先训练后命名贡献。 |
| **参考 PF/RIM 的 K/路由因素** | 作为领域问题可提出，但具体 K 操纵、重采样/路由因子化和消融尚无正式新合同。 | 没有针对该因素的真实主控制、任务/数据流或余量资格。 | 决定0013指出整套 PF-GRU 胜 GRU 只属架构包效应；决定0019/0023让 `reference_k` 的**项目** Gate 1 计分资格保持未决，结构入口只准 `field_reference` 轨。作者原 `AUTH-PFGRU` 每步重采样；待建 `PFGRU-R0` 仅拟去重采样，自有 `PFSTYLE-ID0` 另加身份/上下文/去噪声，不能混称一个参考。 | **否，限当前项目 Gate 1 规则。** 可以另立有界领域研究，结果不自动过项目门；不能把参考 K 科学价值判为零。 |
| **CAL-0004 varied 分布 → EXP-0009 M3/C0** | 仍是上述 M3 块代理。 | CAL-0004 只把稠密背景臂的 d4 KL 校到约0.018–0.023；正式实际主 C0 的唯一 d4 端点 KL均值仅0.012463，低于预写≥0.07余量门。 | EXP-0009 十配对种子资源/数据匹配、结果 `INCONCLUSIVE`；d8 是次级，不能事后升级。 | **否。** 不得把已失败的主端点换为 d8 或重调干扰长度。 |
| **CAL-0006 四模式动态滤波任务** | 任务有候选模式轨迹的解析 oracle，但**没有**配套 H3 候选/参考因子干预；四个2D状态+四 log 权重是小显式充分状态上界。 | Dense/random H64 单流都学会 IID8；OOD12 全六项≥0.08资格失败，dense seed603 KL0.06497，另五项在窗。自然修订切片0.09819不是主门。 | 只属一次性领域 benchmark 筛查 `CEILING`，没有多轨迹模型效果；决定0021关闭此冻结配置。 | **否。** 不能用次级切片或调整0.08重开同规格。 |
| **H1 互反图局部 witness** | 十图各有合法小交换，但互反、谱、motif等仍共变；没有完整匹配 null。 | 无已登记 H1 能力主端点与合格控制；不回答项目 Gate 1 的 K 因素问题。 | 决定0014/0016 的图搜索/上界均未授权拓扑训练或生物图推论。 | **否，属另一因果链。** 不把图存在性 witness 当多轨迹收益。 |

上述行不是候选空间穷尽证明；它们仅列当前仓库有名字、有来源的组合。**共同缺口在操纵因素/解释身份及其实际主控制**，不是“尚未把任务调到刚好不封顶”：有余量的 EXP-0004 已在旧 M3 上给出负效应；后验和动态任务分别出现近 oracle 或边界余量。当前再做候选缺席的 Gate 1 headroom CAL，成功也不能指定应比较谁，失败只关闭规格，故不能代替机制合同。

| 组合与 claim track | 逐项可核文件 | 已有资源账与**现缺材料** |
|---|---|---|
| M3：旧实验是项目 Gate 1 代理；新 `project_gate1` 入口需另一 H3 因素。 | `src/neurotopo/models.py`；`experiments/EXP-0004/v1.md`；`research/evidence/gate1_benchmark_baseline_audit.md`；`artifacts/EXP-0004/v1/analysis.json`；决定0003。 | **已有旧实验匹配：** H64、有效/分配参数6,212/15,428、度/自环及稠密执行代理相等；**缺：**候选特异身份/权重/路由与新的方向预测。旧匹配不替未来新模型配账。 |
| H3-SLOT：只有拟议 `project_gate1` 候选，未获科学/结构入口。 | `research/evidence/h3_slot_candidate_screen.md`；`research/evidence/SCR-0002_noresample_slot_equivalence.md`；决定0007/0024/0025；`research/design/h3_candidate_trajectory_contract.md`。 | **仅有符号账** `dim(C)+K dim(Z)+K`；**缺：**身份/完整更新、数值任务效应、真实主控制及有效/分配参数、实际执行计算。 |
| PF/RIM 参考因素：`field_reference` 可设计；`project_gate1` 对 `reference_k` 现返回归属未决。 | `research/prior_art/particle_filter_rnns.md`；`research/reflections/SCR-0002/reference_disambiguation.md`；决定0013/0019/0023；`src/neurotopo/entry_policy.py`。 | **已有先例定性成本项**粒子×宽、评分/重采样/路由；**缺：**本地参考 cell、K 因素消融、实际主控制，故无已核状态/有效/分配参数/实际 FLOPs 数值。 |
| CAL-0004→EXP-0009：旧 M3 项目代理的已完成正式实验。 | `research/evidence/CAL-0004_cadence_qualification.md`；`experiments/EXP-0009/v1.md`；`research/evidence/exp0009_posterior_transfer.md`；`artifacts/EXP-0009/v1/analysis.json`。 | **已有旧实验匹配：** M3/C0 总 H64、有效/分配参数5,704/14,920、度/自环和稠密执行代理、样本/步数相同；**缺：**预写主 d4 控制余量，且没有新 H3 身份机制。 |
| CAL-0006：独立领域 benchmark 屏幕，未获项目 Gate 1 候选。 | `research/design/CAL-0006_dynamic_counterfactual_qualification_v3.md`；`research/evidence/CAL-0006_dynamic_filter_qualification.md`；`artifacts/calibration/CAL-0006_v3.json`；决定0021。 | **已有屏幕账：** H64、分配参数13,700各臂、有效参数dense13,700/random4,484、同稠密执行代理；**缺：**多轨迹候选/实际匹配主控制，且六臂 OOD12 全过门余量失败。 |
| H1 witness：当前不属于 Gate 1 多轨迹 claim track。 | `research/decisions/0004_h1_h2_evidence_scope.md`、0014、0016；`research/design/h1_method_reentry_table.md`。 | **已有：**局部保度/自环/cross合法交换；**缺：**完整匹配 null、谱/motif/路径容差、合格任务能力主端点及新模型状态/参数/执行计算；不能转成 Gate 1 K 因素账。 |

按**现行项目入口规则及当前仓库材料**判 `NO_REGISTERABLE_PROJECT_GATE1_EXP_NOW`。对现有 H3 代理，项目 Gate 1 分支目前只进入**机制定义**：出现一份版本化、非旧 SCR-0001 拼贴的 H3 更新/身份/读出与数值化可消融反事实、具名实际主控制、显式滤波/PF/RIM 边界及状态/参数/实际执行计算方案，经独立批评后，才有资格另行冻结有限任务菜单并校准该控制。EXP-0004 的旧任务可服务于**另一个**预先具名的检索/记忆主张，但原任务/模型结果不能改写成 H3 语义。若走参考 K 因素的项目轨，必须先前瞻解决决定0019/0023留下的项目计分资格；独立 `field_reference`/`independent_benchmark` 轨保持开放且不得冒充项目效果。任务资格、支持性测试或结构 JSON 通过都不自动推 Gate 1。既有 EXP-0004 `FALSIFIED`、EXP-0009 `INCONCLUSIVE`、CAL-0006 `CEILING`、H1/H2 `UNTESTED` 与 Gate 1 `NOT_PASSED` 全部不改。
