# STR-0002 反思后的运行入口验证协议

2026-09-24。在实现入口校验或运行 dry-run 前保存；这是对 packet-only 反思 R1–R5 的行动规格，不是新实验或 Gate 1 科学规则变更。现有 `ResearchLoop` 在 ITER-0007 `DECIDED` 且反思完成；旧迭代不可改写。目标是让“候选优先只针对项目 Gate 1，独立 benchmark 有单独入口”从文书变成**可执行的 QUESTION 阶段结构校验**。结构通过仍需 PRIOR_ART/CRITIQUE 的独立科学审查，不能当实验许可。

新增版本化 `entry_contract.json`（JSON）必含：`schema_version=1`、`track`、`experiment_id`、`direction`、`question`、独立 `purpose`、资源上限 `max_updates/max_wall_seconds`、`task_search.max_configurations/retain_all_attempts/heldout_rule` 及 `branches.success/failure/borderline`。`track=project_gate1` 必须给具名 H3 候选因素的操作、消融、方向性预测、实际主控制、状态/参数/执行计算匹配和先例边界；无因素则拒绝。项目参考 K 因子的归属在 STR-0001 仍有歧义，入口应返回明确的 **PROJECT_ELIGIBILITY_UNRESOLVED**，不是声称永久禁止；若有独立领域问题，使用 `field_reference` 且 `gate1_claim=false`，结构上可进入。`independent_benchmark` 不需要候选，但必须有独立目的、资源/任务搜索预算与三分支决定，且 `gate1_claim=false`。所有文字字段不能为空；明显 boilerplate 不由 JSON 校验证明有科学意义。

在现有 `ResearchLoop` 增加**显式一次性启用**的 `gate1_candidate_first_v1` 入口策略事件。只有当前迭代 `DECIDED` 且反思已归档时可启用；这只影响**之后**的 `begin`，不重写先前日志。启用后 `begin` 必须附 manifest 路径，先校验 `experiment_id/direction/question` 与命令完全一致、结构与 track，再将 manifest SHA/路径写入新 `QUESTION` 记录；失败不写事件或改变迭代号。未启用的临时测试/历史账本维持旧接口。CLI 提供 `enable-entry-policy` 和 `begin --entry-contract`；工作流文档明确先用 manifest，再 begin。入口策略不决定候选的真实新颖性、资源公平或任务质量，仍由后续独立批评和正式注册检查。

事前 dry-run 验收：A) `project_gate1` 缺候选的完整其他字段申请须被拒、日志不增长；B) `independent_benchmark` 有独立用途、上限、保留所有尝试和三分支的申请须进入 `QUESTION`，但不能自称 Gate 1；C) 参考 K 因子若申报 `project_gate1` 应明确归属未决，若以 `field_reference` 且有操纵/消融/主控制则可进入领域问题。对每例记录输入 SHA、返回结果/拒绝原因和 ledger 变化，不能只报测试绿色。再运行完整测试套件和当前 live 状态迁移演练；只有这两组检查通过才对 live ledger 启用。若发现校验误挡独立研究或让缺因素项目申请通过，修复实现，不降低科学门槛或改变旧决定。
