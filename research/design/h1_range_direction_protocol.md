# DIR-0001：Gate 1 关闭时是否应继续互反图空间范围搜索

2026-09-24，在本方向独立审查和任何新图搜索前保存。唯一问题：现有十张 C0 图各有一次约0.002–0.004的合法互反变化，但 `research/design/h1_null_family_review_criteria.md` 又明确“当前不继续靠加图搜索来回避 H3 障碍”；在 H3-SLOT NO_SPEC、旧 M3 方向关闭且无活动实验的此刻，是否有充分科学理由做**无模型训练**的互反可达范围/谱联合可行性搜索？本审查只决定方法工作顺序，不运行图搜索、不改 Gate 1。

输入限定：`research/stage_gates.md`、`research/program.md`、`research/decisions/0004_h1_h2_evidence_scope.md`、`research/design/h1_null_family_review_criteria.md`、`research/design/h1_reciprocity_feasibility.md`、`research/evidence/h1_reciprocity_feasibility.md`、`research/decisions/0005_h3_identifiability_gate.md`、`research/decisions/0010_pf_preflight_branch_stop.md`、用户章程 Phase C/Gate 1 的持久摘要，以及 BCTpy/reciprocal-degree null 的既有一手审计。最强拒绝理由是用低价值图搜索制造“项目仍在进展”的错觉，悄悄越过“先有通用效应”的阶段顺序；强连通/度/cross 不变仍让谱、motif、路径等混杂，无法直接回答能力问题。

**GO 条件（须全部成立）：**

1. 该搜索只回答图空间方法可行性，并有事前数值目标、固定种子/预算、明确失败判据与可重放图坐标；不训练、不读任务分数、不声称 H1 因果或 Gate 1 变化。
2. 结果不论成功失败都能改变一个具体决定：例如是否继续保留“固定 degree/self-loop/cross/Q 下操纵 reciprocity”作为未来 null 家族候选；不是再找一个局部 witness。
3. 已说明成熟 directed degree-preserving rewiring / reciprocal-degree null 为什么不能直接同时保持本图族要求；自定义算法只作范围**可行性搜索**，不伪称均匀或成熟 null。
4. 当前不存在优先级更高、已登记且可运行的 Gate 1 工作被挤占；本方向结果不能被用作引入生物图/适应/通信/LM 的理由。

任一不成立则 `NO_GO_NOW`，归档有界停止理由，不通过移动 Gate 1 或改任务名称继续图训练。全部成立才允许另写事前固定的搜索规格，独立检查后再实施；本文件本身不授权运行。独立批评应特别核查条件2是否真的有决策后果，以及 GO 是否与现存“当前不继续加图搜索”自相矛盾。
