# 决定：互反计数上界不排除目标，也不证明可达

2026-09-24。BND-0001 的目标 +0.10、十图输入、两层上界和停止规则在 `8a65f06` 事前提交；结果见 `research/evidence/h1_reciprocity_count_bound.md` 与 `artifacts/analysis/h1_reciprocity_count_bound_v1.json`，独立复核见 `docs/bnd0001_independent_review.md`。按预写规则判 **BOUND_NONDECISIVE**：十图原互反比例约0.235–0.269，目标约0.335–0.369；仅固定总跨块数的上界约0.998–1，额外固定完整块矩阵的上界仍约0.95–0.985，0/10 图的任一上界排除目标。

该结果只说明粗边数/块计数必要条件太松，不证明存在能在逐点度、自环、cross、强连通及谱/motif/路径约束下达到目标的图。若要证明可达，须另有构造或有界搜索；若要做 H1 因果任务比较，还须成熟 null/协变量控制和合格 benchmark。本轮按协议**停在上界审计**，不延伸为新的随机交换搜索、不训练模型、不改变 `LOCAL_WITNESS_ONLY` 候选状态。

H1/H2 仍 UNTESTED；Gate 1 仍 NOT_PASSED。探索目标 +0.10 的选取是方法对比尺度而非能力效应阈值，不能被论文叙述成“有用拓扑差异”已验证。
