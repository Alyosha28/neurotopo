# 定向核查：现有有向图 null 是否能解决 H1 混杂

2026-09-24。现存 EXP-0004/0009 的 M3/C0 图对照同时改变跨块边、固定分区 modularity、互反边、连通性和谱混合。此处只评估成熟工具能支持什么，不引入依赖、不生成新科学对照、不声称完成网络神经科学 null 方法综述。

[BCTpy 官方函数索引](https://github.com/aestrivex/bctpy/blob/master/docs/bct.rst)列出 `randmio_dir`、`randmio_dir_connected`、`modularity_dir` 和有向 motif 工具，适合作为将来度保持重连及图指标的首选审计来源。其[官方 README](https://github.com/aestrivex/bctpy)注明 GPL-3.0、NumPy/SciPy 依赖，并坦言测试覆盖不完整、项目仍应视作 beta。仅看函数名和文档不能保证它保留本任务的64个自环、原四块间边数、互反结构或特定谱值；真正采用前需要固定版本、检查函数源码/测试并做确定性不变量验证。本项目目前没有安装或运行 BCTpy。

[Durak 等的 directed reciprocal-degree null](https://arxiv.org/abs/1210.5288)提出匹配有向网络入度、出度及互反度**分布**的随机模型，说明互反结构确实是已有 null 方法明确控制的维度。但分布匹配不自动等于逐节点精确度数、自环及原分区约束匹配；不能把它直接替换 EXP-0004/0009 的主 null，也没有核对其实现许可和测试。更重要的是，在本项目当前等大四组、逐节点固定入/出度16的图族，`Q=0.75−crossEdges/1024` 是精确恒等式；任何工具都不可能在**同一分区和这些约束**下独立改变 crossEdges 而把这个 Q 保持不变。

因此可行的新 H1 因素必须重新定义：例如在固定跨块边数/度数/自环条件下考察**互反结构是否还能独立变化**。随后完成的 `research/evidence/h1_reciprocity_feasibility.md` 在十张冻结 C0 图上各找到一次局部合法交换，证明这一窄义图空间存在性；它没有证明可取得足够大的变化范围、谱等协变量匹配或可用 null 分布。完整图族仍须按 `research/design/h1_null_family_review_criteria.md` 审查成熟方法、目标强度和联合匹配，再考虑受阶段门约束的任务训练；当前没有为下一次图训练注册问题。
