# H1 图空间存在性：互反结构可与跨块边数局部分离

2026-09-24。按预写 `research/design/h1_reciprocity_feasibility.md`，只对 EXP-0009 冻结的十张 C0 图寻找一项局部双边交换 witness；不读取任务分数作筛选。所有输入掩码 SHA 与正式结果相同。完整十种子记录、交换边坐标、新旧 SHA 和指标见 `artifacts/analysis/h1_reciprocity_feasibility_v1.json`，确定性代码为 `scripts/witness_reciprocity_variation.py`。

**十张图均找到 witness（10/10，所需候选尝试1–17次）。** 每次交换保留 H64、1,024条边、逐节点入/出度16、64个自环、原分区跨块边数717–747、强连通性，故也保持 `Q=0.75−cross/1024`。去自环互反边比例每次改变约±0.002083，个别为−0.004167。交换是用独立预定 RNG 选第一项合格候选，没有按性能或所需变化方向择优。图的第二大绝对特征值也随交换变化，幅度约−0.0036到+0.0070，并未被匹配。

这个结果解决一个**存在性**问题：与 crossEdges 和固定划分 modularity 的精确耦合不同，互反边数至少可在这些度数/自环/跨块边约束下改变。它**没有**证明能生成足够宽、均匀、独立于谱/路径/motif 的互反变化分布；一次双边交换产生的互反改变量太小，不能拿来测试任务能力。该脚本不是成熟 null 模型，也没有训练网络、计算效应量或推进 H1/Gate 1。

独立审查 `docs/h1_h2_evidence_independent_review.md` 先发现默认仓库 `.venv` 的 `torch._C` 导入失败并保留失败记录；换用项目锁定的外部 venv 后，不调用原 `witness()`，按 JSON 删/加边坐标独立重建十张新图。逐点入/出度、自环、跨块数、双向强连通、互反比例、原/新 mask SHA 与第二特征值全部吻合。该复核加强存在性证据，但仍不把 witness 升级成 null ensemble。

若将来 H1 独立因子实验在治理上被允许，应先基于 `research/prior_art/topology_null_feasibility.md` 的成熟方法审查完整 reciprocal-degree/连通 null，并证明在目标变化幅度下其他图统计、参数和计算仍可匹配；实际主对照也须独立校准任务余量。当前决定仍是停止把 EXP-0004/0009 的 block/random 差异叫做单独的“模块性”或“桥接”效应。

任务后独立反思见 `research/reflections/H1-RECIPROCITY-20260924/`。反思者只读设计与上述原始 JSON 的 packet，指出变化幅度、谱共同变化和 packet 中缺少正式 mask SHA 对照。主代理保留前两项为未来 null 图族的准入缺口，并对第三项新增逐 seed 的 `input_identity_audit.json`：正式结果文件 SHA 与 witness 记录 SHA、正式 mask SHA 与重建原 mask SHA 均为 10/10 相等。原始 action JSON 和本任务窄结论未改动。
