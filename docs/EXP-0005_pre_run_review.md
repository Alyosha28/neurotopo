# EXP-0005 运行前独立预注册审查

日期：2026-09-23。审查对象为 `experiments/EXP-0005/v2.md`、`configs/exp0005_v2.json`、检查点清单、`replay.py`、`replay_analysis.py`、`study.py`、`loop_worker.py` 和相关合成测试。没有查看或运行 EXP-0005 的新测试流；本审查不构成科学结果。

## V2 判定

**V2 预注册及实现的静态独立审查通过；正式运行仍须满足协议原定的 clean commit、源码哈希验证和完整检查点重放门槛。** V1 从未运行；V2 已消除前次审查指出的主要判定歧义。现有代码中 `SUPPORTED → PARTIALLY_SUPPORTED → FALSIFIED → INCONCLUSIVE` 顺序及数值阈值与 v2 协议一致：full 条件下 M3−C0 均值须 ≤−0.01，M3/C0 full 均值均在 (0.10,0.97)，两臂 blank 均值 >0.10，C0 blank 相对 full 退化不超过 0.03；支持要求 Δ≥0.03、种子层 CI 下界 >0 且 M3 blank 绝对收益 ≥0.03。显著但不足 3 点的正交互归 PARTIALLY_SUPPORTED，不再与 FALSIFIED 重叠。

前次静态核对发现分析器只信任 manifest 内嵌配置的问题，现已在运行器和分析器中修复：运行 manifest 记录配置路径及哈希；分析前核对配置文件字节哈希、解析内容与内嵌配置相同，实验 ID/版本一致，当前源码哈希与运行记录一致，并核对源 EXP-0004 manifest 哈希。任一不符即拒绝分析。新增合成篡改测试由主代理报告通过；本审查核对了相应代码路径，没有独立运行该测试。

## 已通过的静态核对

- 检查点清单有 10 个原种子 × 3 模型共 30 项；本次审查逐字节对比源 EXP-0004 final checkpoint，副本、大小及 SHA-256 均一致。运行器在加载时再验哈希、重建初值和掩码，并对 delay32/64 的原测试流逐臂要求旧数据哈希及全部 metrics 精确相等；`prediction_sha256` 包含于 metrics。主代理所述单种子重放不等于其余 29 项已执行，但逐臂正式门槛覆盖它们。
- `full` 仍由原 `make_batch` 生成。各条件用同一生成种子重建底层实例；`verify_variant` 核对标签、候选映射、事件标签、非干扰输入和反事实 pair 前缀。`repeat` 保持每干扰事件 ID+值的两个 one-hot，却改变时序相关性；`id_only`/`value_only` 等幅度。runner 保存逐条件输入/目标哈希、资源、预测及 pair 重叠，分析器核对三模型输入/目标一致与 M3/C0 资源匹配。
- 主交互先在每个原训练/图种子内计算，再对 10 个种子 bootstrap 20,000 次；代码与协议均没有把 1024 个 pair 当成独立训练复制。delay64、repeat 与单通道对比只作描述。一个新的固定测试流仍不能抵消本方向是在看过 EXP-0004 负结果后选出的事实；结果只支持这些冻结权重在固定实例上的输入敏感性，不能鉴别纯干扰与内在衰减，不能推进 Gate 1。
- 定向先行研究记录 `research/prior_art/interference_diagnostics.md` 现已存在，明确标出 blank 的输入幅度变化、repeat 的相关性 OOD 和解释上限；此处核对文件及其与协议的一致性，没有独立重查所引论文。

针对性 `pytest` 未在本审查环境运行：`uv run --frozen` 因默认缓存目录无权限失败，系统 Python 未安装 pytest。主代理报告新增合成分析测试 7 例通过；正式 `VALIDATED` 阶段的源码哈希匹配测试及逐臂检查点重放仍是运行器强制门槛。V1 的未运行协议保留在 `experiments/EXP-0005/v1.md`，其审查问题及修订见 v2 的 amendment trail；本文件以 v2 当前判定为准。
