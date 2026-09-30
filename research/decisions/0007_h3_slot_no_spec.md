# 决定：H3-SLOT 未形成可注册的独有预测

2026-09-24。按事前保存的 `research/design/h3_slot_candidate_screen_protocol.md`，主代理只对一个无适应、无重采样、无跨候选通信的身份条件化多槽候选执行零训练筛查；结果与独立审查见 `research/evidence/h3_slot_candidate_screen.md`、`docs/h3_slot_screen_independent_review.md`。判定为 **NO_SPEC**，这是候选规格筛查标签，不是 EXP 的四分类科学结果。

受限方程能写出 `C/Z/l/w` 的示意递推，但没有具体身份编码、初值、训练损失或合格任务。对有限已知静态假设，正确 logit 证据累加可由解析滤波完成；PF-RNN 已有带权潜在粒子及观测评分，RIMs 已有选择性模块。固定槽保留低权重假设可能相对重采样 PF-GRU 有晚到反证恢复优势，但显式滤波也能做到，且目前没有主终点、最小有用阈值、资源匹配或失败消融。故不满足预写的 `DISTINCT_SPEC` 条件。

维持 `research/decisions/0005_h3_identifiability_gate.md` 对此前**原始、欠定义 H3 草案**直接实现的条件性 NO-GO；SCR-0001 自身仅排除**H3-SLOT 这一受限规格**按当前定义实施，不对未筛查的其他候选作判定。不写 H3-SLOT 模型、不把它交给新 benchmark 选择，也不以这份纸面审计改变 Gate 1。NO_SPEC 不意味着所有多候选机制都失败；保留被列明的未覆盖反事实作为未来新问题的素材，而非当前机制许可。
