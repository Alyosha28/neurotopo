# PA-0001 任务后补做的检索覆盖审计

2026-09-24。任务后反思指出原 packet 没有逐类证明预写的四类检索均已覆盖。本文件在原 packet **之后**形成；不回写原审计为“事前已完成”。下列是实际检索范围和补查结果，不是穷尽检索或“没有同类工作”的证明。

| 预写类别 | 实际检索式示例与核到的一手来源 | 未覆盖/限制 |
|---|---|---|
| ① 离散/多假设贝叶斯滤波和 MHT | `multiple hypothesis tracking original paper Reid 1979 ...`；读 [Reid 原论文](https://graphics.stanford.edu/courses/cs428-03-spring/Papers/readings/CollaborativeProcessing/Reid_MHT_ieee_trans_ac_1979.pdf) 与 [MH-iSAM2 作者页面](https://www.cs.cmu.edu/~kaess/pub/Hsiao19icra.html)。 | 没有穷尽 MHT 引文树；后证据能重估仍在分支，但已删分支不恢复原身份。 |
| ② 固定候选/多模态 belief 的神经方法 | `neural multi hypothesis tracking ...`、`mixture of recurrent experts ...`；读 [PF-RNN 原论文](https://ojs.aaai.org/index.php/AAAI/article/download/5952/5808)、[Belief Net](https://arxiv.org/html/2511.10571)、[Namikawa–Tani](https://arxiv.org/abs/0706.1317)。任务后补查发现 [Unified PF-LSTM 2026-09-02 预印本](https://arxiv.org/html/2609.01967v1)，它明确维护带权 recurrent-state hypotheses。 | 这些均非“同一固定身份槽低权重存活”的经验证模型；Unified PF-LSTM 的结果是作者报告，未复现。 |
| ③ 重采样与无重采样粒子/混合滤波 | 初次只记录 PF-RNN Table 3 的 **PF-LSTM** `NoResample` 消融，确实未在 packet 中提供独立方法来源。反思后追加 `"without resampling" particle filter`、`sequential importance sampling resampling` 等检索，并读 [Variational Marginal Particle Filters（AISTATS 2022）](https://proceedings.mlr.press/v151/lai22a.html)及其[原文](https://proceedings.mlr.press/v151/lai22a/lai22a.pdf)对 VSMC 无重采样变体的讨论。 | VMPF 研究变分状态空间推断/梯度，不是 H3-SLOT 任务中的固定身份槽恢复；本轮没有找到可直接替代 PF-GRU 无重采样的已验证实现。初次 packet 的第三类覆盖不完整，已明示偏离。 |
| ④ 作者明确关联的开源实现 | PF-RNN 原论文链接的[作者仓库固定提交](https://github.com/Yusufma03/pfrnns/tree/0a76d52a857f06d4128bbc7601c616d714b25734)在既有审计中已只读检查。本轮在 [Belief Net 原文](https://arxiv.org/html/2511.10571)找到明确关联的[作者 GitHub 仓库](https://github.com/Rockostoneo/BeliefNet-HMM)，只核了 README 的论文对应关系。 | Belief Net 源码没有固定提交、查许可/测试或运行；Unified PF-LSTM 的已读 arXiv HTML 没有明确代码链接；Tani/MH-iSAM2 作者实现未确认。初次 packet 没有完整交代第四类覆盖，已明示偏离。 |

补查使一项边界更明确：近期 Unified PF-LSTM 已在部分观测事件日志中使用“weighted recurrent-state hypotheses”，强化宽泛多候选语义的先例；它仍不证明 H3-SLOT 同一固定身份槽的机制等价。矩阵已增该预印本，并将 Belief Net 的代码状态从“未知”收窄为“作者仓库已确认、未审计实现”。所有新发现均作为**任务后补证**，不能用于追认原 packet 的完整覆盖。
