# PA-0001 独立先例审查

2026-09-24。独立 `gpt-6-sol` / high / `fork_turns=none` 只读 PA-0001 预写设计、先例审计、新颖性边界、证据矩阵及其中的一手论文/作者页面；未读对话、未编辑或运行项目。主代理接受其最强批评：`DIRECT_OVERLAP` 只能标记**行为命题**，不能把 Reid 的可剪枝假设树、PF-RNN 的随机粒子或 Belief Net 的 HMM 状态递推说成固定身份 recurrent 槽的架构等价。

逐项核查结果：

- [Reid 原论文](https://graphics.stanford.edu/courses/cs428-03-spring/Papers/readings/CollaborativeProcessing/Reid_MHT_ieee_trans_ac_1979.pdf)允许后续测量重估**仍保留**的分支，但也删除低概率分支、合并相似分支；删掉的原分支不会以原身份等待恢复。
- [PF-RNN 原论文](https://ojs.aaai.org/index.php/AAAI/article/download/5952/5808) Figure 6 是一条 PF-LSTM 定位轨迹中整体信念从错误位置恢复的示例，不能证明同一低权重粒子存活；Table 3 的 `NoResample` 是 **PF-LSTM** 消融，不是 PF-GRU 的现成数值。
- [Belief Net 原文](https://arxiv.org/html/2511.10571)递推有限 HMM 隐状态后验，隐状态可转移；它支持精确/受界滤波作为简单基线，不等于固定身份轨迹或每槽 GRU。

主代理据此把 `research/prior_art/fixed_slot_hypothesis_retention.md` 的判定拆为：多解释与后证据修订 `DIRECT_OVERLAP`，同一低权重固定身份槽恢复的完整机制只 `PARTIAL_OVERLAP`。同步修订新颖性边界、必要基线、开放问题和证据矩阵的纸面/结构化记录。仍未找到新颖性证明或任务效应；所有外部性能叙述只是作者报告，本项目未复现。
