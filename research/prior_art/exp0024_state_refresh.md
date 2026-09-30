# EXP0024先例与解释边界

2026-09-28。本轮复用已核查的原始文献边界，不声明检索覆盖全部领域。

- Wang/Niepert, State-Regularized Recurrent Neural Networks, https://arxiv.org/abs/1901.08817 ：前轮读作者摘要，有限可学习状态及状态正则已有先例。当前只做已有模型的后训练监督原型重整，不复现其随机状态机制。
- 本项目research/prior_art/dir0009_protected_memory.md已核查经典码本投影与decode/re-encode替代解释：原型重整不是新原理；现在的实证对象是已学习但长程失败的模型。
- Littman/Sutton/Singh, Predictive Representations of State, https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf ：本会话已核查动作条件未来预测定义，当前logit不应被默认视为递归充分状态。

本轮不复制外部代码；许可/commit/维护/作者测试均NOT_INSPECTED。最强基线为原模型、保Gram坐标重整及稳定INV。校准目标用于两个原型拟合，测试时只有模型自己的当前预测选原型；这额外引入监督与两状态先验，成功只能支持有限病例的干预见证。原有失败seed全部保留，不能声称无选择偏差或机制唯一性。
