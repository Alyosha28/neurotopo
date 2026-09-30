# EXP0022：等当前输出与未来状态价值

2026-09-28定向核查，研究设计与结果之前完成。搜索词：adversarial perturbations hidden state recurrent neural network future predictions；predictive state representations dynamical systems future tests。原始来源：

- Littman/Sutton/Singh, Predictive Representations of State, NIPS2001，官方论文https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf 。读摘要及状态/未来tests定义：状态可由多步动作条件的未来观测预测表示。已存在“用未来区分状态”的思想，不能将当前读出与未来预测的差异当新原理。
- Deka/Stipanovic/Tomlin, Dynamically Computing Adversarial Perturbations for Recurrent Neural Networks, arXiv2009.02874v1，https://arxiv.org/abs/2009.02874 。读作者摘要及版本元数据：输入扰动、动力学控制、鲁棒余量已有研究。这里是冻结hidden的一次干预，固定当前logit，不声称首次RNN压力测试或全局最小攻击。
- 金融启发来自Hansen/Sargent鲁棒控制及Brunnermeier/Pedersen市场/资金流动性反馈；仅概念动机，本轮不测金融市场、资产定价或交易表现。

对照：NONE、同幅随机候选、保持全部Gram的共同旋转SHAM；主对象INV根据已知稳健性事前选定，RAW/REL完整保留。16有限候选×4搜索后缀选扰动，16独立测试后缀检验迁移。无作者代码复用；commit/license/tests/maintenance均NOT_INSPECTED，本地独立从二次型代数实现。候选族保留读出特征基中的列范数，因此比完整等输出集合更窄，阴性不能排除其它方向。单独当前logit非递归充分状态通常不难构造，本研究价值在量化已训练模型的局部敏感性和控制差异，不由术语保证贡献。
