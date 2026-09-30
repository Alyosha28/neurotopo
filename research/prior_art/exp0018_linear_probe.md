# EXP0018 线性probe解释边界

2026-09-28，读取已有`representation_probes.md`与EXP0006v2冻结probe方法；外部核对原始会议摘要：

- [Hewitt & Liang 2019](https://aclanthology.org/D19-1275/)以control tasks区分任务可预测性和probe自身学习能力。本文只借其控制原则，fit标签置换不是其按词类型赋随机标签的等价实现。
- [Pimentel等2020](https://aclanthology.org/2020.acl-main.420/)讨论信息论探测；限制probe函数类的失败不能证明表示中没有信息。本文不估计互信息，不作信息不存在判词。

本轮没有新架构，只测最终hidden到已知b标签的线性可读性。原CO任务输入端sensor只直接提示j，不能据分量差异断言mod2的固有难度。读出成功也不证明原模型在其任务计算中因果使用该信息。外部来源仅摘要层核对，未复现其论文、未复用外部代码。实际source工具与统计在本库实现，冻结正则，无test调参。
