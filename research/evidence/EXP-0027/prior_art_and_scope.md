# EXP0027 文献与范围

本轮以状态送达后对未来命令的正确性为检验对象。已核查Maheswaranathan等NeurIPS2019原始摘要（https://papers.neurips.cc/paper_files/paper/2019/hash/5f5d472067f77b5c88f69f1bcfda1e08-Abstract.html）：报告不同RNN的几何相似性与计算结构可能分离，不意味着本项目状态功能已识别。已核查Koul/Greydanus/Fern的Learning Finite State Representations of Recurrent Policy Networks（https://arxiv.org/abs/1811.12530）摘要：QBN insertion学习有限内部/观测表示，提供有限state解释先例；本轮不实现或复制其代码，不引用其数值为本项目结果。

Wang/Niepert状态正则RNN、Sussillo/Barak固定与慢点方法、Mante情境计算已在crossfield_research_practice_audit.md记录。这里的FULL与EQUAL必须按实际源代码区分：FULL固定二原型覆盖H；EQUAL保留连续尺度并可能无效回退。没有额外RNN缓存或门控状态；调度器时刻属于运行上下文，固定周期区块边界才能用有限两态拼接推论。当前二分类读出是单个logit z，因此真实裕度为(2y−1)z。

主要对象是已暴露模型的干预兼容性和短反例。INV能同时保持/更新，REL现象不构成任务必然两难。未做EXP0028方法改进或EXP0029新seed验证；本轮的SUPPORTED（如有）只针对具体状态与命令反例，不是可推广机制或算法优势。
