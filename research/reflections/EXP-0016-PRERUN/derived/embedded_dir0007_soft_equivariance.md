# DIR-0007 定向先例核查

2026-09-26。采用 prior-art-audit 流程。检索范围：equivariant recurrent networks、parameter sharing、HashedNets、soft equivariance、regularization。读取原始会议论文/作者摘要；非系统综述。未下载或复用代码，未核固定代码提交、许可证或测试，未复现作者结果。下述“未确认”不等于论文没有做。

| 原始来源与阅读范围 | 已确认重叠 | 本设计边界 |
|---|---|---|
| [Ravanbakhsh et al., ICML 2017](https://proceedings.mlr.press/v70/ravanbakhsh17a/ravanbakhsh17a.pdf)，摘要与参数对称性方法 | 参数共享与离散群等变性的关系及结构化共享已有理论。 | 群投影/权重共享不新。代数复核验证本地公式，不是首创证明。 |
| [Finzi et al., NeurIPS 2021](https://proceedings.nips.cc/paper_files/paper/2021/hash/fc394e9935fbd62c8aedc372464e1965-Abstract.html)，会议作者摘要 | RPP 用软先验替代硬架构约束，在偏好结构解时保留额外表达能力。 | ORBIT-SOFT “完整表达类＋结构偏好”的核心直觉已有直接先例。本文未核其正文实验细节；不称本协议精确复现 RPP。 |
| [Kim et al., ICML 2023](https://proceedings.mlr.press/v202/kim23p/kim23p.pdf)，摘要、§1–2 | 对无硬限制模型加入等变正则；研究混合近似对称性，并自动调节强度。 | 软等变正则也不新。本协议固定强度、禁自动调节，只检验共轭先验的方向；不是优于该论文的主张。 |
| [Flow Equivariant RNN, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/e637029c42aa593850eeebf46616444d-Paper-Conference.pdf)，检索命中、论文可访问 | 循环网络的群/flow 等变已经是专门研究方向。 | 本轮未充分阅读46页正文，不据题名主张某个具体循环实现相同；不声称此领域新颖性。 |
| [Chen et al., ICML 2015](https://proceedings.mlr.press/v37/chenc15.html)，作者摘要 | HashedNets 使用哈希分桶共享权重以压缩参数。 | 随机共享有先例，但它不保证与结构共享同函数类，故不采用其作为唯一主控制。 |

最近邻：RPP/软等变正则；最强常规控制：无该正则的同型全 GRU 与 H64 全 GRU；先例控制：硬等变 GRU；无架构控制：学全转移的 HMM 与只诊断的已知参数 forward oracle。没有新增可复用依赖。

可检验增量是实验规格：同函数类、同强度同谱的共轭先验，对任务坐标系的对齐是否带来最终水平差。它目前**不是已证未被先例覆盖的贡献**。最强未解决反驳仍是“这是已知归纳偏置的实验，是否属于项目 Gate 1 可计因子”，必须前瞻裁定；不能用文献检索未见同一表格作新颖性证明。
