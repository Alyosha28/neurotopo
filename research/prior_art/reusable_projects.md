# 可复用项目审计

2026-09-28 DIR0009本轮只核对原始论文/作者材料，未检查或复制其代码，代码许可/固定提交/测试均NOT_INSPECTED；未增加依赖。量子类比与经典不变量/码字投影只作方法背景，当前可训练设计以本地小原型为目标。

2026-09-28新增[DIR-0008固定提交审计](dir0008_state_update_research.md#5-代码可复用性核对)：DeltaProduct作者库MIT（根与state_tracking分别核），Unlocking子树MIT但非根统一许可判断，NVlabs/GatedDeltaNet为Nvidia Source Code License-NC。源文件hash存`research/evidence/DIR-0008_sources.json`。均只读、未执行、未复用；Triton依赖和no_triton文件的算子导入使Mac CPU可运行性仍未验证。不得把存在测试文件写成测试通过。

决策以主问题为准：训练循环状态的 BPTT，比多种 recurrent 架构在证据修订任务上进行预算匹配比较。下列仓库仅做源代码评估；没有复制、导入、安装或运行外部代码。

| 项目 | 源码与许可 | 与首个实验的适配度 | 决策 |
|---|---|---|---|
| [conn2res](https://github.com/netneurolab/conn2res/tree/3ccb7074261c910847dcd0164b00ff3b02bade90) | BSD-3-Clause。检查 `reservoir.py`、`readout.py`、`tasks.py`、测试目录；固定递归权重，读出通过 sklearn 线性模型拟合。 | 有多种 NeuroGym 延迟任务，值得用于任务边界核对；训练协议是 reservoir + 线性读出，不是要求的训练循环核 BPTT。README 还列出较旧 Gym/NeuroGym 固定版本。 | 不作为主项目依赖或同协议基线；不复制代码。保留论文和任务目录作为 prior-art 参考。 |
| [BCTpy](https://github.com/aestrivex/bctpy) | 官方文档列 `randmio_dir_connected`、有向 modularity/motif 等函数；README 为 GPL-3.0，并提示测试不完整和 beta 状态。当前只读文档/仓库说明，未安装或固定代码提交。 | 可用于将来 H1 图指标及度保持 null 审计，但不能自动保持本任务自环、跨块边或同时独立操纵与跨块边精确耦合的固定分区 modularity。 | 暂不加依赖；若 H1 图因子可行性成立，再固定提交、查具体函数与测试并验证全部图不变量。见 `topology_null_feasibility.md`。 |
| [Transient](https://github.com/jiashuncheng/Transient/tree/81eee3611b6f89b2eecb9ffac3cfabb2b7525691) | GitHub API 未报告许可证。检查 `ODPA/new/model_torch.py`、`RL/examples/trnn.py`、README、`RL/test.py`。代码面向论文多项任务并依赖外部参数/配置及实验路径。 | 与延迟记忆最直接相关；其自抑制、层级拓扑和状态阈值是更具体的 TRNN 实现，若需要轨迹机制对照可借鉴论文定义。 | 不复制/导入仓库代码；只用论文来界定 claim。若要声称瞬态轨迹机制贡献，另行实现受控简化对照并独立验证。 |
| [RIMs](https://github.com/anirudh9119/RIMs/tree/aa61773a949c1e3f499d4c05dd77fcd2adbffa19) | 未找到仓库许可证或正式测试。检查 `event_based/blocks_core.py`、`BlockGRU.py`、`rnn_models.py`、README。实现含注意力路由、top-k 更新、模块间通信等复杂机制。 | 论文是模块化循环最强先例之一，但代码实现比主计划 grouped GRU 更复杂；复用会引入未匹配路由/选择计算，并带来许可不明问题。 | 不复制/导入；使用论文与源码审读来标定模块化基线边界。 |
| [PF-RNN](https://github.com/Yusufma03/pfrnns/tree/0a76d52a857f06d4128bbc7601c616d714b25734) | GPL-3.0；检查 `pfrnns.py`、`model.py`、README、LICENSE、requirements。作者实现有随机转移、粒子评分和软重采样，requirements 固定 PyTorch1.3.1；未见正式测试目录。 | 比固定分块更接近未来“竞争候选后验”的功能基线，但粒子数、随机采样和重采样改变状态与执行预算，直接导入会混淆当前结构主比较。 | 当前不复制/安装；先作为强先例和基线规格。若将来需数值比较，另行做独立实现、许可与资源审查。 |
| [Drosophila connectome reservoir](https://gitlab.com/EuropeanSpaceAgency/fly_connectome/-/tree/90679d672cd25905758b3abc6674fa6a28525d52) | API 未报告许可证。检查 `src/utils.py`、`src/main.ipynb`、README、`environment.yaml`；结果分析依赖 notebook 和外部连接组数据。 | 固定 ESN + 回归读出用于混沌预测，具有匹配参数和结构控制的经验价值；不符合端到端 BPTT 问题。 | 不纳入主实现、不复制。引用论文与报告中的公平比较维度。 |
| [Neuromorphic hierarchical modular reservoirs](https://github.com/fmilisav/milisav_hierarchical_modularity/tree/28c132738775dc0b72802dfb20fa346320ef08ec) | BSD-3-Clause。检查 `network.py`、`task.py`、`run.py`、README、LICENSE。源码调用 conn2res ESN 和 sklearn Ridge；没有测试文件/CI。 | 是模块层级、记忆容量和时间尺度的近期先例，但 reservoir 固定，且主问题任务不同。 | 不复用。其 2026 论文必须出现在新颖性边界中。 |
| [flyvis](https://github.com/TuragaLab/flyvis/tree/92b3845cc426dd309a1a0e1b3890156c42e14021) | MIT。检查网络动力学/初始化/网络求解器，以及 pytest/CI。项目依赖 connectome 图结构和大量领域任务数据。 | 高质量机制模型和测试工程参考，任务是特定果蝇视觉计算；与通用抽象模型比较无直接必要联系。 | 不作为本项目依赖或复制对象。 |
| [Continuous Thought Machines](https://github.com/SakanaAI/continuous-thought-machines/tree/4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466) | Apache-2.0。检查 `models/ctm.py`、`models/modules.py`、测试代码。架构引入神经元历史模型、同步估计、视觉/任务 backbone 和可调内部迭代。 | 多个高级机制会改变参数与推理计算，超出当前受控首问；它不是 grouped GRU 的替代实现。 | 不纳入首轮，不复制。作为“时间表示”和推理预算的相关工作。 |
| [ncps](https://github.com/mlech26l/ncps/tree/695ec22d7d1831f2ced67346ef7d2f08c525bd47) | Apache-2.0。检查 torch LTC/CfC 单元、wiring、测试。 | 连续时间/稀疏回路基线可服务后续问题，但当前四类模型已覆盖首问主要结构维度。 | 不增加首轮依赖；不复制。 |
| BPU | 只确认 [arXiv 论文](https://arxiv.org/abs/2507.10951)。未找到论文明确关联的作者仓库，故无源码许可/测试信息。 | 论文方法固定果蝇幼虫连接组循环核，只训输入输出投影；任务为图像与棋类。 | 不复用，不将第三方同缩写项目视作作者实现。 |

许可字段来自固定提交中的 LICENSE 文件或 GitHub/GitLab 仓库 API；“未报告/未找到”表示本轮证据不足，不等于确认该项目所有文件版权状态。任何代码复用都需要另做逐文件许可与数据许可核查。本轮决策是不把任何外部仓库代码复制到 NeuroTopo。
