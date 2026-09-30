# 决定0102：ARCH-0001 完整架构候选的 A0 合同

2026-09-29。按 `ARCH-GATES-v1`，将 [`SC-REL-v3`](../design/arch0001_successor_compatible_core_v3.md) 记为 **`A0_CONTRACT_FROZEN / DESIGN_ONLY`**。本决定只确认一个完整、可证伪的计算候选已经定义；没有实现或训练该候选，A1–A6 均仍 `NOT_EVALUATED`，旧 Gate 1 仍 `NOT_PASSED`。

## 候选与判决边界

候选保留六值关系状态，模型在每步实际命令转移后试算 HOLD/FLIP 两种下一命令的读出，再用有界右乘修正当前状态；编码、转移、修正器和读出全部拟端到端训练。方向预测是：在控制仍有余量、任务与预算预先固定的条件下，长程逐步命令响应错误相对强 `INV-COMP` 主控制改善至少3个百分点且 HOLD/FLIP 任一条件恶化不超过1个百分点。实际主控制的函数类、有限宽度/内部块菜单、控制侧选择规则、额外监督控制、定义性消融、六值状态和实际执行计算主轴均已在合同中固定。已暴露的二值任务只用于 A1 工程/学习前测，不能凭 INV 封顶条件授予 A1 或制造 A2 比较。

`s_0,s_1` 是当前 Gram 和权重的确定函数，不增加状态信息。因此本候选只检验嵌套计算与训练路径在有限预算下的偏置；成功也不能自动证明新的表示能力。已知 [state-regularized RNN](https://arxiv.org/abs/2212.05178) 与 [predictive state representations](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf) 分别覆盖有限态正则与动作条件未来测试的相邻思想；本决定不主张机制首次出现。EXP0023 的训练一致性负/不确定结果和 EXP0027 的具体状态反例保留，后者只作动机，不计入本候选支持证据。

## 审计与修正记录

事前审计范围在 `research/design/arch0001_a0_audit_predesign.md`。v1 packet-only 独立反思发现三项问题：主控制函数类不定、消融混合推理特征与训练梯度、未检验修正前的假设命令输出。三项全部接受并落实于 v2；另由主编排者识别 `s_v=F_v(g(V))` 的信息冗余，主动收窄主张。v1 的 `packet/review/reaction` 位于 `research/reflections/ARCH-0001-A0/`，校验为 `valid=true / issues=3 / REASSESSMENT_REQUIRED`。

v2 再经全新上下文的 packet-only 反思，只发现未来修订也被误命名 v2 的版本冲突。接受后产生 v3：与 v2 的字节差只涉及标题、当前版/失败版说明和 `SC-REL` 版本号；计算方程、对照、端点及资源规则未改。v2 的反思三件套在 `research/reflections/ARCH-0001-A0-V2/`，校验为 `valid=true / issues=1 / REASSESSMENT_REQUIRED`。因此当前权威合同为 v3，v1/v2 及失败修订轨迹均保留；v3 若在 A1 失败，后继修订须命名 v4，不能原地改 v3。

## 状态和重开条件

A0 完成的是设计门。A1 尚缺实现、确定性工程检查、独立训练实例的学习资格和实测资源。合同所列旧 IID8 二值任务只能承担 A1 的工程/学习**前测**；该条件下强 INV 已接近或达到封顶，不能单凭前测满足 `ARCH-GATES-v1` 对 A1 任务非封顶余量的要求。A1 判词仍须另有事前限定、经控制侧验证有信息及非封顶余量的条件和固定资格规则。任何 A2 效果试验还须按合同先冻结任务菜单与校准分支，用强控制和 oracle 检查信息、学习与非封顶余量，再锁定独立 seed、端点、不确定性规则和 `ARCH-MATCH-v1` 资源比较。若资格失败，保留该配置失败，不依据候选成绩移动任务或端点。A3 的训练消融不能单独归因为在线推理特征；推理配对干预与梯度路径需分别核查。A4–A6 均无证据。

本轮无研究优化更新、无新模型输出，也不触碰旧 ResearchLoop `project_gate1` 字段。仓库中原有用户未提交改动保持原状。
