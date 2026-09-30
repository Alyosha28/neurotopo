# EXP-0013/0014 v3 独立实施前审查

独立`gpt-6-sol`/high只读`experiments/EXP-0013/v3.md`、`EXP-0014/v3.md`及共同冻结`research/registrations/G1-RS-0001_protocol_pair_v2.json`，未生成任务样本、训练或修改文件。裁定 **`READY_FOR_IMPLEMENTATION`**，只允许按冻结协议实现/验证源码，**不**放行任务采样或训练。

上一轮三项硬错已闭合：t16目标块只移植由替代早期u产生的可达内部状态，后段外部输入严格维持原例、以原例oracle评价，隔离早史状态→输出路径；`torch.empty`直接建参及三次uniform抽样、掩码外参数、bias和读出顺序/seed固定，RS/SC同种子初值可字节校验；资格轮`independent_benchmark`入口明确具名`h3_candidate`、实际SC主对照及`gate1_claim=false`的控制资格目的。共同manifest的两v3协议/矩阵/receipt SHA全匹配；静态枚举43,322根无碰撞，136,800研究+1,000工程=137,800更新及7,200秒分段预算正确。switch20相位、逐臂LR、全输入D、主CI/判词一致。

两处非阻断文字：EXP-0014 v3首段“本v2”按上下文应读作本v3；manifest说明中“未使用v1”未逐列另一个未使用草案。权威文件路径/SHA、科研规则和执行条件无歧义，保留冻结字节并在此说明，不为修字重设版本。下一步实现后仍需代码独立批评、干净源验证、资源工程预检和ResearchLoop各阶段；若EXP-0013任务/控制资格失败，EXP-0014正式轮不得启动。
