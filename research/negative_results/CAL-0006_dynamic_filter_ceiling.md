# CAL-0006：动态候选任务仍未留下稳健主对照余量

在预先固定的四模式二维动力学滤波任务上，解析 oracle 显示真实信息与早末修订（OOD12 修订率0.4127），两个 H64 单流控制均学会 IID8 后验；但 OOD12 的 dense seed603 仅有0.06497 nat oracle-relative KL，低于每臂每种子≥0.08的资格下限。其余五项在窗口内，所以负结果只针对**全六项必须合格的冻结筛查规则**，不是普遍天花板或多轨迹失败。详情、所有六项数值与复核见 `research/evidence/CAL-0006_dynamic_filter_qualification.md`、`research/decisions/0021_cal0006_dynamic_screen_ceiling.md` 和原始不可覆写产物。任务不进入架构比较，Gate 1 保持 `NOT_PASSED`。
