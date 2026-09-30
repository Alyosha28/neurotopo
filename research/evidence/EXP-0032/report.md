# EXP0032：修正端点上的控制菜单与预算扫描结果

2026-09-29。判词（v2）：**`INCONCLUSIVE / QUALIFICATION_NOT_REPRODUCED`**；同时记录 v1 的实现缺陷。协议 [`experiments/EXP-0032/v1.md`](../../experiments/EXP-0032/v1.md)（含 v1 缺陷附注）。

## v1 缺陷（作废）

`run_one` 只把 `width/blocks` 写进记录、没有注入传给 `fit_memory` 的配置，而该函数从 `cfg["control"]` 建模型 → 12 个"配置"实际是同一个 `INV_COMP(w32,b1)`：Stage A 全部记录共享 `final_hash` `972eca8c…`、`meaningful` 全为 0.6058(d=2)/0.6162(d=3)，而记录里的参数量只是账目值。v1 产物保留在 `artifacts/EXP-0032/v1/`（另有启动崩溃的空目录 `v1_aborted_preflight`），判词作废；协议规则未改，修正后以 v2 重跑。回归测试 `test_control_config_reaches_the_model` 已加入。

## v2 结果（权威）

Stage A（2000 更新、seed 29101、12 配置×d∈{2,3}）显示配置差异显著：

| d=2 配置 | meaningful | | d=2 配置 | meaningful |
|---|---:|---|---|---:|
| w32b2 | **1.0000**（精确解出） | | w96b2 | 0.6827 |
| w32b3 | 0.9139 | | w48b3 | 0.6784 |
| w64b2 | 0.7455 | | w64b1 | 0.6080 |
| w48b2 | 0.7064 | | w32b1/w48b1/w96b1 | 0.6058 |
| w96b3 | 0.6992 | | w64b3 | 0.4968 |

d=3 最好 0.6359，无一落带。按冻结规则（带内取 `|ln(控制算子/候选1161)|` 最小、并列取参数少者）选中 `w48b2`（0.7064）。Stage C（seed 29102 + 29111–29113）5 seed 为 [0.7064, 0.5659, 0.6941, 0.6112, 0.6496]，均值 **0.6455 < 0.65** → 未复现。

**结论**：`DELAY-2` 并非不可学（`w32b2` 精确解出、多项落带），但"计算最接近候选"的那一项在独立 seed 上未达可学门槛；按协议判 `QUALIFICATION_NOT_REPRODUCED`。该结果为 EXP0033 提供了载体候选集合。
