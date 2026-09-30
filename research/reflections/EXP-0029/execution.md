# EXP-0029 v1 packet 反思执行记录

2026-09-29。按 `docs/post_task_reflection.md` 在决定0104 后、下一轮之前执行一次独立 packet 反思。

## 输入与上下文

- packet：`research/reflections/EXP-0029/packet.json`，SHA-256 `df9cc8feb8ebf3c4462831f859ceaa95b876a3779ba96724c9726165fda2c4c5`。
  - design：`experiments/EXP-0029/v1.md`（运行前冻结协议），SHA-256 `cf30a5332d6b9e75866c5c394e84b183b98dd2d29e7f54d076231b768249b9da`。
  - action：`research/reflections/EXP-0029/action_summary.json`（`artifacts/EXP-0029/v1/analysis.json` 的紧凑视图；省略了每 seed 的 6 个次 cell 明细与逐 seed `current_ok_next_response_bad`，完整值在 `artifacts/EXP-0029/v1/runs/*.json`），SHA-256 `b35b0c033700b87ff1ad2268ed4221b05a2a6fad5598b5cdac53a234b58bc27a`。
  - 追加 action 产物：运行 `manifest.json`、独立重放 `verification.json`、决定 `0104_exp0029_learning_qualification.md`。
- 反思者：独立 subagent、fresh context（无本会话上下文、无先前批评、无候选问题清单），按指令只读 packet 文件；模型标识未向编排者暴露，记为 agent-default。反思者只返回 review 内容，封装字段（`schema_version`、`packet_sha256`、`reviewer`、`overall_assessment`）由编排者按实际设置补齐，issues 原文未改。
- 已知偏离：与 EXP-0028 轮相同，反思者在总结中提到写入目录，提示可能列过目标目录；packet 之外未提供任何研究内容。

## 结果

反思报出 6 项，校验 `valid=true / issues=6 / decision_impact=UNCHANGED`。逐项处置见 `reaction.json`：

| id | 严重度 | 处置 | 行动证据 |
|---|---|---|---|
| merge-prior-seeds | high | accept | 决定0104 判词依据改为只依 12 新 seed 主终点；前 3 seed 单独报告；14/15 降为事后描述性读数（决定、报告已改写） |
| frozen-hash-chain | high | accept | `provenance_check.json` 记录三段哈希链 ca338347→3de25da0→7fb6bf81，说明改动范围与 core 未变；决定/报告更正 |
| run-phase-deviation | medium | accept | `provenance_check.json` 增加 engineering_classification：异常在 36 记录写盘后发生于汇总路径、受影响记录 0、ENGINEERING 行不成立 |
| replay-plan-deviation | medium | accept | `provenance_check.json` 记录重放目标/数量/事后选择；28103 由 EXP0028 重放覆盖 |
| all-pass-row-gap | low | accept | 决定记录零失败情形的空真读法与后续版本要求 |
| per-seed-diagnostics-gap | low | accept | `provenance_check.json` 记录 packet 摘要省略字段与完整来源；决定重开条件要求后续摘要列明省略字段 |

两项 high 均由主模型接受并落实到已存在的行动证据文件；`decision_impact` 为 `UNCHANGED`：更正后判词仍由判词表第 4 行（`CAND` 12/12）给出，`PARTIALLY_SUPPORTED / LEARNING_QUALIFICATION_AT_BUDGET` 不变，A1 仍因缺少非封顶合格任务而未通过。
