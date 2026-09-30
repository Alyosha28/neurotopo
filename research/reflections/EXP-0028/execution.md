# EXP-0028 v1 packet 反思执行记录

2026-09-29。按 `docs/post_task_reflection.md` 在判词（决定0103）后、下一轮之前执行一次独立 packet 反思。

## 输入与上下文

- packet：`research/reflections/EXP-0028/packet.json`，SHA-256 `d6ad393a7fedf95ff613cbe96bd1697d828708121274306f4750636a7e30f92a`。
  - design：`experiments/EXP-0028/v1.md`（运行前冻结协议），SHA-256 `5fefd4f36e18555a17610b39de7b4aa1c707463df4139a488b3fdeb9f2460c42`。
  - action：`research/reflections/EXP-0028/action_summary.json`（`artifacts/EXP-0028/v1/analysis.json` 的去 loss 视图；仅为 packet 的 250KB 上限而对逐 step loss 数组做裁剪，指标、哈希、判词与事后参照标注未改），SHA-256 `46e33de8d96dbef438364ab6c4d2084d9e4c7356b4b4cbbddb02ec204af091f0`。
  - 追加 action 产物（均属回答"执行与判定是否自洽"所必需的已完成产物）：运行 `manifest.json`、独立重放 `verification.json`、决定 `0103_exp0028_a1_qualification.md`。
- 反思者：独立 subagent，`fresh context`（无本会话上下文、无先前批评、未列入候选问题），按指令只读 packet 文件，不浏览仓库、不运行实验；模型标识未向编排者暴露，记为 agent-default。反思者只返回 review 内容，未提供 packet/review 的封装字段（`schema_version`、`packet_sha256`、`reviewer`、`overall_assessment`），这些字段由编排者按实际上下文补齐，issues 内容未改。
- 已知偏离：反思者在其总结中提到写入目录，提示它可能列过目标目录；packet 之外没有给它任何研究内容，仓库其他文件未被读取。

## 结果

反思报出 8 项，校验 `valid=true / issues=8 / decision_impact=UNCHANGED`（`python -m neurotopo.reflection check`）。逐项处置见 `reaction.json`，摘要：

| id | 处置 | 行动证据 |
|---|---|---|
| posthoc_reference_contradiction | accept | 决定0103 补记事后性质与"不进入判词"；`explicit_reference.json` note |
| pretest_replay_evidence_overstated | accept | 扩展重放脚本并重跑；`verification.json` 补 logits 哈希与 BCE 差（全 0） |
| registered_seed_28111_replay_missing | accept | 决定0103 与验证记录明确 28111 的条件性未运行；不补做超设计训练 |
| gate_file_provenance_not_pinned | accept | `provenance_check.json` 固定提交判据哈希、工作区差异范围与未提交限制 |
| n5_qualification_margin_fragile | accept | `posthoc_sensitivity.json`：额外 3 seed＋评估流重采样，5 seed 保持均值 0.5987<.60，判词不变 |
| learning_arm_count_ambiguous | accept | 决定0103 记录本轮读法与后续版本要求；本轮未进入该阶段 |
| decision_table_missing_info_failure_row | accept | 决定0103 记录缺口与先行行适用关系 |
| project_gate1_untouched_claim_unverified | accept | `provenance_check.json` 记录运行前后 `loop status` 关键字段与哈希未变 |

`decision_impact` 为 `UNCHANGED`：没有任何一项改变 `INCONCLUSIVE / NO_QUALIFIED_CONDITION` 判词、A1 状态或阶段门。反思记录不用于追认能力或推进阶段门。
