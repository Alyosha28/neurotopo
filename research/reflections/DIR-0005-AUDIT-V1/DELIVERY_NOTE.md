# DIR-0005 审计 v1 再审查：投递记录

2026-09-25。记录本次再审查的 packet 冻结链与投递方式，以保证独立性可审计。**不是判词。**

## 两次审查的链

| 轮次 | packet sha256 | 判词 | 处置 |
|---|---|---|---|
| v0 审查 | `58151ba7c65c0fddc3b9d5959e552a7d4d26b97b272cbc86f0b919e67e2008dd` | `AUDIT_NEEDS_NARROWING`（8 项：5 high/3 medium，**全部 accept**） | `reaction.json` 落盘，`reflection check` 判 `{"decision_impact": "REASSESSMENT_REQUIRED", "issues": 8, "valid": true}`；按事前承诺出 v1 |
| v1 再审查（本次） | `93ba84cd9949251349195f00aeb4af796c7b0477f143b61d3f6aaad023157022` | 待落盘 | 通过才出决定记录 |

v0 审查的 `review.json` sha `9079170a…`；`design` sha `8cbccc2f…`、`action` sha `9ba09dba…`、bundle sha `477672a7…`（v0 三件套与上述哈希一一对应）。v1 的 `design` sha `ffd4542c…`、`action` sha `4e5baf17…`、bundle sha `137cb9d3…`。

## v1 送审包的证据内容

逐字内嵌：决定0063、0068、0071、0073、0080、0081；DIR-0004 审查的 9 条 issue；**`research/evidence/gate1_benchmark_baseline_audit.md` 全文**（v1 那处"实测余量已存在"更正的原件）；**v0 审查的 8 条 issue**；`research/stage_gates.md` 两段声明行区间（附整文件 sha256）。新增后两项正是本次再审查必须能自查的材料：v1 的改动是否忠实于 v0 审查，以及余量更正是否有原件支撑。

## 投递方式与独立性

- **方式**：文件投递 ＋ 无损抽取（逐段 sha256 与 packet 内嵌值核对），审查者只读 `derived/`。
- **审查者**：全新上下文子代理 `f9a7fa72-5161-48ca-b46a-fdf5bc8f2e93`；提示词限定只读 `derived/`、禁网、禁跑代码、禁读其他文件与 git 历史。
- **引文来源规则**：明确要求 `evidence` **只从 `design.md`/`action.md` 逐字取**（内嵌证据是 JSON 转义串，校验器不匹配）——这是 DIR-0004 轮流程教训的延续。
- **独立性要求**：`overall_assessment` 末尾附 `FILES_READ`；若列出 packet 派生集之外的文件，该次审查独立性作废。

## 未决

- v1 判词未落盘前**不得**出决定记录，也不得称本审计"已获独立审查"。
- 无论判词为何：Gate 1 仍 `NOT_PASSED`；无 EXP 注册、无训练授权；非回归清单 12 条无一重开。
