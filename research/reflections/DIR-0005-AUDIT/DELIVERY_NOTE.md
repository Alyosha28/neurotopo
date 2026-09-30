# DIR-0005 审计审查：投递记录

2026-09-25。本文记录该次独立前瞻审查的**投递方式与 packet 冻结链**，以保证独立性可审计。**不是判词。**

## packet

| 项 | 值 |
|---|---|
| 路径 | `research/reflections/DIR-0005-AUDIT/packet.json` |
| sha256 | `58151ba7c65c0fddc3b9d5959e552a7d4d26b97b272cbc86f0b919e67e2008dd` |
| 大小 | 90,535 bytes |
| `design` | `research/design/dir0005_audit_review_protocol_v0.md`（sha `8cbccc2f…`，七问 ＋ 判词与决策规则） |
| `action` | `research/evidence/DIR-0005_candidate_spec_audit.md`（sha `9ba09dba…`，审计正文） |
| `action.outputs[0]`（verification） | `research/evidence/dir0005_review_bundle_v1.json`（sha `477672a7…`，69,484 bytes） |

## 证据包组成（逐字内嵌，各附 sha256）

决定0063（CAL-0008 关闭）、决定0068（非回归 12 条清单）、决定0071（v3 再入筛查）、决定0073（路由对齐方向停止）、决定0080（入口放行与 `Generic` 缺口）、决定0081（DIR-0004 停止）、DIR-0004 审查的 `review.json` 全部 9 条 issue；`research/stage_gates.md` 逐字内嵌**两段声明行区间**（第 17–30 行门表、第 96–115 行 DIR-0002 标签与 `G1-MATCH-v3` 规则节），并附**整文件** sha256 与行区间声明，使切片可审计而非静默摘取。

## 投递方式

- **方式**：文件投递 ＋ 无损抽取（`scripts/dir0004_extract_packet.py`，逐段 sha256 与 packet 内嵌值核对），审查者只读 `derived/`。
- **审查者**：全新上下文子代理 `6d725c1a-323f-439a-b295-9709ef19741b`；提示词限定只读 `derived/`、禁网、禁跑代码、禁止读其他文件与 git 历史。
- **独立性要求**：`overall_assessment` 末尾须附 `FILES_READ`；若列出 packet 派生集之外的任何文件，该次审查独立性作废。
- **引文来源规则（上一轮教训的流程修正）**：本轮明确要求 `evidence` 引文**只从 `design.md`/`action.md` 取**。上一轮（DIR-0004）我允许引用 `embedded_*`，但校验器只匹配 packet 正文，而内嵌内容是 JSON 转义串，导致首次 `reflection check` 被拒、需审查者做纯格式修复。本轮在投递前即消除该冲突。

## 未决

- 审查结论尚未落地；在 `review.json` 与 `reaction.json` 落盘并跑通 `python -m neurotopo.reflection check` 之前，**不得**称本审计"已获独立审查"，也不得据其标签出决定记录。
- 无论判词为何：Gate 1 仍 `NOT_PASSED`，无 EXP 注册，无训练授权，非回归清单无一重开。
