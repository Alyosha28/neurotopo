# DIR-0004 设计审查：投递记录

2026-09-24。本文记录该次独立前瞻审查的**投递方式与 packet 冻结链**，以保证独立性可审计。**不是判词。**

## packet 冻结链

| 版本 | sha256 | 状态 |
|---|---|---|
| v1 | `f4ae80a73a32c76a4ac680f54121489973fd445471588048ea279c6559889bf7` | **投递前作废**：先例记录 `dir0004_chaos_tracking.md` 的 arXiv 通道事实写错（写成"http 301 ＋ https 全部超时"，实为 `429 Rate exceeded`（4 条）与超时（5 条）混合）。因**尚未向任何审查者投递**，按"先改准、后冻结"处理：更正该文件、重建 bundle、重新冻结。 |
| v2（现行） | `d7a03c66d022ea2aea8364f09a1341210f77c34245fe104954fbbcab4a52c54e` | **已投递** |

设计稿与送审说明在两次冻结之间**未改动**（`design_sha256 = 9b9244d1406af3c3f79780c6940aa2bdd74597beb6769378344d1c66f6576f1a`、`action_sha256 = c0f0c548dc6bbb54c5d400610cb28a0d8eb649272b728fbe5eccdc5550da878c` 两次一致）；v1→v2 只改了 bundle 内嵌的先例记录。

## packet 组成（三段，无其他内容）

- `design` = `research/design/dir0004_review_protocol_v0.md`（七问 ＋ 输入范围 ＋ 最强拒绝 ＋ 判词与决策规则）
- `action` = `research/design/dir0004_chaos_predesign_v0.md`（设计稿 v0.1）
- `action.outputs[0]`（`verification`）= `research/design/dir0004_review_bundle_v1.json`（sha256 `92f2be52b0b6cf743804bfb90f1e8ae651484a6755927428ae50b4c787e16b7d`），逐字内嵌 4 份机器输出、自查更正记录与先例记录，各自附 sha256

## 投递方式

- **方式**：文件投递。因 packet 是单行转义 JSON（18 个物理行），逐字引用会被行截断，故先由 `scripts/dir0004_extract_packet.py` 做**无损抽取**到 `derived/`，并把每段的 sha256 与 packet 内嵌值逐条核对（`derived/EXTRACTION_MANIFEST.json`）。抽取只读 packet、只写派生副本，**内容零改写**。
- **审查者**：全新上下文子代理 `1b481aab-21cc-4a35-af16-7cbeb72f7225`，提示词限定其**只读** `derived/` 下的文件，禁止读其他文件/目录/git 历史/网页，禁止运行实验代码。
- **独立性要求**：审查者须在 `overall_assessment` 末尾附 `FILES_READ`，逐字列出实际读过的所有路径；**若列出 packet 派生集之外的任何文件，该次审查独立性作废**，不得据其结论推进。

## 未决

- 审查结论尚未落地；在 `review.json` 与 `reaction.json` 落盘并跑通 `python -m neurotopo.reflection check` 之前，**不得**称本设计"已获独立审查"。
- 无论判词为何：Gate 1 仍 `NOT_PASSED`，无 EXP 注册，无训练授权。

## 结果与一次格式返工（2026-09-25 补记）

- **判词**：`V0_1_REJECTED`（9 项：7 `high` / 2 `medium`），主代理**逐条 accept**（见 `reaction.json`），按送审前固定的规则停止该设计线（决定0081）。
- **独立性**：`FILES_READ` 只列 packet 派生集内 8 个文件，未读包外文件、未上网、未跑代码 ⇒ 成立。
- **格式返工（记录在案，责任在我）**：首次 `reflection check` **被拒**——送审说明写了"引文可取自 `embedded_*`"，但校验器只把 packet **正文**当来源，而内嵌机器输出在 packet 中是 JSON 转义串（`"` 为 `\"`），故 5 段从 `embedded_*` 逐字复制的引文匹配失败。已请审查者做**纯格式修复**（只替换/删除该 5 段引文，不触碰 `id`/`severity`/`problem`/`why_it_matters`/`falsifiable_check`/判词/`FILES_READ`/`packet_sha256`）。修复后 27 段引文全部逐字命中，`check` 判 `{"decision_impact": "REASSESSMENT_REQUIRED", "issues": 9, "valid": true}`。
- **哈希链**：`review.json` sha `c46b93323bdf8675b82ac171cc893f8ed0894ac76bd5cfa23bd1c91acf8eee61`（首次）→ `06df80511083f599ff927bf5ad930eaff695a5daf945b03c276685d8099bf77d`（格式修复后，判词与内容未变）。`reaction.json` 指向后者。
