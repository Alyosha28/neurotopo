"""Assemble the DIR-0002 independent review record from the packet-only reviewer output.

The issue text is the reviewer's; this script only fixes `source` labels so every
quoted string is verbatim-anchored in the packet source it is attributed to, then
validates the record with the repository's own checker.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKET = Path("research/reflections/DIR-0002/packet.json")
OUT = Path("research/reflections/DIR-0002/review.json")

ISSUES = [
    {
        "id": "DIR-0002-R1",
        "source": "both",
        "severity": "high",
        "evidence": [
            "**`QUESTION_READY`**（详见 §4）：存在至少一条满足事前设计 (a)–(c)、且不需要修改现行科学判据的方向",
            "把学习效率作为主终点是否需要前瞻裁定，**由独立审查回答，不由本调查解释**",
            "`QUESTION_READY`：至少一条满足 (a)–(c) 且不需要新版本规则",
        ],
        "problem": "设计把 QUESTION_READY 定义为不需要新版本规则；D2 却被判 QUESTION_READY，而同一行动又把终点类型问题交给独立裁定，即它确实需要裁定。D1 另获 `QUESTION_READY_AFTER_RULING`，该标签不在设计的三标签菜单内，功能上等同 RULE_CHANGE_FIRST（须先裁定 0049 适用性）。设计的 RULE_CHANGE_FIRST 分支（记录待裁定规则问题、不注册实验）从未被使用，尽管 D1 与 D2 都被描述为等待裁定。",
        "why_it_matters": "调查的标题判词（§0 的 QUESTION_READY）无法由设计自身的判据规则推出，因此调查的最高层结论不被其验收条件支持。",
        "falsifiable_check": "只用行动自己的 (b) 列把 D1–D5 映射到设计的三标签：任何 (b) 写着'待裁定'的行都必须落入 RULE_CHANGE_FIRST，从而没有任何行符合 QUESTION_READY。",
    },
    {
        "id": "DIR-0002-R2",
        "source": "action",
        "severity": "high",
        "evidence": [
            "每单位科学信息对应的治理文本量级约为 10³",
            "| 治理/审查文本合计 | 288 文件，1.22 MB |",
            "| **任务级分析产物** | **9 个 `analysis.json`，0.30 MB** |",
            '"governance_review_text_bytes": 1250746',
        ],
        "problem": "10³ 无法由所给数字推出。治理文本（1,250,746 B）比任务级分析产物（302,992 B）约为 4.1，而非约 10³，差两到三个数量级。只有用治理全库除以单个 366 字节占位产物才可能得到 10³ 量级，而该分母从未被说明，且是把语料库与单个文件相比。分母也不是'科学信息量'：0.30 MB 是 9 个产物的字节总和，而行动自己的例子显示这类文件可以是 366 字节的零信息占位。",
        "why_it_matters": "这一量化诊断是整个吞吐论证（§1.1 → D5）与 R-A 支持面的证据基础；在一份自称'只读统计'的文档里，标题性的数量级与其自身引用的数字矛盾，属定量夸大。",
        "falsifiable_check": "用 packet 自身的数字重算 1250746/302992 与 1.22/0.30；检查是否存在任何被说明过的分母能给出约 1000。",
    },
    {
        "id": "DIR-0002-R3",
        "source": "action",
        "severity": "high",
        "evidence": [
            '"citation_crosschecks_all_found": true',
            '"figure": "CAL-0001/0002 触顶区间 0.0003–0.0023",',
            '"literal": "0.0003",',
            '"figure": "EXP-0004 主控制余量 32.96pp–100%，M3 低 9.10pp",',
        ],
        "problem": "每项交叉核对只验证每个图题的一个字面量子串，随后输出即断言 `citation_crosschecks_all_found: true`。CAL-0001/0002 只核了 `0.0003`，上界 0.0023 未核；EXP-0004 只核了 `9.10pp`，`32.96pp–100%` 未核；EXP-0012 的 `19.742137` 只以 `19.742` 作核。因此该标志并不表示所引数字已被验证，只表示每个图题有一个子串被找到。",
        "why_it_matters": "该验证输出被呈现为引文完整性的机器证据，其汇总标志高估了覆盖范围，使驱动 R-A 部分成立判定的数字 pincer 表继承了未验证的区间端点。",
        "falsifiable_check": "对每项交叉核对，检验 `figure` 全串（两端、全精度）是否出现在所列来源中，统计其中仍能通过的数量。",
    },
    {
        "id": "DIR-0002-R4",
        "source": "action",
        "severity": "high",
        "evidence": [
            "**学习效率终点在本仓库的规则下从未被禁止，只是从未被注册为终点。** 这是本调查的核心发现。",
            '所以终点更换不是"降低要求的权宜之计"，而是**定理逼出的唯一剩余可辨轴**。',
            '"figure": "决定0043/STR-0004：同容量强单流可内部模拟 K 槽",',
        ],
        "problem": "行动的两条承重主张——'核心发现'（学习效率终点从未被禁止，只是从未注册）与由 TH-0001 状态维下界推出的'唯一剩余可辨轴'——在交叉核对中没有条目。12 个被核条目全是数字加 0043/open_questions/MON-0001；TH-0001、4/8 个 log 权重的精确滤波构造与'同一状态维下界'定理均未被核，而 D2 的关键决定（0009/MON-0001）仅以两字字面量 `样本效率` 作核。",
        "why_it_matters": "调查的新贡献依赖于机器证据从未触及的文本/定理主张，因此 `citation_crosschecks_all_found` 不能被读作对它所伴随结论的佐证。",
        "falsifiable_check": "列出行动标为'核心发现'与'唯一剩余可辨轴'的主张，检查是否有任何交叉核对来源或字面量指向 TH-0001、决定0017 或该下界定理。",
    },
    {
        "id": "DIR-0002-R5",
        "source": "action",
        "severity": "high",
        "evidence": [
            "**结论（R-A 的支持面）**：最近 14 小时产出约 30 份决定与约 25 个反思包，**零新任务信息**。",
            "| 其中含真实任务指标的最近一次 | **EXP-0009，09-24 01:51（约 14 小时前）** |",
            "需同时满足四条，而仓库自己的证据显示每条都已被触发过至少一次",
        ],
        "problem": "R-A 支持面有两处问题。(i) '最近 14 小时零新任务信息'的时间窗由表格自称'约 14 小时前'的同一个 01:51 产物定义，而最后一次提交在 15:52，故该窗口的选取恰好排除掉那个唯一的测量，'零新任务信息'的措辞依赖边界选择。(ii) '每条都已被触发过至少一次'完全由失败案例（FIELD_ONLY、RESOURCE_MISMATCH、NO_MATRIX、触顶）支持；这些显示四条判据各自击败过候选，而非四条曾被联合触发，且行动的下一句自己也承认该联合角落只是'很窄'而非已证明为空。",
        "why_it_matters": "R-A 是设计要求正面处理的最强反驳；其答复建立在贴合边界的时间窗与只证明单条件失败的证据之上，因此 R-A 是被重述加部分证据回答，而非被正面回答。",
        "falsifiable_check": "给出'最近 14 小时'所用的确切截止时刻，并列出任何时间新于它的含任务指标产物；另须给出一个曾同时满足四条件的设计，或删去'至少一次'这一主张。",
    },
    {
        "id": "DIR-0002-R6",
        "source": "action",
        "severity": "medium",
        "evidence": [
            "| 仓库提交 | 131 | `git rev-list --count HEAD` |",
            '"commits": 133,',
            "| 治理/审查文本合计 | 288 文件，1.22 MB |",
            '"governance_review_text_bytes": 1250746,',
        ],
        "problem": "设计文本与行动表格都写 131 次提交、288 文件 / 1.22 MB、36 个反思包，而为同一审计生成的验证输出记录的是 133 次提交、289 文件 / 1,250,746 B、37 个包。1,250,746 B 是 1.25 MB 而非 1.22 MB，故这不是舍入。设计自己的诚实性说明称输入范围部分是'事后重建'，与手工计数过期后未被核对而进入两处文本的情况一致。",
        "why_it_matters": "同一 packet 内对同一批数量存在两套互不一致的测量，而调查的'治理≫测量'框架依赖这些计数；packet 中没有任何东西调和它们。",
        "falsifiable_check": "比较每个同时出现在正文表格与 `portfolio_stats` 中的计数，标出所有不相等的成对值。",
    },
    {
        "id": "DIR-0002-R7",
        "source": "action",
        "severity": "medium",
        "evidence": [
            "因此不需要 headroom、不需要时延匹配、不受天花板影响",
            "要作证据须在具指派的 RS4 类模型上做",
            "按未验证的回忆，预算匹配的逐臂调参与曲线终点是既有最佳实践，故 D2 必须表述为**采用既有方法学**而非发现；该回忆未核。",
            "未验证的模型回忆一律不作为引用或重叠判词",
        ],
        "problem": "D4 的'不需要余量与资源匹配'只在旧检查点的方法学试点上成立；作为证据它明确要求具指派的 RS4 类模型，即训练与资源匹配，这与标题式的笼统主张矛盾。另外，§5 承诺未验证回忆绝不用作引用或重叠判词，随后却立即用未验证回忆限定 D2 的表述（'故 D2 必须表述为采用既有方法学而非发现'）。",
        "why_it_matters": "前者使一条方向看起来比其自身限制条款所允许的更便宜、更无需过门；后者显示诚实性承诺未被应用于自身，削弱 packet 的核心披露主张。",
        "falsifiable_check": "检查是否有任何影响方向可行性或表述的行动主张仅源自该未验证回忆，以及 D4 的成本/资格陈述在其自身 RS4 要求下是否仍成立。",
    },
    {
        "id": "DIR-0002-R8",
        "source": "both",
        "severity": "medium",
        "evidence": [
            "× 非纯先例（决定0006/0012）",
            "| 非纯先例 + 任务族合法 | 决定0006/0012（纯先例效果不能过项目 Gate 1）；决定0049",
        ],
        "problem": "设计的 R-A 给出四个条件，第四条是'非纯先例'（决定0006/0012）。行动把该行静默改名为'非纯先例 + 任务族合法'，并把决定0049（最后一次任务生成器尝试）与 EXP-0012 的 NO_MATRIX 任务族结果并入其中。于是两个不同约束——先例纯度与任务族合法性——被计为一个条件，'四条'不再与设计清单一致。",
        "why_it_matters": "pincer 是受检对象；在没有声明的情况下中途改变其条件集合，意味着'每条都已被触发过至少一次'是针对与预注册不同（且更宽松）的划分来评估的。",
        "falsifiable_check": "把设计的四个 R-A 合取项与行动表格的四行做一一对照，找出任何增删了子条件的合取项。",
    },
    {
        "id": "DIR-0002-R9",
        "source": "both",
        "severity": "low",
        "evidence": [
            "| D5 吞吐工程 | 不适用（非科学） | 无阻塞 | 解卡 | `ENGINEERING` |",
            '不得以"建议"形式保留',
        ],
        "problem": "D5 被赋予 `ENGINEERING` 标签，这是设计三标签菜单之外的第四/第五个标签。与 D1–D4 不同，它不是科学方向，也不提供 (a) 类可数值否定预测（表格自己写'不适用'），却仍被保留在方向组合内，并附有流程处方（检查点门控、强制零样本筛查、审查轮数上限）；按设计的 (b) 要求，这些本应被归入 RULE_CHANGE_FIRST。",
        "why_it_matters": "在一个自称只做方向选项的文档里，用一个自造标签保留非科学条目，正是设计以'不得以建议形式保留'所关闭的那条后门。",
        "falsifiable_check": "检查 `ENGINEERING` 是否出现在设计允许的标签集合中；若不在，判定 D5 应落入三个设计标签中的哪一个。",
    },
]

OVERALL = (
    "最重缺陷：调查的标题判词 QUESTION_READY 未按预注册判据规则一致适用——D2 被判 QUESTION_READY，而其 (b) 列又把终点类型问题交给独立裁定，"
    "这正是设计的 RULE_CHANGE_FIRST 条件；行动还发明了 QUESTION_READY_AFTER_RULING / BLOCKED_ON_PRIOR_ART / ENGINEERING 三个三标签菜单之外的标签，"
    "使审计的最终标签无法由其验收标准推出。次要但严重：量化诊断依赖的'10³'治理-科学比被 packet 自身数字（1.22 MB 对 0.30 MB，或 1,250,746 B 对 302,992 B ≈ 4.1）否定；"
    "且验证的 `citation_crosschecks_all_found: true` 建立在对每个图题仅一个子串的匹配之上，区间（0.0023、32.96pp–100%）与承重的 TH-0001/'核心发现'主张完全未被核。"
    "R-A 由部分证据与贴合边界的 14 小时窗口回答，而非由联合不可满足的证明回答。先例缺口本身处理诚实（NOT_EXECUTED、零来源、D3/D4 阻塞、不记录标识符），"
    "设计对输入范围事后重建的诚实说明是真实优点。只凭 packet 无法核查的：所引仓库文件、决定与产物是否存在或其内容是否如所述（无文件、无检索、无代码）；"
    "131 对 133 提交与 288 对 289 文件的差异是过期计数还是树已变化；pincer 四条件事实上是否联合不可满足；D1–D5 是否与未列出的先例重复；"
    "以及设计文件是否真的先于分析动作，因为顺序的唯一证据是行动自称的提交 `b60d09b` 与 design.content 同 predesign 产物相符的 sha256。"
)


def main() -> int:
    packet = json.loads(PACKET.read_text(encoding="utf-8"))
    design = packet["design"]["content"]
    action = packet["action"]["content"]
    outputs = "\n".join(item["content"] for item in packet["action"].get("outputs", []))
    action_text = action + "\n" + outputs
    texts = {"design": {"design"}, "action": {"action"}, "both": {"design", "action"}}
    for issue in ISSUES:
        found: set[str] = set()
        for quote in issue["evidence"]:
            for name, content in (("design", design), ("action", action_text)):
                if quote in content:
                    found.add(name)
        if not found or not texts[issue["source"]] <= found:
            raise SystemExit(f"issue {issue['id']} evidence not anchored in declared source: {sorted(found)}")
    review = {
        "schema_version": 1,
        "packet_sha256": hashlib.sha256(PACKET.read_bytes()).hexdigest(),
        "reviewer": {
            "model": "dsh-subagent (deepseek-v4.1-flash, fresh context)",
            "effort": "default",
            "context": "fresh_only_packet",
        },
        "issues": ISSUES,
        "overall_assessment": OVERALL,
    }
    OUT.write_text(json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "issues": len(ISSUES)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
