"""Assemble the CAL-0008 sixth pre-run review record (READY_TO_RUN) from the reviewer output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKET = Path("research/reflections/CAL-0008-PRERUN/packet.json")
OUT = Path("research/reflections/CAL-0008-PRERUN/review.json")
RULINGS_OUT = Path("research/reflections/CAL-0008-PRERUN/rulings.json")

ISSUES = [
    {
        "id": "CAL-0008-PRERUN6-R1",
        "source": "both",
        "severity": "low",
        "evidence": ["未训练下限 `0.10` nat 与上界 `0.25` nat", "未训练 KL 不得超过 `ln 2^M + 0.5`"],
        "problem": "审查说明自身的门槛清单仍写未训练上界 `0.25` nat，而 v7 的冻结阈值表已改为逐 seed 的 `ln 2^M + 0.5` nat。两份文本对生效上界的说法不一致。",
        "why_it_matters": "`0.25` nat 正是第五轮阻断的那个误标门槛。设计文档是权威冻结文本，且 packet 自身数字（C1 1.29–1.43、C2 2.58–2.60、C3 1.49–1.53、C4 1.38–1.45，对应上界 3.27／6.05）显示没有任何配置被训练前拦截。不改变本筛查能发出的任何判词，故非阻断项。",
        "falsifiable_check": "在设计中检索 `0.25`、在行动文本中检索 `ln 2^M + 0.5`；把审查说明问题 1 改写为 v7 上界即可消除不一致，且不改变任何配置的训练前门结果。",
    },
    {
        "id": "CAL-0008-PRERUN6-R2",
        "source": "both",
        "severity": "low",
        "evidence": [
            "措辞并在 R-h 写出 `LR_TOL`/`κ` 析取",
            "已加 LR 不敏感门（两个注册 LR 都须显示残余）",
            "删除过时 LR 措辞",
        ],
        "problem": "第四项要求只做了一半：§4 的冻结跨 LR 门已写出 `LR_TOL`/`κ` 析取，但 §7 的反驳条目 R-h 仍保留现时的过时标签且未写析取，而工程回应却声称该措辞已删除。",
        "why_it_matters": "读 R-h 的实现者可能采用更严的“两个 LR 都必须显示残余”，但该读法是 §4 合取项（另一注册 LR 的末端 KL 必须 > `ε_abs`）的子集，而 §4 才是冻结门。`HEADROOM`/`SATURATED`/`LEARNING_FAILURE`/`INCONCLUSIVE` 无一改变；属编辑问题，非阻断。",
        "falsifiable_check": "在 v7 行动文本中检索 `LR 不敏感门`：R-h 仍有一处现时表述。删除它并在该处复述 §4 的析取，两节即一致。",
    },
    {
        "id": "CAL-0008-PRERUN6-R3",
        "source": "both",
        "severity": "low",
        "evidence": [
            "本次送审为 v7，第六轮",
            "累计四轮预运行审查均判 `REVISE_BEFORE_RUN`",
            "**四轮**独立预运行审查均判 `REVISE_BEFORE_RUN`",
            "CAL-0008 v5 注册运行前的工程核对（非科学结果）",
        ],
        "problem": "版本账目仍过期：两份文档都说四轮审查判 `REVISE_BEFORE_RUN`，而本次是第六轮、且第五轮本身也被描述为判 `REVISE_BEFORE_RUN`；工程产物的任务标签仍写 v5；v7 台账行未逐项列出第五轮的第 2、4、6、7 项。",
        "why_it_matters": "纯账目问题。没有任何阈值、统计量、分支或门依赖审查计数或台账行，故不改变任何发出的判词。非阻断。",
        "falsifiable_check": "把设计中的审查小节计数（1–5 轮加本轮第六）与“四轮”一语对照；更正计数、重标产物任务名、补齐四条台账行。",
    },
    {
        "id": "CAL-0008-PRERUN6-R4",
        "source": "action",
        "severity": "low",
        "evidence": ["残差由 0.27–0.48 降至 ≤0.07", "1.3612", "1.2897086143493652", "1.4531859159469604"],
        "problem": "工程回应称熵恒等式残差降至至多 0.07，但 packet 每个配置只公布一个 `ln2^M − H` 值与一个未训练 KL 的 min–max 区间，无法逐 seed 重算残差。按公布的端点，C1 残差达 0.0715（1.3612 对 1.2897086143493652）、C4 达 0.0712（1.382 对 1.4531859159469604）。",
        "why_it_matters": "这是工程证据中的一句数字主张，packet 自身数字并不支持。§4 已明确该残差只作诊断、不设门，故不改变任何判词；严重度低、非阻断，但该句按字面为假。",
        "falsifiable_check": "逐 seed 公布未训练 KL 与该 seed 的后验熵；若任一最大残差超过 0.07，须更正该句。",
    },
    {
        "id": "CAL-0008-PRERUN6-R5",
        "source": "action",
        "severity": "low",
        "evidence": [
            "`BUDGET_ADEQUACY_ANCHOR_FAILED`（被 C1 锚点降级）",
            "`downgraded_by = budget_adequacy_anchor_failed`",
            "其他门失败是门槛标定问题",
        ],
        "problem": "新的 `INCONCLUSIVE` 原因分类不是单值的，且其兜底项被错误解释：C1 锚点降级出现两种拼写，而将吸收 n=3 下 `ε_abs` t 区间跨零情形的 `OTHER_GATE_FAILED`，被解释为门槛标定问题而非精度问题。",
        "why_it_matters": "§8 按原因归档，两种拼写会把同一原因拆进两个桶，而受精度限制的 `INCONCLUSIVE` 会被记成标定缺陷。判词无论如何都是 `INCONCLUSIVE`，`HEADROOM`/`SATURATED` 结果不变；严重度低、非阻断。",
        "falsifiable_check": "检索两种拼写并冻结单一代码表；构造一个 `ε_abs` 跨零的合成情形，检查记录的原因是否指向精度而非门槛标定。",
    },
]

RULINGS = {
    "verdict": "READY_TO_RUN",
    "can_a_barely_trained_control_be_labelled_headroom": "no",
    "blocking_issues": [],
    "non_blocking_issues": [i["id"] for i in ISSUES],
    "seven_changes": {
        "1_untrained_scale_measurement_reconciled": "implemented",
        "2_ceiling_aggregation_level": "implemented",
        "3_inconclusive_reason_codes": "implemented",
        "4_stale_lr_wording_removed": "partially",
        "5_version_ledger": "partially",
        "6_pilot_aggregate_prose": "implemented",
        "7_rho_well_posed": "implemented",
    },
    "note": "第五轮唯一阻断项已修复：误标的未训练门被替换为逐 seed 良态上界 `ln 2^M + 0.5`，熵恒等式降为诊断，且 `H(后验)` 与未训练 KL 取自同一条 512-episode 评估流；按 packet 自身数字无任何配置被训练前拦截。审查者明确判定筛查可运行，五项问题均为低严重度且不改变任何可发出的判词。",
}

OVERALL = (
    "第五轮唯一阻断项已修复。误标的未训练门被移除：上界现为逐 seed 良态界 `ln 2^M + 0.5` nat，熵恒等式只作诊断，"
    "且 `H(posterior)` 与未训练 KL 取自同一条 512-episode 评估流。按 packet 自身数字核对，每个配置都落在 "
    "`[0.10, ln 2^M + 0.5]` 内（C1 1.29–1.43 对 3.27；C2 2.58–2.60、C3 1.49–1.53、C4 1.38–1.45 对 6.05），"
    "故训练前无任何配置被拦截。欠训练的控制不可能被判 `HEADROOM`：`LEARNING_FAILURE` 先于 `SATURATED` 与 `HEADROOM`，"
    "`κ=0.90` 须逐 seed 满足，平台规则要求 500/1000/2000/4000 平坦、自 250 起无逐步上升且全网格无净上升，"
    "跨 LR 门要求第二个注册 LR 同样未饱和，而 LR 是在独立验证流上从两个候选中择优——这对余量主张是保守方向。"
    "七项要求中五项完全落实，第 4、5 项部分落实，但两者都只是标签与账目缺陷（过时的反驳标签，其生效门已在冻结表中完整写出；"
    "过时的审查计数与产物标签，以及一行台账不完整），不改变本筛查能发出的任何判词。"
    "R1–R5 全部为低严重度，按协议自身的比例原则均非阻断；无一改变阈值、统计量、分支顺序或门结果。筛查终于可运行。"
)


def main() -> int:
    packet = json.loads(PACKET.read_text(encoding="utf-8"))
    design = packet["design"]["content"]
    action = packet["action"]["content"]
    outputs = "\n".join(item["content"] for item in packet["action"].get("outputs", []))
    action_text = action + "\n" + outputs
    wanted = {"design": {"design"}, "action": {"action"}, "both": {"design", "action"}}
    for issue in ISSUES:
        found: set[str] = set()
        for quote in issue["evidence"]:
            for name, content in (("design", design), ("action", action_text)):
                if quote in content:
                    found.add(name)
        if not found or not wanted[issue["source"]] <= found:
            raise SystemExit(f"issue {issue['id']} evidence not anchored: {sorted(found)}")
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
    RULINGS_OUT.write_text(
        json.dumps({"schema_version": 1, "packet_sha256": review["packet_sha256"],
                    "review_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
                    "rulings": RULINGS}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"review": str(OUT), "verdict": RULINGS["verdict"], "issues": len(ISSUES),
                      "headroom_possible_for_untrained": RULINGS["can_a_barely_trained_control_be_labelled_headroom"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
