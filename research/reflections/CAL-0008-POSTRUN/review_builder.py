"“”Assemble the CAL-0008 post-run science review record from the reviewer output.“”"

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKET = Path("research/reflections/CAL-0008-POSTRUN/packet.json")
OUT = Path("research/reflections/CAL-0008-POSTRUN/review.json")
RULINGS_OUT = Path("research/reflections/CAL-0008-POSTRUN/rulings.json")

ISSUES = [
    {
        "id": "CAL-0008-POSTRUN-R1",
        "source": "both",
        "severity": "high",
        "evidence": [
            "送审 packet 只含：本审查说明、注册设计（判据与分支的权威文本）、结果记录、以及运行产物 `screen_v1.json`。",
            "**它本身是事后撰写的**（运行已完成），因此不得被表述为预注册。",
            "注册设计与判据见 `research/design/cal0008_terminal_headroom_predesign.md`（v7，第六轮预运行审查判 `READY_TO_RUN`",
        ],
        "problem": "packet 只有运行后审查说明（自称非预注册）与结果记录；权威的预设计 v7（判据、分支顺序、§2 规则、§8、门定义）不在其中。",
        "why_it_matters": "预注册合规、`ε` 标定、模型定义核对与 §8 的授权范围都无法审计。",
        "falsifiable_check": "打开预设计 v7，确认其判据、分支顺序与 §8 在 packet 中完全不存在。",
    },
    {
        "id": "CAL-0008-POSTRUN-R2",
        "source": "both",
        "severity": "medium",
        "evidence": [
            "末端 KL 的 3-seed t 区间上界是否都 ≤ `ε_abs = 0.05`",
            "terminal_kl_ci95",
            "0.0003399297217701377",
        ],
        "problem": "`terminal_kl_ci95` 不是 t 区间：C1 半宽 `8.974e-5` 等于 `1.96·s/√3`，而非 `t(0.975,2)·s/√3 = 1.970e-4`（C2–C4 同样差 0.456 倍），故正确的 3-seed 区间宽约 2.2 倍（如 C1 上界 `0.000627`、C2 `0.004191`）。",
        "why_it_matters": "判据用的是 t 上界，而公布的 12–96 倍裕度用的是 z 上界；判词不变，但统计量被标错。",
        "falsifiable_check": "由逐 seed 末端 KL 重算 `t(0.975,2)·s/√3`。",
    },
    {
        "id": "CAL-0008-POSTRUN-R3",
        "source": "action",
        "severity": "medium",
        "evidence": [
            "`C2` 的平台门未过（`250→500` 落差 `0.26–0.30` > `τ=0.01`）",
            "0.02790415659546852",
            "0.05971292406320572",
        ],
        "problem": "所引 `0.26–0.30` 属于**非判定**的 LR=0.001 流；使 `plateau_all_seeds` 为假的**选定 LR** 流其 C2 落差为 `0.0261–0.0279`（seed 0：由 `0.05971292406320572` 降至 `0.0318087674677372`）。",
        "why_it_matters": "唯一未过的门用了一条并未决定它的流来解释，且所引范围无法由 `plateau_gaps_per_seed` 复现；结论本身不受影响。",
        "falsifiable_check": "对两种学习率分别由每行 `kl_curve` 重算 250→500 落差。",
    },
    {
        "id": "CAL-0008-POSTRUN-R4",
        "source": "action",
        "severity": "medium",
        "evidence": ["absolute_residual_ci_lower", "TERMINAL_KL_CI_UPPER_AT_OR_BELOW_EPSILON", "tail_flat"],
        "problem": "四个配置的 `absolute_residual_ci_lower` 均为 false，而判词都是 `SATURATED`、reason 为 `TERMINAL_KL_CI_UPPER_AT_OR_BELOW_EPSILON`，与“绝对残余门单独决定”的说法表面矛盾；`tail_flat` 在 C2 六个单元全为 false，而两个尾窗（`0.0084`、`0.0034`）低于 `τ=0.01`。两个旗标在 packet 内均无定义。",
        "why_it_matters": "决定判词的分支无法审计，判词实际只依赖绝对 `ε`。",
        "falsifiable_check": "查阅注册设计或脚本中的门定义。",
    },
    {
        "id": "CAL-0008-POSTRUN-R5",
        "source": "action",
        "severity": "medium",
        "evidence": [
            "在本项目资源约束与该生成器族下未找到保留末端余量的配置",
            "不存在（在本族与本预算下）",
            "只关闭这 4 个固定配置",
        ],
        "problem": "该句由四个配置（`M` 为 4/8、`T` 为 16/32、`λ` 为 0.4/0.7、一个阻挫变体、图种子 4101/8307）外推到**族级**，而限制条款只关闭那四个；族级这一步依赖 packet 中不存在的预设计 §8。",
        "why_it_matters": "`M=8` 只有单一图种子，四个点无法承担族级“不可能有余量”的主张。",
        "falsifiable_check": "统计 `configs` 中不同的 `(M, T, λ, graph_seed)` 组合数。",
    },
]

RULINGS = {
    "verdict": "RESULTS_NEED_NARROWING",
    "q1_verdict_follows_from_criteria": (
        "有条件成立：按记录中重述的判据（预注册原文缺失，见 R1），全部 24 个单元末端 KL 至多 `0.00863` nat（最差为 C2 seed 1、LR 0.001）；"
        "公布的均值可由各行复现（`0.000430`、`0.003780`、`0.001163`、`0.001362`），区间端点亦相符，故每个上界都 ≤ `ε=0.05`，`CLOSE_CONFIGS` 成立。"
        "分支顺序应用一致：C2 在 `plateau_all_seeds` 为假时仍判 `SATURATED`，`lr_cross_agreement` 四项皆假。稳健性：正确的 t 上界至多 `0.0042` nat，且按逐 seed 取较差 LR 时每个单元仍至多 `0.00863`。"
    ),
    "q2_alternative_explanations": (
        "预算不足不可信：未训练 KL 为 `1.4–2.6` nat，24 个单元全部消除其中 `99.85–99.98%`，最差单元仍比 `ε` 低 5.8 倍。"
        "保留意见：C2 在 `2000→4000` 之间减半（`0.00705`→`0.00363`），而 C1 锚点只覆盖最容易的配置，故只能说“在 `B=4000` 下低于阈值”，不能说收敛。"
        "门槛标定误差有界：`ε=0.05` 是唯一生效的门且其标定依据缺失，但即使 `ε` 降到 `0.0042` 判词仍成立。实现缺陷无法排除：packet 内没有任务规格、发射模型或精确后验代码。"
        "平凡捷径不是替代解释：饱和由对精确后验的 KL 定义。"
    ),
    "q3_failed_gates_reported": (
        "两点都如实报告：C2 平台门（`plateau_all_seeds` 为假，三个 `plateau_ok_per_seed` 皆假）与跨 LR 门（四项 `lr_cross_agreement` 皆假）。"
        "两者都不改变判词：记录的分支顺序让绝对门先绑定，且 24 个单元全部至多 `0.00863` nat。两处报告缺陷：所引 C2 落差来自非判定流（判定值为 `0.0261–0.0279`）；"
        "“两个 LR 都饱和则跨 LR 门必然失败”这一解释是断言而非可核。C2 的失败仅来自 250→500 早窗（尾窗低于 `τ`）。"
    ),
    "q4_inference_boundary_respected": (
        "大体遵守：记录明确否认架构有效性、Gate 1 或阶段门变化、新颖性与对既有判词的追认，只关闭四个固定配置，并把任务建议标为需要新的预注册。"
        "两处越界：族级措辞，以及把 `M/T/λ/阻挫` 的比较作族级解读（`M=8` 只有单一图种子）。"
    ),
    "rejections": {
        "R1_task_too_easy": "部分成立，作用是限制范围而非否定：`0.1–0.4%` 的残余本身表明控制在此不受容量限制，而匹配状态定理使这一点在预期之内。它只禁止族级读法。",
        "R2_budget_insufficient": "不成立：24 个单元全部至多 `0.00863` nat 对 `ε=0.05`，并消除未训练差距的 `99.85–99.98%`。保留意见更窄：C2 在 `2000→4000` 仍减半，故应说“在注册预算下低于阈值”，而非收敛。",
        "R3_model_definition_error": "无法由 packet 排除或确认，记录亦如此承认：没有任务规格、发射模型或精确后验代码，且核对是在重建缺失的 §2 规则。应把结果读作“相对于所记录参照后验的 KL”。",
        "R4_gates_never_bind": "确认且被如实报告：`relative_residual_all_seeds` 与 `lr_cross_agreement` 四项皆假，故只有绝对 `ε` 门在决定。这不是分支逻辑错误，但筛查没有提供独立佐证，且跨 LR 的解释不可核。",
        "R5_plateau_gate_failed": "成立且报告正确：C2 的失败仅来自 250→500 窗口；两个尾窗低于 `τ`。但 C2 在 `2000→4000` 之间减半且 `tail_flat` 在六个 C2 单元全假，故应说“低于阈值”，而非“平坦”。",
    },
    "recommended_verdict_wording": (
        "在注册预算 4000 updates 下，四个注册配置全部 `SATURATED`：稠密 64 单元 GRU 控制的末端 oracle-relative KL 为 `0.00043–0.0038` nat，"
        "全部 24 个单元至多 `0.0087` nat，即使按重算的 3-seed t 区间上界也至多 `0.0042` nat，对 `ε=0.05`。该族在本预算下没有配置保留末端余量，故匹配稠密控制在此不受容量限制。"
        "它只关闭这四个配置、且限于 `B=4000`；不表明该族不可能产生余量、不认证收敛，也不是架构有效性、Gate 1 或新颖性证据。"
    ),
}

OVERALL = (
    "`RESULTS_NEED_NARROWING`。分支决定可复现且稳健：24/24 单元至多 `0.00863` nat，公布的均值与区间相符，`CLOSE_CONFIGS` 由绝对门推出，分支顺序应用一致，两个未过的门均已披露。"
    "缺陷不会翻转判词：区间是 z 基却被标为 3-seed t 区间；所引 C2 平台落差来自非判定流；两个门旗标未定义；族级措辞超出四个抽样配置。"
    "决定本身可靠，但措辞必须收窄；在补交预设计之前，问题 1 只能针对记录中重述的判据认证。"
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
    print(json.dumps({"review": str(OUT), "verdict": RULINGS["verdict"], "issues": len(ISSUES)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
