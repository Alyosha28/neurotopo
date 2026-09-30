"""Assemble the G1-KSI-0001 rule-ruling review record from the packet-only reviewer output.

Issue text is the reviewer's; `source` labels are checked against the packet anchors.
The rulings object is written separately because review.json's schema admits only
schema_version, packet_sha256, reviewer, issues and overall_assessment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKET = Path("research/reflections/G1-KSI-0001-RULINGS-V1/packet.json")
OUT = Path("research/reflections/G1-KSI-0001-RULINGS-V1/review.json")
RULINGS_OUT = Path("research/reflections/G1-KSI-0001-RULINGS-V1/rulings.json")

ISSUES = [
    {
        "id": "G1-KSI-0001-RULINGS-V1-R1",
        "source": "both",
        "severity": "high",
        "evidence": [
            "两者只差一个在任何资格样本前固定的耦合矩阵",
            "且耦合同时改变难度。",
            '"gap_F_minus_X_normalised": 0.026051926739033338',
            "难度差数字，均已逐条写入请求正文",
        ],
        "problem": "“仅耦合不同”只对核参数成立，对学习器看到的数据分布不成立：X 的平稳律不是 p_J（偏差 0.0596，约为一个均匀格点质量 0.0625 的 95%），而 F 由构造即为均匀；难度代理相差 0.0261（SE 0.0049，5.3 SE，X 更易）。请求报告了 0.0596 却从未报告该难度差，且协议要求的反驳 A-R2 在请求的反驳段中缺失——该段只回答了 A-R1；因此“难度差数字已写入请求正文”这一说法为假。",
        "why_it_matters": "5.3 SE 的难度差加上未实现的冻结先验，会混淆 K × 耦合交互。",
        "falsifiable_check": "在注册的 n 下重算难度差与其 SE；要求 |gap| 落在预注册界内，并把实际平稳律（而非 p_J）注册为 X 的生成对象。另更正 X 的边缘容差（实为 1.1e-15，不是 <3e-16）。",
    },
    {
        "id": "G1-KSI-0001-RULINGS-V1-R2",
        "source": "action",
        "severity": "high",
        "evidence": [
            "K 个独立递归块 vs 资源匹配稀疏控制 + 合成后验任务",
            "这一宽问题自 EXP-0004 起未变。",
            "(ii) 主终点由末端性能/OOD headroom 改为学习曲线 `AUC_B`（＋终端非劣）",
        ],
        "problem": "新意是被断言的，不是被展示的：请求承认骨架相同、宽问题自 EXP-0004 起未变，却没有给出新的研究问题，也没有给出把“独立性”与“再参数化”区分开的判据。第 (ii) 条差异轴正是只有请求 B 才能授权的终点变更；第 (iv) 条只是卫生要求。",
        "why_it_matters": "0049 保留的路径要求真正新的问题；缺少判别器时 ACCEPT 会使该封顶形同虚设。",
        "falsifiable_check": "去掉第 (ii) 条差异轴后重新评分 A；要求给出一句话的新问题，其否定不被“多轨迹组织有益”所蕴含。",
    },
    {
        "id": "G1-KSI-0001-RULINGS-V1-R3",
        "source": "both",
        "severity": "high",
        "evidence": [
            "而非已冻结的数字，因此可否定性尚未成立。",
            "由**八个单元（2 K × 2 条件 × 臂）的配对功效计算**预注册",
        ],
        "problem": "每一项保护性条件都是未冻结的承诺：没有 δ_min、τ_term、ε、每单元 n、alpha、功效目标、检验统计量、自由度或 K 取值。八个单元下交互检验本身脆弱，其功效取决于未说明的 n。",
        "why_it_matters": "依赖未来承诺的裁定当下不可否定；B 对“降低门槛”这一质疑的答复完全建立在这些数字上。",
        "falsifiable_check": "要求注册配置把上述全部写为数字。",
    },
    {
        "id": "G1-KSI-0001-RULINGS-V1-R4",
        "source": "action",
        "severity": "medium",
        "evidence": [
            "最终性能终点在本项目已四次不可用",
            "是**必需 regime**，不是取消资格。",
            "**不降低任何资源或控制标准**",
        ],
        "problem": "四次校准失败说明的是这些**任务**把精确后验交给了匹配控制（任务缺乏末端余量），请求却把它读作“终点类别不可用”，并且从未考虑一个控制不饱和的任务。“不降低任何资源或控制标准”是在断言争议中的结论。",
        "why_it_matters": "若缺陷在于任务过于容易，那么更换终点就是以移动判据来保全一个已被封顶的方向。",
        "falsifiable_check": "在 packet 内引用 CAL/EXP-0009 的数字，并说明为何不存在一个保留末端余量的可接受任务。",
    },
    {
        "id": "G1-KSI-0001-RULINGS-V1-R5",
        "source": "action",
        "severity": "medium",
        "evidence": [
            "已用配对种子、每臂预注册 LR、同 kernel 与交互设计（条件依赖的斜率差难以由伪影产生）",
        ],
        "problem": "对伪影的防御未被具体化：配对种子处理的是种子方差，预注册 LR 处理的是调参不对称，而交互设计并不排除依赖条件的伪影，例如损失地形 × 初始化或 batch 顺序。",
        "why_it_matters": "由伪影驱动的 AUC_B 交互正属于决定0043 排除的效应类别。",
        "falsifiable_check": "预注册一个扰动探针（LR 缩放或 batch 顺序置换），并规定符号或显著性翻转即判为伪影。",
    },
    {
        "id": "G1-KSI-0001-RULINGS-V1-R6",
        "source": "both",
        "severity": "high",
        "evidence": [
            "**设计 v3 的全文不在 packet 内**",
            "这些检查只核对解析主张与实现一致性，不构成任务资格、控制余量、功效或任何架构效果证据。",
        ],
        "problem": "packet 无法支撑该裁定：v3 的 §2/§4.7/§5 以及已关闭方向线的生成器、终点与问题都不在其中，因此生成器同一性与问题独立性无法核查；核对是 oracle 侧的、在 T=4 而非 T=16、承认曾存在共享约定缺陷，且自我声明不涉及资格、控制余量或功效。",
        "why_it_matters": "两个问题都取决于 packet 之外的事实；ACCEPT 将只能建立在请求自身的断言上。",
        "falsifiable_check": "把 v3 的 §2/§4.7/§5 以及 EXP-0012/0013 与 G1-RS 的记录放进 packet。",
    },
]

RULINGS = {
    "question_1_request_A_0049": "NARROW",
    "question_2_request_B_efficiency_endpoint": "NARROW",
    "question_3_packet_sufficient_for_ruling": "no",
    "conditions_if_narrowed": (
        "A（仅非项目轨 `independent_benchmark`）：(A1) 在任何资格样本前注册一句话的新问题，其否定不被"
        "“多轨迹组织有益”蕴含，并逐字记录 EXP-0012/0013 的问题、生成器、终点与估计对象的记录 ID；"
        "(A2) 去掉第 (ii) 条差异轴后重新评分 A；(A3) 在注册的 n 下报告 `gap_F_minus_X_normalised` 及其 SE，"
        "并要么在两个条件下以预注册界守住 §4.7 门，要么把难度预注册为协变量并固定模型与自由度；"
        "(A4) 把核的实际平稳律（而非 `p_J`）注册为 X 的生成对象；(A5) 记录不主张任何 Gate 1 计分，"
        "在资格前固定 K、实际匹配控制与预测，并声明不重开 EXP-0013 生成器。"
        "B（主终点类别仅在下述全部满足时允许）：(B1) 在任何效果轮前冻结 `AUC_B` 定义、δ_min 及其最小有意义效应理由、"
        "τ_term 与单侧非劣检验、实际主控制的 ε、每单元 n、alpha、δ_min 处功效 0.8、以及确切的交互统计量与自由度，"
        "资格后不得修改；(B2) 具名两个 K 取值，并称之为对照而非剂量反应；(B3) 使 τ_term 成为门——违反即判 `INCONCLUSIVE`，"
        "不论 `AUC_B` 如何；(B4) 预注册伪影探针及其判废规则；(B5) 使不可行停止可机器核验，并禁止事后下调 δ_min；"
        "(B6) 要求资格门报告实际控制的 `AUC_B` 及其 CI 与该估计量的最小可分辨差，并按决定0044 要求控制达 ε，否则不启动效果轮；"
        "(B7) 保持 MON-0001/决定0009 路径关闭，不放宽任何资源或控制匹配。"
    ),
    "what_would_change_the_verdict": (
        "A 转 ACCEPT：以记录锚定地写出已关闭方向线的问题，并给出判别器说明新问题的答案不被旧问题蕴含，"
        "且在预注册界内给出难度匹配的 F/X 对。A 转 REJECT：证据显示 EXP-0012/0013 已使用 Markov 潜过程、"
        "或已估计过 K × 耦合交互。B 转 ACCEPT：冻结 δ_min、τ_term、ε 并完成功效计算，证明在可行 n 内 δ_min 处功效 0.8，"
        "且资格轮中实际控制达 ε 而终端非劣完好。B 转 REJECT：冻结数字显示有意义功效不可达，或为使效应存活而放宽 τ_term。"
        "问题 3 转 yes：把 v3 的 §2/§4.7/§5 与已关闭方向线的记录放进 packet。"
    ),
}

OVERALL = (
    "两项请求都坦诚：报告了 0.0596 的平稳律偏差，把 oracle 侧事实标为假设，也未主张任何阶段门变化。坦诚并不等于 packet 充分。"
    "A 的“仅耦合不同”对核参数成立，对学习器所见的分布不成立——X 的平稳律偏离 p_J 达一个均匀格点质量的 95%，难度差 5.3 SE，"
    "而这一被要求正面处理的反驳在请求中从未出现；A 的新意论证还依赖只有 B 才能授权的终点变更。"
    "B 的终点类别按其引用的规则文本可有条件允许，但所有使其不可被游戏的条款都未冻结，其“未降低任何标准”的主张是在断言争议结论。"
    "仅凭 packet 阅读：v3、已关闭方向线的记录、CAL 数字以及全部控制余量与功效证据都不在其中，"
    "因此生成器同一性、问题独立性与不可游戏性在此无法核查。"
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
        json.dumps(
            {
                "schema_version": 1,
                "packet_sha256": review["packet_sha256"],
                "review_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
                "rulings": RULINGS,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"review": str(OUT), "rulings": str(RULINGS_OUT), "issues": len(ISSUES),
                      "A": RULINGS["question_1_request_A_0049"], "B": RULINGS["question_2_request_B_efficiency_endpoint"],
                      "packet_sufficient": RULINGS["question_3_packet_sufficient_for_ruling"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
