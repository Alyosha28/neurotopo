"""Regenerate the DIR-0002 verification record from the current repository state.

Read-only: computes counts, hashes and per-literal citation cross-checks. No task
samples, no training, no recomputation of frozen metrics.
"""

from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os
import subprocess

OUT = "research/reflections/DIR-0002/verification.json"
DOCS = [
    "research/design/dir0002_direction_options_predesign.md",
    "research/evidence/DIR-0002_direction_options_survey.md",
    "research/prior_art/dir0002_direction_prior_art.md",
]
LITERALS = [
    ("CAL-0001 触顶下界", "0.0003", "research/evidence/auto_research_failure_synthesis.md"),
    ("CAL-0001 触顶上界", "0.0005", "research/evidence/auto_research_failure_synthesis.md"),
    ("CAL-0002 触顶下界", "0.0018", "research/evidence/auto_research_failure_synthesis.md"),
    ("CAL-0002 触顶上界", "0.0023", "research/evidence/auto_research_failure_synthesis.md"),
    ("CAL-0006 dense seed603", "0.06497", "research/evidence/CAL-0006_dynamic_filter_qualification.md"),
    ("CAL-0006 失败线", "0.08", "research/evidence/CAL-0006_dynamic_filter_qualification.md"),
    ("EXP-0009 C0 KL 值", "0.012463", "research/evidence/GOV-0002_gate1_portfolio_entry.md"),
    ("EXP-0009 余量下限", "0.07", "research/evidence/GOV-0002_gate1_portfolio_entry.md"),
    ("EXP-0004 控制余量下界", "32.96", "research/evidence/GOV-0002_gate1_portfolio_entry.md"),
    ("EXP-0004 M3 低幅", "9.10pp", "research/evidence/GOV-0002_gate1_portfolio_entry.md"),
    ("EXP-0011 比值下界", "1.4811", "research/negative_results/exp0011_v6_resource_mismatch.md"),
    ("EXP-0011 比值上界", "1.7629", "research/negative_results/exp0011_v6_resource_mismatch.md"),
    ("EXP-0011 失败线", "1.12", "research/negative_results/exp0011_v6_resource_mismatch.md"),
    ("EXP-0012 D8 最小角实测", "0.044597", "research/negative_results/exp0012_matrix_no_candidate.md"),
    ("EXP-0012 D8 下界", "0.05", "research/negative_results/exp0012_matrix_no_candidate.md"),
    ("EXP-0012 D32 下界实测", "5.325199", "research/negative_results/exp0012_matrix_no_candidate.md"),
    ("EXP-0012 D32 上界实测", "19.742137", "research/negative_results/exp0012_matrix_no_candidate.md"),
    ("EXP-0012 D32 天花板", "4.5", "research/negative_results/exp0012_matrix_no_candidate.md"),
    ("TH-0001 4/8 权重精确滤波", "4/8-log-weight exact filters", "research/stage_gates.md"),
    (
        "TH-0001 同维下界适用单流与分组",
        "same state-dimension lower bound applies to single and grouped H64 states",
        "research/stage_gates.md",
    ),
    ("open_questions 学习效率交互可另行注册", "K×model interaction on learning efficiency", "research/open_questions.md"),
    ("open_questions 4/8 权重", "only4/8 log weights", "research/open_questions.md"),
    ("MON-0001 不排除新注册样本效率问题", "不排除不同候选/任务上新注册的样本效率问题", "research/open_questions.md"),
    ("GOV-0002 块即竞争假设不能推论", "块即竞争假设", "research/evidence/GOV-0002_gate1_portfolio_entry.md"),
    ("0049 最后一次任务生成器尝试", "最后一次任务生成器尝试", "research/decisions/0049_g1rs_task_search_cap.md"),
    ("EXP-0013 阶段显露预审最强拒绝", "四条独立流恰好迎合RS4", "docs/EXP-0013_phase_reveal_predecision_review.md"),
    ("EXP-0013 v1 预审 worker 缺口", "一次从VALIDATED跑到ANALYZED", "docs/EXP-0013_v1_pre_run_review.md"),
    ("0053 资格/效果拆轮理由", "不能在一个worker内完成SC/D资格", "research/decisions/0053_exp0013_split_qualification_and_formal.md"),
    ("0043 强单流可内部模拟K槽", "强单流可内部模拟K槽", "research/loop/current_frontier.md"),
    ("0043 强控制内部形成槽不失格", "internally form slots", "research/stage_gates.md"),
    ("0053 每臂自选 LR 规则", "控制每臂用自己的预写性能优选LR", "research/decisions/0053_exp0013_split_qualification_and_formal.md"),
]


def sha256(path: str) -> str:
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def size(path: str) -> int:
    return os.path.getsize(path)


def check(figure: str, literal: str, path: str) -> dict[str, object]:
    with open(path, encoding="utf-8") as stream:
        lines = [n + 1 for n, line in enumerate(stream.read().splitlines()) if literal in line]
    return {"figure": figure, "literal": literal, "source": path, "found": bool(lines), "lines": lines}


def build() -> dict[str, object]:
    governance = list(
        {
            p
            for pattern in ("research/decisions/*.md", "research/reflections/**/*.*", "docs/*review*.md", "docs/*audit*.md")
            for p in glob.glob(pattern, recursive=True)
            if os.path.isfile(p) and not os.path.basename(p).startswith("._")
        }
    )
    research_records = [
        p
        for pattern in ("research/evidence/*.md", "research/negative_results/*.md", "research/design/*.md", "research/prior_art/*.md")
        for p in glob.glob(pattern)
        if not os.path.basename(p).startswith("._")
    ]
    analyses = sorted(glob.glob("artifacts/EXP-*/*/analysis.json"))
    last = "artifacts/EXP-0009/v1/analysis.json"
    checks = [check(*item) for item in LITERALS]
    return {
        "schema": "dir0002_verification_v2",
        "task": "DIR-0002 方向组合调查（非实验轮次，只读）",
        "read_only_actions": "未生成任务样本、未训练、未重算冻结指标、未改阈值/决定/阶段门/账本",
        "scope": (
            "citation_literals 逐条只证明所列字面量出现在所列文件中；它不验证该字面量所在句子的解释、"
            "不验证区间以外的数值、也不构成对任何推断的支持。计数为生成本文件时的仓库状态。"
        ),
        "generator": "research/reflections/DIR-0002/verification_builder.py",
        "portfolio_stats": {
            "commits": int(subprocess.run(["git", "rev-list", "--count", "HEAD"], capture_output=True, text=True).stdout.strip()),
            "decisions": len([p for p in glob.glob("research/decisions/*.md") if not os.path.basename(p).startswith("._")]),
            "reflection_packages": len([d for d in glob.glob("research/reflections/*") if os.path.isdir(d)]),
            "governance_review_text_files": len(governance),
            "governance_review_text_bytes": sum(size(p) for p in governance),
            "research_record_text_files": len(research_records),
            "task_level_analysis_artifacts": len(analyses),
            "task_level_analysis_bytes": sum(size(p) for p in analyses),
            "ratio_governance_bytes_over_analysis_bytes": round(
                sum(size(p) for p in governance) / sum(size(p) for p in analyses), 2
            ),
            "ratio_governance_files_over_analysis_files": round(len(governance) / len(analyses), 1),
            "last_run_with_task_metrics": {
                "path": last,
                "local_mtime": datetime.datetime.fromtimestamp(os.path.getmtime(last)).strftime("%Y-%m-%d %H:%M"),
                "bytes": size(last),
            },
            "latest_analysis_artifact": {
                "path": "artifacts/EXP-0011/v6/analysis.json",
                "bytes": size("artifacts/EXP-0011/v6/analysis.json"),
                "note": "资源闸门占位，零任务信息",
            },
            "repo_span_local": {"first_commit": "2026-09-23 21:03", "last_commit": "2026-09-24 15:52"},
        },
        "citation_literals": checks,
        "citation_literals_all_found": all(item["found"] for item in checks),
        "prior_art_execution": {
            "status": "NOT_EXECUTED",
            "web_search": {
                "attempts": 3,
                "error": "HTTP 402 Insufficient Balance",
                "endpoint": "api.deepseek.com/anthropic/v1/messages",
            },
            "web_fetch": {
                "hosts_tried": [
                    "arxiv.org",
                    "export.arxiv.org",
                    "example.com",
                    "google.com",
                    "api.semanticscholar.org",
                    "openreview.net",
                    "r.jina.ai",
                ],
                "error": "resolves to a non-public IP address",
            },
            "sources_retrieved": 0,
            "delegated_checker": "独立子代理报告同样的两类失败；其模型回忆未采纳，标识符未记录",
        },
        "outputs": [{"path": p, "sha256": sha256(p), "bytes": size(p)} for p in DOCS],
        "forbidden_inferences": [
            "H3 一般证伪",
            "Gate 1 或任何阶段门状态变化",
            "先例新颖性",
            "某方向必然有效",
            "追认任何冻结结果",
            "注册实验或授权训练",
        ],
    }


def main() -> int:
    document = build()
    with open(OUT, "w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "output": OUT,
                "literals": len(document["citation_literals"]),
                "all_found": document["citation_literals_all_found"],
                "ratios": [
                    document["portfolio_stats"]["ratio_governance_bytes_over_analysis_bytes"],
                    document["portfolio_stats"]["ratio_governance_files_over_analysis_files"],
                ],
                "docs": {item["path"]: item["sha256"][:12] for item in document["outputs"]},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
