"""Build the evidence bundle for the DIR-0005 audit review.

The audit's decisive claims are about repository records (twelve closures, CAL-0008
saturation, the v3 matching rule, three review verdicts). A packet-only reviewer cannot
check them unless the records travel with the packet, so every cited record is embedded
**verbatim**, each with its own sha256. `stage_gates.md` is the one exception: only two
declared line ranges are embedded (the gate table and the v3 rule section) because the
rest of that file is unrelated history; the full-file sha256 and the line ranges are
recorded so the slice is auditable rather than silent.

Deterministic: same inputs -> same bytes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FULL = [
    ("decision", "research/decisions/0063_cal0008_close_configs.md",
     "CAL-0008: the four fixed configurations saturate, closing same-family difficulty inflation."),
    ("decision", "research/decisions/0068_gate1_matching_clause.md",
     "Section 5 lists the twelve non-regression closures the audit's ledger refers to."),
    ("decision", "research/decisions/0071_g1matchv3_reentry_screen.md",
     "The v3 re-entry screen: zero revivals, and no closure was caused by the matching rule."),
    ("decision", "research/decisions/0073_routing_alignment_direction_stopped.md",
     "The stopping record whose reasoning tension T1 generalises."),
    ("decision", "research/decisions/0080_entry_rule_architecture_organization.md",
     "Entry policy v2 and the recorded Generic gap (self-binding requirement)."),
    ("decision", "research/decisions/0081_dir0004_stopped.md",
     "The DIR-0004 stop, including the two structural fatal findings the audit cites."),
    ("review", "research/reflections/DIR-0004-PREDESIGN/review.json",
     "The nine accepted issues of the DIR-0004 review, verbatim."),
    ("evidence", "research/evidence/gate1_benchmark_baseline_audit.md",
     "The record showing EXP-0004's prespecified matched control had substantial measured "
     "primary-endpoint headroom, which corrects the v0 audit's 'no measured-headroom task' claim."),
    ("review", "research/reflections/DIR-0005-AUDIT/review.json",
     "The v0 audit review (AUDIT_NEEDS_NARROWING, eight issues) that v1 answers."),
    ("audit_version", "research/evidence/DIR-0005_candidate_spec_audit_v1.md",
     "The v1 audit text, so the v2 changes can be checked against what they replace."),
    ("review", "research/reflections/DIR-0005-AUDIT-V1/review.json",
     "The v1 review (V1_NEEDS_NARROWING, six issues) that v2 answers."),
]

SLICES = [
    ("gate_document", "research/stage_gates.md", [[17, 30], [96, 115]],
     "Gate table (Gate 0/1 requirements and states) and the DIR-0002 label plus the G1-MATCH-v3 rule section."),
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    artifacts = []
    for kind, rel, why in FULL:
        content = (ROOT / rel).read_text(encoding="utf-8")
        artifacts.append({"kind": kind, "path": rel, "why_needed": why, "verbatim": "full_file",
                          "sha256": sha256_text(content), "bytes": len(content.encode("utf-8")),
                          "content": content})
    for kind, rel, ranges, why in SLICES:
        path = ROOT / rel
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        parts = ["".join(lines[a - 1:b]) for a, b in ranges]
        content = "\n".join(parts)
        artifacts.append({
            "kind": kind, "path": rel, "why_needed": why,
            "verbatim": f"line_ranges {ranges} of {len(lines)} lines",
            "full_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sha256": sha256_text(content), "bytes": len(content.encode("utf-8")),
            "content": content,
        })
    bundle = {
        "schema": "dir0005_review_bundle_v3",
        "purpose": ("Every repository record the DIR-0005 audit relies on, embedded verbatim, "
                    "so the audit can be checked inside the packet without browsing the repository."),
        "artifacts": artifacts,
    }
    out = ROOT / "research/evidence/dir0005_review_bundle_v3.json"
    payload = json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out.write_text(payload, encoding="utf-8")
    print(json.dumps({"output": str(out.relative_to(ROOT)),
                      "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                      "bytes": len(payload.encode("utf-8")),
                      "artifacts": {a["path"]: a["sha256"][:16] for a in artifacts}},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
