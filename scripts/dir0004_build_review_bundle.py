"""Build the single verification bundle the DIR-0004 design review is allowed to read.

The bundle embeds every machine output and note the review needs *verbatim*, each with its
own sha256, so a packet-only reviewer never has to trust a path or a summary. Deterministic
by construction: same inputs -> same bytes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS = [
    ("verification", "research/design/dir0004_precision_check.json",
     "Quantised-storage ladder on the linear hyperbolic map: symmetric vs asymmetric bit "
     "allocation, hidden unstable mode. Source of T* = 14 / 45 and of the 1-step-per-bit law."),
    ("verification", "research/design/dir0004_precision_check_observed.json",
     "Same ladder with the expanding coordinate re-observed every step (round-1 control; "
     "note its own eps = 1.0 configuration, which is NOT the affordance screen's control)."),
    ("verification", "research/design/dir0004_affordance_screen.json",
     "Section-4 affordance screen: minimal stored bits b(T) versus horizon, its fitted slope "
     "and bootstrap interval, plus the feasible observation control (sigma <= eps/10)."),
    ("verification", "research/design/dir0004_nonlinear_bitcheck.json",
     "Nonlinear replication of the bits-per-step law on the tent map and logistic r = 4."),
    ("correction", "research/design/dir0004_claim_correction_v0.md",
     "Main agent's own pre-review correction: the v0 separation claim narrowed to C1/C2, the "
     "predictive-formula unit error, and the screen's first parameterisation bug."),
    ("prior_art", "research/prior_art/dir0004_chaos_tracking.md",
     "Prior-art check closing the design's section-9 gap: AUS field, Kalman covariance "
     "collapse into the unstable subspace, textbook predictability time. Metadata-level."),
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    artifacts = []
    for kind, rel, why in ARTIFACTS:
        path = ROOT / rel
        content = path.read_text(encoding="utf-8")
        artifacts.append({
            "kind": kind,
            "path": rel,
            "why_needed": why,
            "sha256": sha256_text(content),
            "bytes": len(content.encode("utf-8")),
            "content": content,
        })
    bundle = {
        "schema": "dir0004_review_bundle_v1",
        "purpose": (
            "Everything the DIR-0004 packet-only design review may read, embedded verbatim. "
            "No repository browsing is required or permitted."
        ),
        "artifacts": artifacts,
    }
    out = ROOT / "research/design/dir0004_review_bundle_v1.json"
    payload = json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out.write_text(payload, encoding="utf-8")
    print(json.dumps({
        "output": str(out.relative_to(ROOT)),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "bytes": len(payload.encode("utf-8")),
        "artifacts": {a["path"]: a["sha256"] for a in artifacts},
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
