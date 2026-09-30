"""Post-review nonfinite-input guard; original frozen planning code is retained."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from dir0007_power_plan import classify


def classify_checked(lower: list[float], upper: list[float], *, complete: bool = True,
                     gates: bool = True) -> str:
    if (len(lower) != 4 or len(upper) != 4
            or any(not math.isfinite(x) for x in lower + upper)
            or any(l > u for l, u in zip(lower, upper))):
        return "INCONCLUSIVE_EXECUTION"
    return classify(lower, upper, complete=complete, gates=gates)


def main() -> None:
    original = json.loads(Path("research/design/dir0007_power_plan.json").read_text())
    for row in original["decision_cases"]:
        assert classify_checked(row["lower"], row["upper"], **row["options"]) == row["expected"]
    cases = []
    for bad_name, bad in [("nan", float("nan")), ("positive_inf", float("inf")), ("negative_inf", -float("inf"))]:
        for side in ["lower", "upper"]:
            for index in range(4):
                lower, upper = [.03]*4, [.04]*4
                (lower if side == "lower" else upper)[index] = bad
                actual = classify_checked(lower, upper)
                assert actual == "INCONCLUSIVE_EXECUTION"
                cases.append(dict(nonfinite=bad_name, side=side, index=index, result=actual))
    assert classify_checked([.03]*3, [.04]*4) == "INCONCLUSIVE_EXECUTION"
    assert classify_checked([.05]*4, [.04]*4) == "INCONCLUSIVE_EXECUTION"
    result = dict(scope="POST_REVIEW_DECISION_GUARD_ONLY", original_cases_preserved=8,
                  nonfinite_cases=cases, malformed_cases_passed=2, all_pass=True,
                  original_nan_bug_reproduced=(classify([.03]*4, [float("nan")]*4) == "SUPPORTED_LOCAL_ONLY"),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  research_samples=0, optimization_updates=0, included_in_review_packet=False)
    target=Path("research/design/dir0007_decision_guard.json")
    if target.exists():
        raise FileExistsError("Preserve prior validation output")
    target.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "nonfinite_cases"}))


if __name__ == "__main__":
    main()
