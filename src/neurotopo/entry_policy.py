"""Structural entry checks for prospective NeuroTopo research questions.

Passing this check opens QUESTION only; it does not validate scientific claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


POLICY = "gate1_candidate_first_v2"  # v2: project_gate1 also admits architecture_organization factors
TRACKS = {"project_gate1", "field_reference", "independent_benchmark"}
BRANCHES = {"success", "failure", "borderline"}
FACTOR_FIELDS = {"kind", "name", "manipulation", "ablation", "prediction",
                 "actual_primary_control", "state_matching", "parameter_matching",
                 "executed_compute_matching", "prior_art_boundary"}
CONTRACT_FIELDS = {"schema_version", "track", "experiment_id", "direction", "question",
                   "purpose", "gate1_claim", "resource_cap", "task_search", "branches", "factor"}


class EntryRejected(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EntryRejected("MISSING_FIELD", f"Nonempty text required: {field}")
    return value.strip()


def _exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise EntryRejected("SCHEMA_MISMATCH", f"Expected {sorted(expected)} in {field}")
    return value


def validate_entry_contract(path: Path, *, experiment: str, direction: str,
                            question: str) -> dict[str, Any]:
    """Check one immutable candidate/benchmark entry packet before loop mutation."""
    try:
        raw = path.read_bytes()
        contract = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise EntryRejected("INVALID_CONTRACT", str(error)) from error
    if not raw:
        raise EntryRejected("INVALID_CONTRACT", "Empty contract")
    _exact_keys(contract, CONTRACT_FIELDS, "contract")
    if type(contract["schema_version"]) is not int or contract["schema_version"] != 1:
        raise EntryRejected("SCHEMA_MISMATCH", "schema_version must be 1")
    for field, expected in (("experiment_id", experiment), ("direction", direction),
                            ("question", question)):
        if _text(contract[field], field) != expected:
            raise EntryRejected("QUESTION_MISMATCH", f"{field} differs from begin request")
    _text(contract["purpose"], "purpose")
    track = contract["track"]
    if not isinstance(track, str) or track not in TRACKS:
        raise EntryRejected("INVALID_TRACK", str(track))
    if type(contract["gate1_claim"]) is not bool or contract["gate1_claim"] != (track == "project_gate1"):
        raise EntryRejected("CLAIM_TRACK_MISMATCH", "Only project_gate1 may claim project Gate 1")

    cap = _exact_keys(contract["resource_cap"], {"max_updates", "max_wall_seconds"}, "resource_cap")
    if type(cap["max_updates"]) is not int or cap["max_updates"] < 0:
        raise EntryRejected("INVALID_BUDGET", "max_updates must be a nonnegative integer")
    if (type(cap["max_wall_seconds"]) not in (int, float)
            or not math.isfinite(cap["max_wall_seconds"]) or cap["max_wall_seconds"] <= 0):
        raise EntryRejected("INVALID_BUDGET", "max_wall_seconds must be positive")
    search = _exact_keys(contract["task_search"],
                         {"max_configurations", "retain_all_attempts", "heldout_rule"}, "task_search")
    if (type(search["max_configurations"]) is not int or search["max_configurations"] < 1
            or search["retain_all_attempts"] is not True):
        raise EntryRejected("INVALID_SEARCH_BUDGET", "Finite task menu and full attempt retention required")
    _text(search["heldout_rule"], "task_search.heldout_rule")
    branches = _exact_keys(contract["branches"], BRANCHES, "branches")
    for branch in sorted(BRANCHES):
        _text(branches[branch], f"branches.{branch}")

    factor = contract["factor"]
    if factor is None:
        if track != "independent_benchmark":
            raise EntryRejected("MISSING_FACTOR", "Gate 1 or field reference entry requires a named factor")
    else:
        _exact_keys(factor, FACTOR_FIELDS, "factor")
        for field in FACTOR_FIELDS:
            _text(factor[field], f"factor.{field}")
        if factor["kind"] not in {"h3_candidate", "architecture_organization", "reference_k", "independent"}:
            raise EntryRejected("INVALID_FACTOR", "Unknown factor.kind")
        if track == "project_gate1" and factor["kind"] == "reference_k":
            raise EntryRejected("PROJECT_ELIGIBILITY_UNRESOLVED",
                                "Reference K factor may use field_reference; project scoring needs a prospective rule")
        if track == "project_gate1" and factor["kind"] not in {"h3_candidate", "architecture_organization"}:
            raise EntryRejected("TRACK_MISMATCH",
                                "project_gate1 requires an H3 candidate or a declared architecture_organization factor")
        if track == "field_reference" and factor["kind"] != "reference_k":
            raise EntryRejected("TRACK_MISMATCH", "field_reference requires reference_k")
    return {"track": track, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw),
            "policy": POLICY, "scientific_approval": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_entry_contract(args.contract, experiment=args.experiment,
                                         direction=args.direction, question=args.question)
    except EntryRejected as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
