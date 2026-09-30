"""Minimal-input packet and evidence checks for independent post-task reflection.

This module validates records; a separate fresh-context model supplies the review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
PACKET_BUNDLE_VERSION = 2
SOURCES = {"design", "action", "both"}
SEVERITIES = {"high", "medium", "low"}
DISPOSITIONS = {"accept", "reject", "defer"}
DECISION_IMPACTS = {"UNCHANGED", "REASSESSMENT_REQUIRED"}
ACTION_OUTPUT_KINDS = {"manifest", "verification", "decision"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes())


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Nonempty string required: {field}")
    return value


def _part(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    if not raw or len(raw) > 250_000:
        raise ValueError("Packet input must contain 1..250000 bytes")
    return {"sha256": sha256(raw), "content": raw.decode("utf-8")}


def make_packet(design_path: Path, action_path: Path, output_path: Path,
                action_outputs: list[tuple[str, Path]] | None = None) -> dict[str, Any]:
    """Copy only registered design and selected action outputs into a packet."""
    if output_path.exists():
        raise FileExistsError(output_path)
    design, action = _part(design_path), _part(action_path)
    extras = action_outputs or []
    if extras:
        seen: set[str] = set()
        outputs = []
        for kind, path in extras:
            if kind not in ACTION_OUTPUT_KINDS or kind in seen:
                raise ValueError("Unknown or duplicate action output kind")
            seen.add(kind)
            outputs.append({"kind": kind, **_part(path)})
        action = {**action, "outputs": outputs}
    packet = {"schema_version": PACKET_BUNDLE_VERSION if extras else SCHEMA_VERSION,
              "design": design, "action": action}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(packet, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return packet


def validate_packet(path: Path) -> dict[str, Any]:
    packet = _load(path)
    if set(packet) != {"schema_version", "design", "action"} or packet["schema_version"] not in {SCHEMA_VERSION, PACKET_BUNDLE_VERSION}:
        raise ValueError("Packet may contain only schema_version, design and action")
    for name in ("design", "action"):
        part = packet[name]
        required = {"sha256", "content"}
        if name == "action" and packet["schema_version"] == PACKET_BUNDLE_VERSION:
            required.add("outputs")
        if not isinstance(part, dict) or set(part) != required:
            raise ValueError(f"Malformed packet {name}")
        content = _nonempty(part["content"], name)
        if sha256(content.encode("utf-8")) != part["sha256"]:
            raise ValueError(f"Packet {name} digest mismatch")
    if packet["schema_version"] == PACKET_BUNDLE_VERSION:
        outputs = packet["action"]["outputs"]
        if not isinstance(outputs, list) or not outputs:
            raise ValueError("Action output bundle must be nonempty")
        seen: set[str] = set()
        for output in outputs:
            if not isinstance(output, dict) or set(output) != {"kind", "sha256", "content"}:
                raise ValueError("Malformed action output")
            kind = output["kind"]
            if kind not in ACTION_OUTPUT_KINDS or kind in seen:
                raise ValueError("Unknown or duplicate action output kind")
            seen.add(kind)
            content = _nonempty(output["content"], f"action.{kind}")
            if sha256(content.encode("utf-8")) != output["sha256"]:
                raise ValueError(f"Action output {kind} digest mismatch")
    return packet


def validate_review(packet_path: Path, review_path: Path) -> dict[str, Any]:
    packet = validate_packet(packet_path)
    review = _load(review_path)
    if set(review) != {"schema_version", "packet_sha256", "reviewer", "issues", "overall_assessment"}:
        raise ValueError("Review schema mismatch")
    if review["schema_version"] != SCHEMA_VERSION or review["packet_sha256"] != file_sha256(packet_path):
        raise ValueError("Review does not match packet")
    reviewer = review["reviewer"]
    if not isinstance(reviewer, dict) or set(reviewer) != {"model", "effort", "context"}:
        raise ValueError("Reviewer metadata required")
    for key in reviewer:
        _nonempty(reviewer[key], f"reviewer.{key}")
    if reviewer["context"] != "fresh_only_packet":
        raise ValueError("Review must declare fresh_only_packet context")
    _nonempty(review["overall_assessment"], "overall_assessment")
    if not isinstance(review["issues"], list):
        raise ValueError("issues must be a list")
    ids: set[str] = set()
    fields = {"id", "source", "severity", "evidence", "problem", "why_it_matters", "falsifiable_check"}
    for issue in review["issues"]:
        if not isinstance(issue, dict) or set(issue) != fields:
            raise ValueError("Issue schema mismatch")
        ident = _nonempty(issue["id"], "issue.id")
        if ident in ids:
            raise ValueError("Duplicate issue ID")
        ids.add(ident)
        if issue["source"] not in SOURCES or issue["severity"] not in SEVERITIES:
            raise ValueError("Invalid issue source or severity")
        raw_evidence = issue["evidence"]
        quotes = [raw_evidence] if isinstance(raw_evidence, str) else raw_evidence
        if not isinstance(quotes, list) or not 1 <= len(quotes) <= 4:
            raise ValueError("Issue evidence must be one quote or a list of 1..4 quotes")
        design_texts = [packet["design"]["content"]]
        action_texts = [packet["action"]["content"]]
        action_texts += [output["content"] for output in packet["action"].get("outputs", [])]
        found_sources: set[str] = set()
        for quote in quotes:
            excerpt = _nonempty(quote, "issue.evidence")
            found = set()
            if any(excerpt in content for content in design_texts):
                found.add("design")
            if any(excerpt in content for content in action_texts):
                found.add("action")
            if not found or (issue["source"] != "both" and issue["source"] not in found):
                raise ValueError(f"Issue {ident} evidence is absent from named packet source")
            found_sources |= found
        if issue["source"] == "both" and found_sources != {"design", "action"}:
            raise ValueError(f"Issue {ident} must quote both design and action")
        for key in ("problem", "why_it_matters", "falsifiable_check"):
            _nonempty(issue[key], f"issue.{key}")
    return review


def validate_reaction(packet_path: Path, review_path: Path, reaction_path: Path) -> dict[str, Any]:
    packet = validate_packet(packet_path)
    review = validate_review(packet_path, review_path)
    reaction = _load(reaction_path)
    fields = {"schema_version", "packet_sha256", "review_sha256", "responses", "decision_impact", "next_action"}
    if set(reaction) != fields or reaction["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Reaction schema mismatch")
    if reaction["packet_sha256"] != file_sha256(packet_path) or reaction["review_sha256"] != file_sha256(review_path):
        raise ValueError("Reaction does not match packet and review")
    if reaction["decision_impact"] not in DECISION_IMPACTS:
        raise ValueError("Invalid decision impact")
    _nonempty(reaction["next_action"], "next_action")
    if not isinstance(reaction["responses"], list):
        raise ValueError("responses must be a list")
    expected = {issue["id"] for issue in review["issues"]}
    seen: set[str] = set()
    for response in reaction["responses"]:
        if not isinstance(response, dict) or set(response) != {"id", "disposition", "reason", "action", "evidence_paths"}:
            raise ValueError("Response schema mismatch")
        ident = _nonempty(response["id"], "response.id")
        if ident in seen:
            raise ValueError("Duplicate response ID")
        seen.add(ident)
        if response["disposition"] not in DISPOSITIONS:
            raise ValueError("Invalid disposition")
        _nonempty(response["reason"], "response.reason")
        _nonempty(response["action"], "response.action")
        paths = response["evidence_paths"]
        if not isinstance(paths, list) or (response["disposition"] == "accept" and not paths):
            raise ValueError("Accepted issue needs action evidence paths")
        for raw in paths:
            path = Path(_nonempty(raw, "evidence_path"))
            if path.resolve() in {packet_path.resolve(), review_path.resolve(), reaction_path.resolve()}:
                raise ValueError("Reflection files cannot serve as action evidence")
            if not path.is_file() or not path.read_bytes():
                raise ValueError(f"Missing or empty action evidence: {path}")
            packet_input_hashes = {packet["design"]["sha256"], packet["action"]["sha256"]}
            packet_input_hashes.update(output["sha256"] for output in packet["action"].get("outputs", []))
            if file_sha256(path) in packet_input_hashes:
                raise ValueError("Original design or action output cannot serve as new reaction evidence")
    if seen != expected:
        raise ValueError("Every issue must receive exactly one response")
    return reaction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    packet = subs.add_parser("packet")
    packet.add_argument("--design", type=Path, required=True)
    packet.add_argument("--action", type=Path, required=True)
    packet.add_argument("--output", type=Path, required=True)
    packet.add_argument("--action-output", action="append", default=[], metavar="KIND=PATH")
    check = subs.add_parser("check")
    check.add_argument("--packet", type=Path, required=True)
    check.add_argument("--review", type=Path, required=True)
    check.add_argument("--reaction", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "packet":
            outputs = []
            for item in args.action_output:
                if "=" not in item:
                    raise ValueError("--action-output must use KIND=PATH")
                kind, path = item.split("=", 1)
                outputs.append((kind, Path(path)))
            result = make_packet(args.design, args.action, args.output, outputs)
            summary = {"output": str(args.output), "packet_sha256": file_sha256(args.output),
                       "design_sha256": result["design"]["sha256"], "action_sha256": result["action"]["sha256"]}
        else:
            result = validate_reaction(args.packet, args.review, args.reaction)
            summary = {"valid": True, "issues": len(validate_review(args.packet, args.review)["issues"]),
                       "decision_impact": result["decision_impact"]}
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
