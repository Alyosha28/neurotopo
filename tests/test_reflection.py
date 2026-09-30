"""Post-task reflection packet, grounding and loop gate checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurotopo.loop import PHASES, ResearchLoop
from neurotopo.reflection import (file_sha256, make_packet, validate_packet,
                                  validate_reaction, validate_review)


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _records(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    design = tmp_path / "design.md"
    action = tmp_path / "action.json"
    design.write_text("Hypothesis: block recurrence improves matched control.\nPrimary: delay32.\n")
    action.write_text('{"block":0.58,"control":0.67,"status":"completed"}\n')
    audit = tmp_path / "action_audit.md"
    audit.write_text("Recorded a new construct-validity audit after independent review.\n")
    packet = tmp_path / "packet.json"
    make_packet(design, action, packet)
    review = _write(tmp_path / "review.json", {
        "schema_version": 1, "packet_sha256": file_sha256(packet),
        "reviewer": {"model": "independent-test", "effort": "max", "context": "fresh_only_packet"},
        "issues": [{"id": "R1", "source": "design", "severity": "high",
                    "evidence": "Primary: delay32.",
                    "problem": "The task may have a shortcut",
                    "why_it_matters": "A gain may not measure hypothesis maintenance",
                    "falsifiable_check": "Compare to a cache-and-select oracle"}],
        "overall_assessment": "One design weakness requires a targeted check",
    })
    reaction = _write(tmp_path / "reaction.json", {
        "schema_version": 1, "packet_sha256": file_sha256(packet),
        "review_sha256": file_sha256(review),
        "responses": [{"id": "R1", "disposition": "accept",
                       "reason": "The frozen task can be solved by lookup",
                       "action": "Record a construct-validity audit",
                       "evidence_paths": [str(audit)]}],
        "decision_impact": "UNCHANGED", "next_action": "Use a newly registered benchmark question",
    })
    return design, action, packet, review, reaction


def test_packet_contains_only_two_source_texts_and_detects_tampering(tmp_path: Path) -> None:
    design, action, packet, _, _ = _records(tmp_path)
    value = validate_packet(packet)
    assert set(value) == {"schema_version", "design", "action"}
    assert value["design"]["content"] == design.read_text()
    assert value["action"]["content"] == action.read_text()
    assert "design.md" not in packet.read_text()
    with pytest.raises(FileExistsError):
        make_packet(design, action, packet)
    value["extra_context"] = "prior model opinion"
    _write(packet, value)
    with pytest.raises(ValueError, match="only"):
        validate_packet(packet)


def test_review_must_quote_packet_and_reaction_must_cover_every_issue(tmp_path: Path) -> None:
    _, action, packet, review, reaction = _records(tmp_path)
    assert len(validate_review(packet, review)["issues"]) == 1
    assert validate_reaction(packet, review, reaction)["decision_impact"] == "UNCHANGED"
    bad = json.loads(review.read_text())
    bad["issues"][0]["evidence"] = "unsupported external claim"
    _write(review, bad)
    with pytest.raises(ValueError, match="absent"):
        validate_review(packet, review)
    bad["issues"][0]["evidence"] = "Primary: delay32."
    _write(review, bad)
    response = json.loads(reaction.read_text())
    response["review_sha256"] = file_sha256(review)
    response["responses"] = []
    _write(reaction, response)
    with pytest.raises(ValueError, match="Every issue"):
        validate_reaction(packet, review, reaction)
    response["responses"] = [{"id": "R1", "disposition": "accept", "reason": "Agree",
                              "action": "Record audit", "evidence_paths": [str(action.with_name("missing.md"))]}]
    _write(reaction, response)
    with pytest.raises(ValueError, match="Missing or empty"):
        validate_reaction(packet, review, reaction)
    response["responses"][0]["evidence_paths"] = [str(action)]
    _write(reaction, response)
    with pytest.raises(ValueError, match="Original design or action"):
        validate_reaction(packet, review, reaction)


def test_both_source_issue_uses_separate_exact_quotes(tmp_path: Path) -> None:
    _, _, packet, review, _ = _records(tmp_path)
    value = json.loads(review.read_text())
    value["issues"][0]["source"] = "both"
    value["issues"][0]["evidence"] = ["Primary: delay32.", '"control":0.67']
    _write(review, value)
    assert validate_review(packet, review)["issues"][0]["source"] == "both"
    value["issues"][0]["evidence"] = ["Primary: delay32."]
    _write(review, value)
    with pytest.raises(ValueError, match="must quote both"):
        validate_review(packet, review)


def test_new_loop_requires_reflection_and_matches_registered_artifacts(tmp_path: Path) -> None:
    design, action, packet, review, reaction = _records(tmp_path)
    loop = ResearchLoop(tmp_path / "live")
    loop.init(design)
    loop.begin("test_direction", "Does the effect hold?", "EXP-0099")
    for phase in PHASES[1:-1]:
        loop.advance(phase, design if phase == "REGISTERED" else action)
    loop.advance("DECIDED", action, decision="FALSIFIED", next_action="Revise benchmark",
                 finding="Matched block effect was negative")
    assert loop.watchdog(stale_seconds=1)["active"] is True
    with pytest.raises(ValueError, match="reflection and reaction"):
        loop.begin("next_direction", "Next question", "EXP-0100")
    wrong_packet = tmp_path / "wrong_packet.json"
    make_packet(action, action, wrong_packet)
    with pytest.raises(ValueError, match="registered design"):
        loop.reflect(wrong_packet, review, reaction)
    saved = loop.reflect(packet, review, reaction)
    assert saved["current"]["reflection"]["issue_count"] == 1
    assert loop.reflect(packet, review, reaction) == saved
    assert loop.watchdog(stale_seconds=1)["active"] is False
    assert loop.begin("next_direction", "Next question", "EXP-0100")["iteration"] == 2


def test_action_bundle_exposes_completion_outputs_and_checks_ledger_identity(tmp_path: Path) -> None:
    design, action, _, _, _ = _records(tmp_path)
    manifest = _write(tmp_path / "manifest.json", {"status": "COMPLETED", "models": 3})
    verification = _write(tmp_path / "verification.json", {"reproduced": True})
    decision = tmp_path / "decision.md"
    decision.write_text("Decision retained after independent reproduction.\n")
    packet = tmp_path / "bundle.json"
    make_packet(design, action, packet, [("manifest", manifest), ("verification", verification),
                                         ("decision", decision)])
    value = validate_packet(packet)
    assert value["schema_version"] == 2
    assert set(value) == {"schema_version", "design", "action"}
    assert [o["kind"] for o in value["action"]["outputs"]] == ["manifest", "verification", "decision"]
    review = _write(tmp_path / "bundle_review.json", {
        "schema_version": 1, "packet_sha256": file_sha256(packet),
        "reviewer": {"model": "independent-test", "effort": "max", "context": "fresh_only_packet"},
        "issues": [{"id": "V1", "source": "action", "severity": "low",
                    "evidence": '"reproduced": true', "problem": "Reproduction scope may be narrow",
                    "why_it_matters": "One seed may not cover all model variability",
                    "falsifiable_check": "Inspect the declared reproduction scope"}],
        "overall_assessment": "Check scope",
    })
    reaction = _write(tmp_path / "bundle_reaction.json", {
        "schema_version": 1, "packet_sha256": file_sha256(packet),
        "review_sha256": file_sha256(review),
        "responses": [{"id": "V1", "disposition": "reject", "reason": "Scope was explicitly bounded",
                       "action": "Keep the scope statement", "evidence_paths": []}],
        "decision_impact": "UNCHANGED", "next_action": "Retain the registered decision",
    })
    assert validate_reaction(packet, review, reaction)
    loop = ResearchLoop(tmp_path / "live_bundle")
    loop.init(design)
    loop.begin("bundle_check", "Can this task be reviewed?", "EXP-0101")
    for phase in PHASES[1:-1]:
        selected = {"REGISTERED": design, "ANALYZED": action,
                    "REVIEWED": verification}.get(phase, action)
        loop.advance(phase, selected)
    loop.advance("DECIDED", decision, decision="INCONCLUSIVE", next_action="Retain bounded result",
                 finding="One bounded result")
    assert loop.reflect(packet, review, reaction)["current"]["reflection"]["issue_count"] == 1
