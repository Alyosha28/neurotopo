"""Durable workflow checks; no experiment is run by these tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurotopo.loop import PHASES, ResearchLoop, main
from neurotopo.reflection import file_sha256, make_packet


def _file(path: Path, text: str = "recorded evidence") -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _started(tmp_path: Path) -> tuple[ResearchLoop, Path]:
    loop = ResearchLoop(tmp_path / "live")
    loop.init(_file(tmp_path / "task.md", "frozen task"))
    loop.begin("baseline_learnability", "Can the baseline learn delay8?", "EXP-0003")
    return loop, _file(tmp_path / "evidence.md")


def _finish(loop: ResearchLoop, evidence: Path) -> None:
    for phase in PHASES[1:-1]:
        loop.advance(phase, evidence)
    loop.advance("DECIDED", evidence, decision="FALSIFIED", next_action="Retain negative result",
                 finding="Delay8 baseline remains unlearned under registered budget")
    _reflect(loop, evidence)


def _reflect(loop: ResearchLoop, evidence: Path) -> None:
    packet = evidence.parent / f"packet-{loop.status()['iteration']}.json"
    review = evidence.parent / f"review-{loop.status()['iteration']}.json"
    reaction = evidence.parent / f"reaction-{loop.status()['iteration']}.json"
    make_packet(evidence, evidence, packet)
    review.write_text(json.dumps({"schema_version": 1, "packet_sha256": file_sha256(packet),
                                  "reviewer": {"model": "test-independent", "effort": "test",
                                               "context": "fresh_only_packet"},
                                  "issues": [], "overall_assessment": "No issue identified in test fixture"}))
    reaction.write_text(json.dumps({"schema_version": 1, "packet_sha256": file_sha256(packet),
                                    "review_sha256": file_sha256(review), "responses": [],
                                    "decision_impact": "UNCHANGED", "next_action": "Retain decision"}))
    loop.reflect(packet, review, reaction)


def test_order_and_evidence_are_enforced(tmp_path: Path) -> None:
    loop, evidence = _started(tmp_path)
    with pytest.raises(ValueError, match="Cannot advance"):
        loop.advance("CRITIQUE", evidence)
    with pytest.raises(ValueError, match="does not exist"):
        loop.advance("PRIOR_ART", tmp_path / "missing.md")
    with pytest.raises(ValueError, match="empty"):
        loop.advance("PRIOR_ART", _file(tmp_path / "empty.md", ""))
    assert loop.status()["phase"] == "QUESTION"
    loop.advance("PRIOR_ART", evidence)
    assert loop.status()["current"]["evidence"]["PRIOR_ART"]["size"] > 0


def test_duplicate_transition_does_not_append_and_different_content_rejected(tmp_path: Path) -> None:
    loop, evidence = _started(tmp_path)
    first = loop.advance("PRIOR_ART", evidence)
    count = len((loop.root / "iteration_log.jsonl").read_text().splitlines())
    assert loop.advance("PRIOR_ART", evidence) == first
    assert len((loop.root / "iteration_log.jsonl").read_text().splitlines()) == count
    evidence.write_text("changed evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="different content"):
        loop.advance("PRIOR_ART", evidence)


def test_repeat_direction_needs_explicit_prior_iteration(tmp_path: Path) -> None:
    loop, evidence = _started(tmp_path)
    _finish(loop, evidence)
    with pytest.raises(ValueError, match="replication-of"):
        loop.begin("baseline_learnability", "Repeat", "EXP-0004")
    with pytest.raises(ValueError, match="replication-of"):
        loop.begin("baseline_learnability", "Repeat", "EXP-0004", "ITER-9999")
    second = loop.begin("baseline_learnability", "Independent repeat", "EXP-0004", "ITER-0001")
    assert second["current"]["replication_of"] == "ITER-0001"
    assert json.loads((loop.root / "directions_tried.json").read_text()) == {
        "baseline_learnability": ["ITER-0001", "ITER-0002"]
    }


def test_watchdog_only_appends_guardian_log(tmp_path: Path) -> None:
    loop, _ = _started(tmp_path)
    progress = (loop.root / "progress.json").read_bytes()
    history = (loop.root / "iteration_log.jsonl").read_bytes()
    findings = (loop.root / "findings.jsonl").read_bytes()
    assert loop.watchdog(stale_seconds=1)["phase"] == "QUESTION"
    assert (loop.root / "progress.json").read_bytes() == progress
    assert (loop.root / "iteration_log.jsonl").read_bytes() == history
    assert (loop.root / "findings.jsonl").read_bytes() == findings
    guardian = (loop.root / "logs/orchestrator.jsonl").read_text().splitlines()
    assert len(guardian) == 1 and json.loads(guardian[0])["kind"] == "watchdog"


def test_completed_iteration_is_not_a_stalled_job(tmp_path: Path, monkeypatch) -> None:
    from datetime import datetime, timedelta, timezone
    import neurotopo.loop as module
    loop, evidence = _started(tmp_path)
    _finish(loop, evidence)
    before = (loop.root / "progress.json").read_bytes()
    future = datetime.now(timezone.utc) + timedelta(days=2)
    class FutureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return future
    monkeypatch.setattr(module, "datetime", FutureClock)
    status = loop.watchdog(stale_seconds=60)
    assert status["age_seconds"] > 60
    assert status["active"] is False and status["stale"] is False
    assert (loop.root / "progress.json").read_bytes() == before


def test_negative_finding_counts_and_reload_repairs_views(tmp_path: Path) -> None:
    loop, evidence = _started(tmp_path)
    _finish(loop, evidence)
    assert loop.status()["findings_count"] == 1
    assert loop.status()["decisions_count"] == 1
    finding = json.loads((loop.root / "findings.jsonl").read_text().splitlines()[0])
    assert finding["finding"].startswith("Delay8")
    assert finding["evidence"]["sha256"]
    (loop.root / "progress.json").unlink()
    (loop.root / "findings.jsonl").write_text("")
    reloaded = ResearchLoop(loop.root)
    assert reloaded.status()["phase"] == "DECIDED"
    assert reloaded.status()["current"]["decision"] == "FALSIFIED"
    assert len((loop.root / "findings.jsonl").read_text().splitlines()) == 1
    result = reloaded.snapshot(tmp_path / "archive.json")
    assert result["findings"] == 1
    assert json.loads((tmp_path / "archive.json").read_text())["iteration_log"][-1]["kind"] == "reflection"


def test_early_finding_and_decision_requirements(tmp_path: Path) -> None:
    loop, evidence = _started(tmp_path)
    loop.advance("PRIOR_ART", evidence, finding="A cited method overlaps this proposal")
    assert loop.status()["findings_count"] == 1
    with pytest.raises(ValueError, match="different content"):
        loop.advance("PRIOR_ART", evidence)
    for phase in PHASES[2:-1]:
        loop.advance(phase, evidence)
    with pytest.raises(ValueError, match="requires"):
        loop.advance("DECIDED", evidence, decision="INCONCLUSIVE", next_action="Inspect controls")
    assert loop.status()["phase"] == "REVIEWED"


def test_no_new_finding_tracks_staleness_without_metric_rule(tmp_path: Path) -> None:
    loop, evidence = _started(tmp_path)
    for iteration in range(1, 5):
        for phase in PHASES[1:-1]:
            loop.advance(phase, evidence)
        result = loop.advance("DECIDED", evidence, decision="INCONCLUSIVE",
                              next_action="Revise the registered question", no_new_finding=True)
        assert result["stale_count"] == iteration
        assert result["pivot_required"] is (iteration >= 2)
        assert result["review_required"] is (iteration >= 4)
        assert result["findings_count"] == 0
        assert result["current"]["next_action"] == "Revise the registered question"
        assert loop.advance("DECIDED", evidence, decision="INCONCLUSIVE",
                            next_action="Revise the registered question", no_new_finding=True) == result
        _reflect(loop, evidence)
        if iteration < 4:
            loop.begin(f"direction_{iteration + 1}", "Another registered question",
                       f"EXP-{iteration + 3:04d}")
    assert (loop.root / "findings.jsonl").read_text() == ""
    loop.begin("direction_5", "Test a disconfirming control", "EXP-0008")
    _finish(loop, evidence)
    status = loop.status()
    assert status["stale_count"] == 0
    assert not status["pivot_required"] and not status["review_required"]
    assert status["findings_count"] == 1


def test_snapshot_never_overwrites_existing_archive(tmp_path: Path) -> None:
    loop, _ = _started(tmp_path)
    target = tmp_path / "archive.json"
    first = loop.snapshot(target)
    original = target.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        loop.snapshot(target)
    assert target.read_bytes() == original
    assert first["events"] == 2


def test_cli_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "live"
    spec = _file(tmp_path / "spec.md")
    assert main(["--root", str(root), "init", "--task-spec", str(spec)]) == 0
    assert main(["status", "--root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["version"] == 2
