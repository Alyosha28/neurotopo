"""Dry-run structural entry checks; fixture content is not an experiment proposal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neurotopo.entry_policy import EntryRejected, POLICY, validate_entry_contract
from neurotopo.loop import PHASES, ResearchLoop


FIXTURES = Path(__file__).parent / "fixtures" / "entry_policy"


def _finished_loop(tmp_path: Path) -> ResearchLoop:
    from neurotopo.reflection import file_sha256, make_packet

    tmp_path.mkdir(parents=True, exist_ok=True)
    loop = ResearchLoop(tmp_path / "live")
    evidence = tmp_path / "evidence.md"
    evidence.write_text("frozen test evidence", encoding="utf-8")
    loop.init(evidence)
    loop.begin("prior_direction", "Prior registered question", "EXP-8001")
    for phase in PHASES[1:-1]:
        loop.advance(phase, evidence)
    loop.advance("DECIDED", evidence, decision="INCONCLUSIVE", next_action="Audit next entry",
                 finding="A bounded earlier result")
    packet = tmp_path / "packet.json"
    make_packet(evidence, evidence, packet)
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema_version": 1, "packet_sha256": file_sha256(packet),
                                  "reviewer": {"model": "test-reviewer", "effort": "test",
                                               "context": "fresh_only_packet"},
                                  "issues": [], "overall_assessment": "Bounded test review"}), encoding="utf-8")
    reaction = tmp_path / "reaction.json"
    reaction.write_text(json.dumps({"schema_version": 1, "packet_sha256": file_sha256(packet),
                                    "review_sha256": file_sha256(review), "responses": [],
                                    "decision_impact": "UNCHANGED", "next_action": "Check next entry"}), encoding="utf-8")
    loop.reflect(packet, review, reaction)
    return loop


def _fixture(name: str) -> tuple[Path, dict]:
    path = FIXTURES / f"{name}.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_policy_activation_requires_completed_reflection_and_is_idempotent(tmp_path: Path):
    loop = ResearchLoop(tmp_path / "live")
    evidence = tmp_path / "spec.md"
    evidence.write_text("spec", encoding="utf-8")
    loop.init(evidence)
    with pytest.raises(ValueError, match="DECIDED, reflected"):
        loop.enable_entry_policy(POLICY)
    complete = _finished_loop(tmp_path / "other")
    activated = complete.enable_entry_policy(POLICY)
    assert activated["entry_policy"] == POLICY
    events = len((complete.root / "iteration_log.jsonl").read_text().splitlines())
    assert complete.enable_entry_policy(POLICY) == activated
    assert len((complete.root / "iteration_log.jsonl").read_text().splitlines()) == events


def test_missing_factor_rejected_without_mutating_live_ledger(tmp_path: Path):
    loop = _finished_loop(tmp_path)
    loop.enable_entry_policy(POLICY)
    path, value = _fixture("project_missing_factor")
    before = (loop.root / "iteration_log.jsonl").read_bytes()
    with pytest.raises(ValueError, match="ENTRY_CONTRACT_REQUIRED"):
        loop.begin(value["direction"], value["question"], value["experiment_id"])
    with pytest.raises(EntryRejected, match="MISSING_FACTOR"):
        loop.begin(value["direction"], value["question"], value["experiment_id"],
                   entry_contract=path)
    assert (loop.root / "iteration_log.jsonl").read_bytes() == before
    assert loop.status()["iteration"] == 1


def test_independent_benchmark_enters_question_with_bound_contract(tmp_path: Path):
    loop = _finished_loop(tmp_path)
    loop.enable_entry_policy(POLICY)
    fixture, value = _fixture("independent_benchmark")
    path = tmp_path / "entry.json"
    path.write_bytes(fixture.read_bytes())
    result = loop.begin(value["direction"], value["question"], value["experiment_id"],
                        entry_contract=path)
    assert result["phase"] == "QUESTION" and result["iteration"] == 2
    assert result["current"]["entry_track"] == "independent_benchmark"
    assert result["current"]["entry_contract"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    snapshot = Path(result["current"]["entry_contract"]["snapshot_path"])
    assert snapshot.read_bytes() == path.read_bytes()
    path.write_text("modified after question opened", encoding="utf-8")
    assert snapshot.read_bytes() == fixture.read_bytes()
    assert result["current"]["evidence"] == {}


def test_reference_factor_is_field_only_until_project_eligibility_is_decided(tmp_path: Path):
    project = _finished_loop(tmp_path / "project")
    project.enable_entry_policy(POLICY)
    path, value = _fixture("project_reference_pending")
    with pytest.raises(EntryRejected, match="PROJECT_ELIGIBILITY_UNRESOLVED"):
        project.begin(value["direction"], value["question"], value["experiment_id"],
                      entry_contract=path)
    assert project.status()["iteration"] == 1
    field = _finished_loop(tmp_path / "field")
    field.enable_entry_policy(POLICY)
    path, value = _fixture("field_reference")
    result = field.begin(value["direction"], value["question"], value["experiment_id"],
                         entry_contract=path)
    assert result["current"]["entry_track"] == "field_reference"
    assert not value["gate1_claim"]


def test_structural_gate_rejects_unbounded_search_and_question_mismatch(tmp_path: Path):
    path, value = _fixture("independent_benchmark")
    with pytest.raises(EntryRejected, match="QUESTION_MISMATCH"):
        validate_entry_contract(path, experiment=value["experiment_id"],
                                direction=value["direction"], question="Different question")
    value["task_search"]["retain_all_attempts"] = False
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(EntryRejected, match="INVALID_SEARCH_BUDGET"):
        validate_entry_contract(altered, experiment=value["experiment_id"],
                                direction=value["direction"], question=value["question"])


def test_independent_benchmark_may_name_its_own_factor_but_cannot_claim_gate1(tmp_path: Path):
    _, value = _fixture("independent_benchmark")
    value["factor"] = {
        "kind": "independent", "name": "Independent nuisance process",
        "manipulation": "Hold observation variance fixed or variable.",
        "ablation": "Remove the variance schedule.",
        "prediction": "The scheduled nuisance changes a predeclared calibration metric.",
        "actual_primary_control": "The same model under fixed observation variance.",
        "state_matching": "Use the same hidden state size.",
        "parameter_matching": "Use identical parameters and initialization.",
        "executed_compute_matching": "Use the same token count and recurrent operations.",
        "prior_art_boundary": "Compare against existing evidence-integration tasks."
    }
    path = tmp_path / "independent_with_factor.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert validate_entry_contract(path, experiment=value["experiment_id"],
                                   direction=value["direction"], question=value["question"])["track"] == "independent_benchmark"
    value["factor"].pop("executed_compute_matching")
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(EntryRejected, match="SCHEMA_MISMATCH"):
        validate_entry_contract(path, experiment=value["experiment_id"],
                                direction=value["direction"], question=value["question"])


def test_existing_symlink_cannot_masquerade_as_content_addressed_snapshot(tmp_path: Path):
    loop = _finished_loop(tmp_path)
    loop.enable_entry_policy(POLICY)
    fixture, value = _fixture("independent_benchmark")
    snapshot = loop.root / "entry_contracts" / f"{hashlib.sha256(fixture.read_bytes()).hexdigest()}.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(fixture)
    events = (loop.root / "iteration_log.jsonl").read_bytes()
    with pytest.raises(ValueError, match="unaliased regular file"):
        loop.begin(value["direction"], value["question"], value["experiment_id"],
                   entry_contract=fixture)
    assert (loop.root / "iteration_log.jsonl").read_bytes() == events
    assert not list(snapshot.parent.glob(".entry-contract.*"))


def test_complete_orphan_snapshot_after_interruption_can_be_reused(tmp_path: Path):
    loop = _finished_loop(tmp_path)
    loop.enable_entry_policy(POLICY)
    fixture, value = _fixture("independent_benchmark")
    snapshot = loop.root / "entry_contracts" / f"{hashlib.sha256(fixture.read_bytes()).hexdigest()}.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(fixture.read_bytes())  # atomic rename succeeded before the prior process stopped
    result = loop.begin(value["direction"], value["question"], value["experiment_id"],
                        entry_contract=fixture)
    assert result["phase"] == "QUESTION"
    assert snapshot.read_bytes() == fixture.read_bytes()

def test_project_gate1_admits_a_declared_architecture_organization_factor(tmp_path: Path) -> None:
    """User-authorized rule change (decision 0080): a non-H3 declared organisation factor may enter
    the project track; a reference K factor still may not, and gate1_claim stays tied to the track."""
    from neurotopo.entry_policy import EntryRejected, validate_entry_contract

    factor = {
        "kind": "architecture_organization",
        "name": "expanding/contracting state organisation",
        "manipulation": "state stores only the expanding-subspace coordinates; contracting components are re-derived",
        "ablation": "symmetric storage of all coordinates under the same state budget",
        "prediction": "the trackable horizon crosses over at a pre-computed T* and diverges afterwards",
        "actual_primary_control": "state- and compute-matched dense recurrence with symmetric storage",
        "state_matching": "equal state dimension and equal executed inner steps, disclosed as measured ratios",
        "parameter_matching": "equal allocated and effective parameter counts, disclosed",
        "executed_compute_matching": "matched-or-operator-profiled executed compute",
        "prior_art_boundary": "reservoir computing and classical filtering already exploit known dynamics; no novelty is claimed",
    }
    contract = {
        "schema_version": 1, "track": "project_gate1", "experiment_id": "EXP-9001",
        "direction": "chaos_tracking", "question": "Does declared expanding/contracting organisation extend the trackable horizon?",
        "purpose": "Bounded project Gate 1 test of a declared organisation factor under matched state budget.",
        "gate1_claim": True,
        "resource_cap": {"max_updates": 10, "max_wall_seconds": 60.0},
        "task_search": {"max_configurations": 1, "retain_all_attempts": True, "heldout_rule": "frozen before any run"},
        "branches": {"success": "s", "failure": "f", "borderline": "b"},
        "factor": factor,
    }
    import json
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    result = validate_entry_contract(path, experiment="EXP-9001", direction="chaos_tracking",
                                     question=contract["question"])
    assert result["track"] == "project_gate1" and result["scientific_approval"] is False

    contract["factor"] = {**factor, "kind": "reference_k"}
    path.write_text(json.dumps(contract))
    try:
        validate_entry_contract(path, experiment="EXP-9001", direction="chaos_tracking",
                                question=contract["question"])
    except EntryRejected as error:
        assert error.code == "PROJECT_ELIGIBILITY_UNRESOLVED"
    else:  # pragma: no cover
        raise AssertionError("reference_k must stay unresolved on the project track")


def test_legacy_migration_preserves_events_and_decision_and_admits_next_question(tmp_path, monkeypatch):
    import neurotopo.entry_policy as entry_policy
    loop = _finished_loop(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(entry_policy, 'POLICY', 'gate1_candidate_first_v1')
        loop.enable_entry_policy(entry_policy.POLICY)
    ledger = loop.root / 'iteration_log.jsonl'
    before = ledger.read_bytes()
    previous = loop.status()
    authority = tmp_path / 'authority.md'
    authority.write_text('Explicitly authorized v2 entry-rule change')
    migrated = loop.migrate_entry_policy('gate1_candidate_first_v1', POLICY, authority)
    assert ledger.read_bytes().startswith(before)
    assert len(ledger.read_text().splitlines()) == len(before.splitlines()) + 1
    assert migrated['current'] == previous['current']
    assert migrated['iteration'] == previous['iteration']
    receipt = json.loads(ledger.read_text().splitlines()[-1])
    assert receipt['kind'] == 'entry_policy_migration'
    assert receipt['evidence']['sha256'] == hashlib.sha256(authority.read_bytes()).hexdigest()
    after = ledger.read_bytes()
    assert loop.migrate_entry_policy('gate1_candidate_first_v1', POLICY, authority) == migrated
    assert ledger.read_bytes() == after
    authority.write_text('Changed authority must not silently reattribute the event')
    with pytest.raises(ValueError, match='reattribute'):
        loop.migrate_entry_policy('gate1_candidate_first_v1', POLICY, authority)
    assert ledger.read_bytes() == after
    path, value = _fixture('independent_benchmark')
    assert loop.begin(value['direction'], value['question'], value['experiment_id'],
                      entry_contract=path)['phase'] == 'QUESTION'


def test_migration_fails_closed_for_unsupported_source_and_unfinished_iteration(tmp_path, monkeypatch):
    import neurotopo.entry_policy as entry_policy
    loop = _finished_loop(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(entry_policy, 'POLICY', 'gate1_candidate_first_v1')
        loop.enable_entry_policy(entry_policy.POLICY)
        path, value = _fixture('independent_benchmark')
        loop.begin(value['direction'], value['question'], value['experiment_id'], entry_contract=path)
    authority = tmp_path / 'authority.md'
    authority.write_text('Authority')
    before = (loop.root / 'iteration_log.jsonl').read_bytes()
    with pytest.raises(ValueError, match='DECIDED, reflected'):
        loop.migrate_entry_policy('gate1_candidate_first_v1', POLICY, authority)
    with pytest.raises(ValueError, match='Unsupported'):
        loop.migrate_entry_policy('unknown_v0', POLICY, authority)
    with pytest.raises(ValueError, match='does not exist'):
        loop.migrate_entry_policy('gate1_candidate_first_v1', POLICY, tmp_path/'missing.md')
    assert (loop.root / 'iteration_log.jsonl').read_bytes() == before
