"""Local, evidence-gated research workflow ledger.

The CLI records work performed by researchers; it does not perform research.
``python -m neurotopo.loop --root artifacts/auto_research/live --help``
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator


VERSION = 2
PHASES = (
    "QUESTION", "PRIOR_ART", "CRITIQUE", "REGISTERED", "VALIDATED",
    "RUNNING", "ANALYZED", "REVIEWED", "DECIDED",
)
DECISIONS = ("SUPPORTED", "PARTIALLY_SUPPORTED", "INCONCLUSIVE", "FALSIFIED")
DEFAULT_ROOT = Path("artifacts/auto_research/live")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Evidence file does not exist: {path}")
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
            size += len(chunk)
    if size == 0:
        raise ValueError(f"Evidence file is empty: {path}")
    return {"path": str(path.resolve()), "sha256": sha.hexdigest(), "size": size}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _append(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class ResearchLoop:
    """Serialized local operations over a durable research ledger."""

    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".lock").open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _read(self) -> tuple[dict[str, Any], dict[str, Any]]:
        events = _lines(self.root / "iteration_log.jsonl")
        if not events:
            raise ValueError("Research loop is not initialized")
        latest = events[-1]
        progress = latest["progress"]
        directions = latest["directions_tried"]
        # The event ledger is authoritative if a process stopped between its
        # append and the atomic materialized-view writes.
        for name, value in (("progress.json", progress), ("directions_tried.json", directions)):
            path = self.root / name
            if not path.exists() or _json(path) != value:
                _atomic_json(path, value)
        existing = {row["event_id"] for row in _lines(self.root / "findings.jsonl")}
        for event in events:
            if event.get("finding") and event["id"] not in existing:
                _append(self.root / "findings.jsonl", {
                    "event_id": event["id"], "iteration": event["progress"]["iteration"],
                    "phase": event["progress"]["phase"], "finding": event["finding"],
                    "evidence": event["evidence"], "time": event["time"],
                })
                existing.add(event["id"])
        work_ids = {row["event_id"] for row in _lines(self.root / "logs/work.jsonl")}
        for event in events:
            if event["id"] not in work_ids:
                _append(self.root / "logs/work.jsonl", {"event_id": event["id"],
                    "kind": event["kind"], "time": event["time"],
                    "iteration": event["progress"]["iteration"],
                    "phase": event["progress"]["phase"]})
        return progress, directions

    def _commit(self, event: dict[str, Any], progress: dict[str, Any], directions: dict[str, Any]) -> None:
        event.update({"progress": progress, "directions_tried": directions})
        _append(self.root / "iteration_log.jsonl", event)
        _atomic_json(self.root / "progress.json", progress)
        _atomic_json(self.root / "directions_tried.json", directions)
        if event.get("finding"):
            _append(self.root / "findings.jsonl", {
                "event_id": event["id"], "iteration": progress["iteration"],
                "phase": progress["phase"], "finding": event["finding"],
                "evidence": event["evidence"], "time": event["time"],
            })
        _append(self.root / "logs/work.jsonl", {"event_id": event["id"],
            "kind": event["kind"], "time": event["time"],
            "iteration": progress["iteration"], "phase": progress["phase"]})

    def init(self, task_spec: Path | str) -> dict[str, Any]:
        evidence = _digest(Path(task_spec))
        with self._lock():
            if (self.root / "iteration_log.jsonl").exists():
                progress, _ = self._read()
                if progress["task_spec"] != evidence:
                    raise ValueError("Already initialized with a different task spec")
                return progress
            for name in ("findings.jsonl", "logs/work.jsonl", "logs/orchestrator.jsonl", "logs/heartbeat.jsonl"):
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            progress = {
                "version": VERSION, "task_spec": evidence, "iteration": 0,
                "phase": None, "current": None, "findings_count": 0,
                "decisions_count": 0, "stale_count": 0,
                "pivot_required": False, "review_required": False,
                "reflection_required": True, "last_seen": _now(),
            }
            self._commit({"id": "event-000001", "kind": "init", "time": progress["last_seen"]}, progress, {})
            return progress

    def enable_entry_policy(self, policy: str) -> dict[str, Any]:
        """Prospectively require structural entry contracts for new questions."""
        from .entry_policy import POLICY

        if policy != POLICY:
            raise ValueError(f"Unknown entry policy: {policy}")
        with self._lock():
            old, directions = self._read()
            if old.get("entry_policy") == policy:
                return old
            if old.get("entry_policy") is not None:
                raise ValueError("An entry policy is already enabled")
            if old["phase"] != "DECIDED" or not old["current"].get("reflection"):
                raise ValueError("Enable entry policy only after a DECIDED, reflected iteration")
            progress = {**old, "entry_policy": policy, "last_seen": _now()}
            event_id = f"event-{len(_lines(self.root / 'iteration_log.jsonl')) + 1:06d}"
            self._commit({"id": event_id, "kind": "entry_policy", "time": progress["last_seen"],
                          "policy": policy}, progress, directions)
            return progress

    def migrate_entry_policy(self, from_policy: str, policy: str,
                             authority: Path | str) -> dict[str, Any]:
        """Append the authorized v1->v2 transition without rewriting old evidence."""
        from .entry_policy import POLICY

        if (from_policy, policy) != ("gate1_candidate_first_v1", "gate1_candidate_first_v2") or policy != POLICY:
            raise ValueError("Unsupported entry policy migration")
        evidence = _digest(Path(authority))
        with self._lock():
            old, directions = self._read()
            events = _lines(self.root / "iteration_log.jsonl")
            if old.get("entry_policy") == policy:
                matching = [event for event in events
                            if event.get("kind") == "entry_policy_migration"
                            and event.get("from_policy") == from_policy
                            and event.get("policy") == policy
                            and event.get("evidence") == evidence]
                if not matching:
                    raise ValueError("No identical migration receipt; refusing to reattribute authority")
                return old
            if old.get("entry_policy") != from_policy:
                raise ValueError("Migration source policy does not match active policy")
            if old["phase"] != "DECIDED" or not old["current"].get("reflection"):
                raise ValueError("Migrate entry policy only after a DECIDED, reflected iteration")
            progress = {**old, "entry_policy": policy, "last_seen": _now()}
            self._commit({"id": f"event-{len(events)+1:06d}",
                          "kind": "entry_policy_migration", "time": progress["last_seen"],
                          "from_policy": from_policy, "policy": policy, "evidence": evidence},
                         progress, directions)
            return progress

    def begin(self, direction: str, question: str, experiment: str,
              replication_of: str | None = None,
              entry_contract: Path | str | None = None) -> dict[str, Any]:
        direction, question = direction.strip(), question.strip()
        if not direction or not question or not re.fullmatch(r"EXP-\d{4,}", experiment):
            raise ValueError("Direction and question are required; experiment must match EXP-XXXX")
        with self._lock():
            old, directions = self._read()
            if old["phase"] not in (None, "DECIDED"):
                raise ValueError("Current iteration must reach DECIDED before beginning another")
            if (old["phase"] == "DECIDED" and old.get("reflection_required", False)
                    and not old["current"].get("reflection")):
                raise ValueError("Complete independent reflection and reaction before beginning another iteration")
            entry_snapshot = None
            entry_track = None
            entry_raw: bytes | None = None
            if old.get("entry_policy") is not None and entry_contract is None:
                raise ValueError("ENTRY_CONTRACT_REQUIRED: this loop requires an entry contract before begin")
            if entry_contract is not None:
                from .entry_policy import POLICY, validate_entry_contract

                if old.get("entry_policy") not in (None, POLICY):
                    raise ValueError("Unsupported active entry policy")
                path = Path(entry_contract)
                validated = validate_entry_contract(path, experiment=experiment,
                                                    direction=direction, question=question)
                entry_raw = path.read_bytes()
                if hashlib.sha256(entry_raw).hexdigest() != validated["sha256"]:
                    raise ValueError("Entry contract changed during validation")
                snapshot_path = self.root / "entry_contracts" / f"{validated['sha256']}.json"
                entry_snapshot = {"path": str(path.resolve()), "sha256": validated["sha256"],
                                  "size": len(entry_raw),
                                  "snapshot_path": str(snapshot_path.parent.resolve() / snapshot_path.name)}
                entry_track = validated["track"]
            prior = directions.get(direction, [])
            if prior and replication_of not in prior:
                raise ValueError("Repeated direction requires --replication-of with an existing iteration ID")
            if not prior and replication_of is not None:
                raise ValueError("Replication target must be an earlier iteration of this direction")
            iteration = old["iteration"] + 1
            ident = f"ITER-{iteration:04d}"
            if any(e.get("experiment") == experiment for e in _lines(self.root / "iteration_log.jsonl")):
                raise ValueError("Experiment ID was already used")
            if entry_raw is not None:
                snapshot_file = Path(entry_snapshot["snapshot_path"])
                snapshot_file.parent.mkdir(parents=True, exist_ok=True)
                fd, staging_name = tempfile.mkstemp(prefix=".entry-contract.", dir=snapshot_file.parent)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(entry_raw)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if snapshot_file.is_symlink() or snapshot_file.exists():
                        if (snapshot_file.is_symlink() or not snapshot_file.is_file()
                                or snapshot_file.stat().st_nlink != 1):
                            raise ValueError("Entry snapshot target must be an unaliased regular file")
                    else:
                        # Under the loop lock, rename atomically publishes one complete file.
                        # A crash after rename leaves a valid nlink=1 snapshot for retry.
                        os.replace(staging_name, snapshot_file)
                    if snapshot_file.is_symlink() or not snapshot_file.is_file() or snapshot_file.read_bytes() != entry_raw:
                        raise ValueError("Content-addressed entry snapshot differs from validated bytes")
                finally:
                    Path(staging_name).unlink(missing_ok=True)
            current = {"id": ident, "direction": direction, "question": question,
                       "experiment": experiment, "replication_of": replication_of,
                       "evidence": {}, "findings": {}}
            if entry_snapshot is not None:
                current["entry_contract"] = entry_snapshot
                current["entry_track"] = entry_track
            progress = {**old, "iteration": iteration, "phase": "QUESTION", "current": current,
                        "last_seen": _now()}
            updated = {**directions, direction: [*prior, ident]}
            event_id = f"event-{len(_lines(self.root / 'iteration_log.jsonl')) + 1:06d}"
            self._commit({"id": event_id, "kind": "begin", "time": progress["last_seen"],
                          "experiment": experiment, "entry_contract": entry_snapshot}, progress, updated)
            return progress

    def advance(self, phase: str, evidence_path: Path | str, *, decision: str | None = None,
                next_action: str | None = None, finding: str | None = None,
                no_new_finding: bool = False) -> dict[str, Any]:
        phase = phase.upper()
        if phase not in PHASES or phase == "QUESTION":
            raise ValueError("Invalid advance phase")
        evidence = _digest(Path(evidence_path))
        finding = finding.strip() if finding else None
        next_action = next_action.strip() if next_action else None
        if phase == "DECIDED":
            if decision not in DECISIONS or not next_action or (bool(finding) == no_new_finding):
                raise ValueError("DECIDED requires --decision, --next-action and exactly one of --finding or --no-new-finding")
        elif decision or next_action or no_new_finding:
            raise ValueError("Decision, next action and no-new-finding apply only to DECIDED")
        with self._lock():
            old, directions = self._read()
            current_phase = old["phase"]
            if current_phase is None:
                raise ValueError("Begin an iteration first")
            current = old["current"]
            if phase == current_phase:
                prior = current["evidence"].get(phase)
                if prior == evidence and current["findings"].get(phase) == finding and (phase != "DECIDED" or
                    (current.get("decision"), current.get("next_action"), current.get("finding"),
                     current.get("no_new_finding", False))
                    == (decision, next_action, finding, no_new_finding)):
                    return old
                raise ValueError("Phase is already recorded with different content")
            if PHASES.index(phase) != PHASES.index(current_phase) + 1:
                raise ValueError(f"Cannot advance from {current_phase} to {phase}")
            updated_current = {**current, "evidence": {**current["evidence"], phase: evidence},
                               "findings": {**current["findings"], phase: finding}}
            if phase == "DECIDED":
                updated_current.update(decision=decision, next_action=next_action,
                                       finding=finding, no_new_finding=no_new_finding)
            stale_count = (0 if finding else old["stale_count"] + 1 if no_new_finding
                           else old["stale_count"])
            progress = {**old, "phase": phase, "current": updated_current, "last_seen": _now(),
                        "findings_count": old["findings_count"] + bool(finding),
                        "decisions_count": old["decisions_count"] + (phase == "DECIDED"),
                        "stale_count": stale_count, "pivot_required": stale_count >= 2,
                        "review_required": stale_count >= 4}
            event_id = f"event-{len(_lines(self.root / 'iteration_log.jsonl')) + 1:06d}"
            self._commit({"id": event_id, "kind": "advance", "phase": phase,
                          "time": progress["last_seen"], "evidence": evidence,
                          "finding": finding}, progress, directions)
            return progress

    def reflect(self, packet_path: Path | str, review_path: Path | str,
                reaction_path: Path | str) -> dict[str, Any]:
        """Archive one validated, independent-context review and main-agent reaction."""
        from .reflection import validate_packet, validate_reaction, validate_review

        packet_path, review_path, reaction_path = map(Path, (packet_path, review_path, reaction_path))
        with self._lock():
            old, directions = self._read()
            if old["phase"] != "DECIDED":
                raise ValueError("Post-task reflection requires a completed DECIDED iteration")
            packet = validate_packet(packet_path)
            current = old["current"]
            expected = current["evidence"]
            if (packet["design"]["sha256"] != expected["REGISTERED"]["sha256"] or
                    packet["action"]["sha256"] != expected["ANALYZED"]["sha256"]):
                raise ValueError("Packet must contain exactly this iteration's registered design and analyzed action output")
            for output in packet["action"].get("outputs", []):
                phase = {"verification": "REVIEWED", "decision": "DECIDED"}.get(output["kind"])
                if phase and output["sha256"] != expected[phase]["sha256"]:
                    raise ValueError(f"Bundled {output['kind']} must match this iteration's {phase} evidence")
            review = validate_review(packet_path, review_path)
            reaction = validate_reaction(packet_path, review_path, reaction_path)
            records = {name: _digest(path) for name, path in
                       (("packet", packet_path), ("review", review_path), ("reaction", reaction_path))}
            prior = current.get("reflection")
            if prior:
                if prior["records"] == records:
                    return old
                raise ValueError("Reflection is already recorded with different content")
            updated = {"records": records, "issue_count": len(review["issues"]),
                       "decision_impact": reaction["decision_impact"]}
            progress = {**old, "version": VERSION, "reflection_required": True,
                        "current": {**current, "reflection": updated}, "last_seen": _now()}
            event_id = f"event-{len(_lines(self.root / 'iteration_log.jsonl')) + 1:06d}"
            self._commit({"id": event_id, "kind": "reflection", "time": progress["last_seen"],
                          "evidence": records, "finding": None}, progress, directions)
            return progress

    def status(self) -> dict[str, Any]:
        with self._lock():
            progress, _ = self._read()
            return progress

    def heartbeat(self, note: str = "") -> dict[str, Any]:
        with self._lock():
            progress, _ = self._read()
            time = _now()
            _append(self.root / "logs/heartbeat.jsonl", {"time": time, "iteration": progress["iteration"],
                                                          "phase": progress["phase"], "note": note})
            # Heartbeat is operational presence, not scientific progress.
            return {"time": time, "iteration": progress["iteration"], "phase": progress["phase"]}

    def watchdog(self, stale_seconds: int = 3600) -> dict[str, Any]:
        if stale_seconds <= 0:
            raise ValueError("Stale threshold must be positive")
        with self._lock():
            progress, _ = self._read()
            beats = _lines(self.root / "logs/heartbeat.jsonl")
            last = max(beats[-1]["time"], progress["last_seen"]) if beats else progress["last_seen"]
            age = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds())
            active = progress["phase"] not in (None, "DECIDED") or (
                progress["phase"] == "DECIDED" and progress.get("reflection_required", False)
                and not progress["current"].get("reflection"))
            result = {"time": _now(), "iteration": progress["iteration"], "phase": progress["phase"],
                      "last_seen": last, "age_seconds": int(age), "active": active,
                      "stale": active and age > stale_seconds,
                      "stale_seconds": stale_seconds}
            _append(self.root / "logs/orchestrator.jsonl", {"kind": "watchdog", **result})
            return result

    def snapshot(self, output: Path | str) -> dict[str, Any]:
        with self._lock():
            if Path(output).exists():
                raise ValueError(f"Snapshot output already exists: {output}")
            progress, directions = self._read()
            data = {"version": VERSION, "exported_at": _now(), "progress": progress,
                    "directions_tried": directions,
                    "iteration_log": _lines(self.root / "iteration_log.jsonl"),
                    "findings": _lines(self.root / "findings.jsonl"),
                    "logs": {name: _lines(self.root / f"logs/{name}.jsonl")
                             for name in ("work", "orchestrator", "heartbeat")}}
            _atomic_json(Path(output), data)
            return {"output": str(Path(output).resolve()), "iteration": progress["iteration"],
                    "events": len(data["iteration_log"]), "findings": len(data["findings"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    subs = parser.add_subparsers(dest="command", required=True)
    init = subs.add_parser("init")
    init.add_argument("--task-spec", type=Path, required=True)
    begin = subs.add_parser("begin")
    begin.add_argument("--direction", required=True)
    begin.add_argument("--question", required=True)
    begin.add_argument("--experiment", required=True)
    begin.add_argument("--replication-of")
    begin.add_argument("--entry-contract", type=Path)
    entry_policy = subs.add_parser("enable-entry-policy")
    entry_policy.add_argument("--policy", required=True)
    migration = subs.add_parser("migrate-entry-policy")
    migration.add_argument("--from-policy", required=True)
    migration.add_argument("--policy", required=True)
    migration.add_argument("--authority", type=Path, required=True)
    advance = subs.add_parser("advance")
    advance.add_argument("--phase", required=True, choices=PHASES[1:])
    advance.add_argument("--evidence", type=Path, required=True)
    advance.add_argument("--decision", choices=DECISIONS)
    advance.add_argument("--next-action")
    advance.add_argument("--finding")
    advance.add_argument("--no-new-finding", action="store_true")
    subs.add_parser("status")
    heartbeat = subs.add_parser("heartbeat")
    heartbeat.add_argument("--note", default="")
    watchdog = subs.add_parser("watchdog")
    watchdog.add_argument("--stale-seconds", type=int, default=3600)
    snapshot = subs.add_parser("snapshot")
    snapshot.add_argument("--output", type=Path, required=True)
    reflect = subs.add_parser("reflect")
    reflect.add_argument("--packet", type=Path, required=True)
    reflect.add_argument("--review", type=Path, required=True)
    reflect.add_argument("--reaction", type=Path, required=True)
    for command in subs.choices.values():
        command.add_argument("--root", type=Path, default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    loop = ResearchLoop(args.root)
    try:
        if args.command == "init":
            result = loop.init(args.task_spec)
        elif args.command == "begin":
            result = loop.begin(args.direction, args.question, args.experiment,
                                args.replication_of, args.entry_contract)
        elif args.command == "enable-entry-policy":
            result = loop.enable_entry_policy(args.policy)
        elif args.command == "migrate-entry-policy":
            result = loop.migrate_entry_policy(args.from_policy, args.policy, args.authority)
        elif args.command == "advance":
            result = loop.advance(args.phase, args.evidence, decision=args.decision,
                                  next_action=args.next_action, finding=args.finding,
                                  no_new_finding=args.no_new_finding)
        elif args.command == "status":
            result = loop.status()
        elif args.command == "heartbeat":
            result = loop.heartbeat(args.note)
        elif args.command == "watchdog":
            result = loop.watchdog(args.stale_seconds)
        elif args.command == "reflect":
            result = loop.reflect(args.packet, args.review, args.reaction)
        else:
            result = loop.snapshot(args.output)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
