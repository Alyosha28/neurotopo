"""Execute one validated study with liveness monitoring; independent review stays separate."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .loop import ResearchLoop
from .run import write_json


WORKFLOWS = {"refresh_schedule": ("neurotopo.refresh_schedule", "neurotopo.schedule_analysis"),
             "refresh_replication": ("neurotopo.refresh_replication", "neurotopo.refresh_replication_analysis"),
             "state_refresh": ("neurotopo.state_refresh", "neurotopo.state_refresh_analysis"),
             "composition_repair": ("neurotopo.composition_repair", "neurotopo.composition_repair_analysis"),
             "equal_output_stress": ("neurotopo.equal_output_stress", "neurotopo.equal_output_stress_analysis"),
             "protected_replication": ("neurotopo.protected_replication", "neurotopo.protected_replication_analysis"),
             "study": ("neurotopo.study", "neurotopo.study_analysis"),
             "replay": ("neurotopo.replay", "neurotopo.replay_analysis"),
             "probe": ("neurotopo.probe", "neurotopo.probe_analysis"),
             "transfer_probe": ("neurotopo.transfer_probe", "neurotopo.transfer_probe_analysis"),
             "final_probe": ("neurotopo.final_probe", "neurotopo.final_probe_analysis"),
             "source_belief": ("neurotopo.source_belief_study", "neurotopo.source_belief_analysis"),
             "pfstyle_field": ("neurotopo.pfstyle_study", "neurotopo.pfstyle_analysis"),
             "phase_reveal_qualification": ("neurotopo.phase_reveal_qualification",
                                            "neurotopo.phase_reveal_qualification_analysis"),
             "phase_reveal_formal": ("neurotopo.phase_reveal_formal",
                                     "neurotopo.phase_reveal_formal_analysis"),
             "orbit_information": ("neurotopo.orbit_information", "neurotopo.orbit_information_analysis"),
             "orbit_control_pilot": ("neurotopo.orbit_control_pilot", "neurotopo.orbit_control_pilot_analysis"),
             "order_observation": ("neurotopo.order_observation", "neurotopo.order_observation_analysis"),
             "frozen_binary_probe": ("neurotopo.frozen_binary_probe", "neurotopo.frozen_binary_probe_analysis"),
             "co_paired_dynamics": ("neurotopo.co_paired_dynamics", "neurotopo.co_paired_dynamics_analysis"),
             "learned_protected_pilot": ("neurotopo.learned_protected_pilot", "neurotopo.learned_protected_pilot_analysis")}
WALL_CAPPED_WORKFLOWS = {"refresh_schedule", "refresh_replication", "state_refresh", "composition_repair", "equal_output_stress", "protected_replication", "pfstyle_field", "phase_reveal_qualification", "phase_reveal_formal", "orbit_information", "orbit_control_pilot", "order_observation", "frozen_binary_probe", "co_paired_dynamics", "learned_protected_pilot"}


def remaining_time_seconds(started: float, timeout: float, now: float) -> float:
    """Return the exact remaining shared run+analysis allowance or fail closed."""
    remaining = timeout - (now - started)
    if remaining <= 0:
        raise TimeoutError("No registered wall-time remains for analysis")
    return remaining


def workflow_commands(config_path: Path, config: dict, output_override: Path | None = None,
                      loop_root: Path | None = None
                      ) -> tuple[list[str], list[str], Path]:
    workflow = config.get("workflow", "study")
    if workflow not in WORKFLOWS:
        raise ValueError(f"Unknown registered workflow: {workflow}")
    if output_override is not None and not output_override.is_absolute():
        raise ValueError("Output override must be an absolute path")
    run_module, analysis_module = WORKFLOWS[workflow]
    output = output_override or Path(config["output"])
    command = [sys.executable, "-m", run_module, "--config", str(config_path)]
    if output_override is not None:
        command += ["--output", str(output_override)]
    if workflow == "phase_reveal_formal" and loop_root is not None:
        command += ["--loop-root", str(loop_root.resolve())]
    analysis = [sys.executable, "-m", analysis_module, "--run", str(output)]
    return command, analysis, output


def monitored(command: list[str], loop: ResearchLoop, log_path: Path, timeout: float) -> int:
    start = time.monotonic()
    last_beat = start - 10
    with log_path.open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                now = time.monotonic()
                if now - start > timeout:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise TimeoutError(f"Registered job exceeded {timeout} seconds")
                if now - last_beat >= 5:
                    loop.heartbeat(f"pid={child.pid}; module={command[2]}; elapsed={now-start:.1f}s")
                    loop.watchdog(stale_seconds=60)
                    last_beat = now
                time.sleep(0.2)
            return int(child.returncode)
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/auto_research/live"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, help="Absolute output path for a clean source checkout")
    args = parser.parse_args()
    loop = ResearchLoop(args.root)
    status = loop.status()
    c = json.loads(args.config.read_text())
    if status["phase"] != "VALIDATED" or status["current"]["experiment"] != c["experiment_id"]:
        raise ValueError("Current registered experiment must be VALIDATED")
    if (c.get("workflow") in WALL_CAPPED_WORKFLOWS
            and args.timeout != c["resource"]["worker_wall_seconds_cap"]):
        raise ValueError("Workflow requires its registered worker wall-time cap")
    prefix = args.root / f"jobs/{status['current']['id']}"
    prefix.mkdir(parents=True, exist_ok=False)
    start_evidence = prefix / "job_start.json"
    command, analysis_command, output = workflow_commands(args.config, c, args.output,
                                                          args.root.resolve())
    write_json(start_evidence, {"command": command, "timeout_seconds": args.timeout,
                                "experiment": c["experiment_id"], "config": c,
                                "actual_output": str(output)})
    loop.advance("RUNNING", start_evidence)
    started = time.monotonic()
    current_child_stage = "run"
    try:
        code = monitored(command, loop, prefix / "train.log", args.timeout)
        if code:
            raise RuntimeError(f"Training exited {code}; inspect {prefix/'train.log'}")
        remaining = remaining_time_seconds(started, args.timeout, time.monotonic())
        current_child_stage = "analysis"
        code = monitored(analysis_command, loop, prefix / "analysis.log", remaining)
        if code:
            raise RuntimeError(f"Analysis exited {code}; inspect {prefix/'analysis.log'}")
        if time.monotonic() - started > args.timeout:
            raise TimeoutError("Registered run plus analysis exceeded shared wall-time cap")
        analysis = output / "analysis.json"
        loop.advance("ANALYZED", analysis)
        write_json(prefix / "job_end.json", {"status": "COMPLETED", "seconds": time.monotonic()-started,
                                             "review": "PENDING_INDEPENDENT_REVIEW"})
        print(f"ANALYZED {c['experiment_id']}; independent review required before DECIDED", flush=True)
    except BaseException as exc:
        write_json(prefix / "job_end.json", {"status": "FAILED", "error": repr(exc), "seconds": time.monotonic()-started})
        if c.get("workflow") in WALL_CAPPED_WORKFLOWS and isinstance(exc, (TimeoutError, RuntimeError)):
            output.mkdir(parents=True, exist_ok=True)
            failure_path = output / "worker_failure.json"
            if failure_path.exists():
                raise RuntimeError("Worker failure record already exists") from exc
            logs = {}
            for name in ("train.log", "analysis.log"):
                path = prefix / name
                if path.is_file():
                    logs[name] = {"path": str(path),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            write_json(failure_path, {
                "experiment_id": c["experiment_id"], "version": c["version"],
                "substatus": "INCOMPLETE_BUDGET" if isinstance(exc, TimeoutError) else "ENGINEERING_ABORT",
                "stage": current_child_stage, "error": repr(exc),
                "elapsed_seconds": time.monotonic() - started,
                "registered_timeout_seconds": args.timeout, "logs": logs,
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            })
            if (output / "analysis.json").is_file():
                existing = output / "analysis.json"
                preserved = output / "analysis_aborted_before_fallback.json"
                if preserved.exists():
                    raise RuntimeError("Analysis fallback provenance already exists") from exc
                # Preserve the completed-but-late analysis while clearing the
                # canonical path for analyzers that correctly refuse overwrite.
                existing.replace(preserved)
            analysis_module = importlib.import_module(WORKFLOWS[c["workflow"]][1])
            analysis_module.analyze(output)
            loop.advance("ANALYZED", output / "analysis.json")
            write_json(prefix / "job_end.json", {
                "status": "RECOVERED_INCONCLUSIVE", "error": repr(exc),
                "seconds": time.monotonic() - started,
                "analysis": str(output / "analysis.json"),
            })
            print(f"ANALYZED {c['experiment_id']} as INCONCLUSIVE after worker failure", flush=True)
            return
        raise


if __name__ == "__main__":
    main()
