from pathlib import Path
import json
import sys

import pytest

from neurotopo.loop import ResearchLoop
from neurotopo import loop_worker
from neurotopo.loop_worker import remaining_time_seconds, workflow_commands


def test_source_belief_workflow_uses_absolute_external_output():
    output = Path("/private/tmp/example-exp9-output")
    run, analysis, actual = workflow_commands(Path("configs/exp0009_v1.json"),
                                              {"workflow": "source_belief", "output": "artifacts/EXP-0009/v1"}, output)
    assert run[2] == "neurotopo.source_belief_study"
    assert run[-2:] == ["--output", str(output)]
    assert analysis[2] == "neurotopo.source_belief_analysis"
    assert analysis[-1] == str(output)
    assert actual == output


def test_output_override_must_be_absolute_and_unknown_workflow_rejected():
    with pytest.raises(ValueError, match="absolute"):
        workflow_commands(Path("config.json"), {"workflow": "study", "output": "runs/x"}, Path("relative"))
    with pytest.raises(ValueError, match="Unknown"):
        workflow_commands(Path("config.json"), {"workflow": "unknown", "output": "runs/x"})


def test_pfstyle_workflow_and_exact_shared_timeout():
    run, analysis, output = workflow_commands(
        Path("configs/exp0011_v6.json"),
        {"workflow": "pfstyle_field", "output": "artifacts/EXP-0011/v6"},
    )
    assert run[2] == "neurotopo.pfstyle_study"
    assert analysis[2] == "neurotopo.pfstyle_analysis"
    assert output == Path("artifacts/EXP-0011/v6")
    assert remaining_time_seconds(10.0, 3000.0, 3009.5) == 0.5
    with pytest.raises(TimeoutError, match="No registered wall-time"):
        remaining_time_seconds(10.0, 3000.0, 3010.0)


def test_phase_reveal_workflow_commands_are_distinct():
    qualification, qualification_analysis, q_output = workflow_commands(
        Path("configs/exp0013_v3.json"),
        {"workflow": "phase_reveal_qualification", "output": "artifacts/EXP-0013/v3"},
    )
    formal, formal_analysis, f_output = workflow_commands(
        Path("configs/exp0014_v3.json"),
        {"workflow": "phase_reveal_formal", "output": "artifacts/EXP-0014/v3"},
        loop_root=Path("/private/tmp/neurotopo-loop-main/live"),
    )
    assert qualification[2] == "neurotopo.phase_reveal_qualification"
    assert qualification_analysis[2] == "neurotopo.phase_reveal_qualification_analysis"
    assert q_output == Path("artifacts/EXP-0013/v3")
    assert formal[2] == "neurotopo.phase_reveal_formal"
    assert formal[-2:] == ["--loop-root", "/private/tmp/neurotopo-loop-main/live"]
    assert formal_analysis[2] == "neurotopo.phase_reveal_formal_analysis"
    assert f_output == Path("artifacts/EXP-0014/v3")


def test_pfstyle_worker_timeout_reaches_analyzed_without_formal_training(tmp_path, monkeypatch):
    spec = tmp_path / "task.md"
    spec.write_text("isolated registered test")
    evidence = tmp_path / "evidence.md"
    evidence.write_text("isolated fixture")
    loop = ResearchLoop(tmp_path / "live")
    loop.init(spec)
    loop.begin("isolated-pfstyle", "Can failure close the loop?", "EXP-0011")
    for phase in ("PRIOR_ART", "CRITIQUE", "REGISTERED", "VALIDATED"):
        loop.advance(phase, evidence)
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/exp0011_v6.json").read_text())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "run"
    def expire(*args, **kwargs):
        raise TimeoutError("isolated worker timeout")
    monkeypatch.setattr(loop_worker, "monitored", expire)
    monkeypatch.setattr(sys, "argv", ["loop_worker", "--root", str(loop.root), "--config",
                                 str(config_path), "--timeout", "3000", "--output", str(output)])
    loop_worker.main()
    assert loop.status()["phase"] == "ANALYZED"
    analysis = json.loads((output / "analysis.json").read_text())
    assert analysis["decision"] == "INCONCLUSIVE"
    assert analysis["substatus"] == "INCOMPLETE_BUDGET"
    assert not (output / "formal").exists()
    assert json.loads((loop.root / "jobs/ITER-0001/job_end.json").read_text())["status"] == "RECOVERED_INCONCLUSIVE"


def test_phase_reveal_qualification_timeout_closes_without_task_data(tmp_path, monkeypatch):
    spec = tmp_path / "task.md"
    spec.write_text("isolated phase-reveal worker test")
    evidence = tmp_path / "evidence.md"
    evidence.write_text("isolated registered evidence")
    loop = ResearchLoop(tmp_path / "live")
    loop.init(spec)
    loop.begin("isolated-phase-reveal", "Can a qualification timeout close?", "EXP-0013")
    for phase in ("PRIOR_ART", "CRITIQUE", "REGISTERED", "VALIDATED"):
        loop.advance(phase, evidence)
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/exp0013_v3.json").read_text())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "run"

    def expire(*args, **kwargs):
        raise TimeoutError("isolated qualification timeout")

    monkeypatch.setattr(loop_worker, "monitored", expire)
    monkeypatch.setattr(sys, "argv", ["loop_worker", "--root", str(loop.root), "--config",
                                 str(config_path), "--timeout", "2400", "--output", str(output)])
    loop_worker.main()
    assert loop.status()["phase"] == "ANALYZED"
    analysis = json.loads((output / "analysis.json").read_text())
    assert analysis["qualification_status"] == "INCOMPLETE_BUDGET"
    assert analysis["gate1_credit"] is False
    assert not (output / "task_information.npz").exists()


def test_late_analysis_is_preserved_before_inconclusive_fallback(tmp_path, monkeypatch):
    import json
    import sys
    import neurotopo.loop_worker as worker
    output=tmp_path/'output';output.mkdir()
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(workflow='orbit_information',experiment_id='EXP-0015',version='v1',output=str(output),resource=dict(worker_wall_seconds_cap=300))))
    transitions=[]
    class FakeLoop:
        def __init__(self, root):pass
        def status(self):return dict(phase='VALIDATED',current=dict(experiment='EXP-0015',id='ITER-TEST'))
        def advance(self,phase,evidence):transitions.append(phase)
    def fake_monitored(command,loop,log_path,timeout):
        log_path.write_text('bounded fake child')
        if command[2].endswith('_analysis'):
            (output/'analysis.json').write_text('{"original": "late output"}')
        return 0
    ticks=iter([0.,1.,301.,302.,303.,304.])
    monkeypatch.setattr(worker,'ResearchLoop',FakeLoop)
    monkeypatch.setattr(worker,'monitored',fake_monitored)
    monkeypatch.setattr(worker.time,'monotonic',lambda:next(ticks))
    monkeypatch.setattr(sys,'argv',['worker','--config',str(config),'--root',str(tmp_path/'loop'),'--timeout','300'])
    worker.main()
    assert json.loads((output/'analysis_aborted_before_fallback.json').read_text())=={'original':'late output'}
    fallback=json.loads((output/'analysis.json').read_text())
    assert fallback['decision']=='INCONCLUSIVE' and fallback['substatus']=='INCOMPLETE_BUDGET'
    assert transitions==['RUNNING','ANALYZED']
