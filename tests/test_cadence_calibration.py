import numpy as np
import torch
from pathlib import Path
from runpy import run_path

from neurotopo.source_belief_task import make_source_belief_batch


_script = run_path(str(Path(__file__).resolve().parents[1] / "scripts/calibrate_cadence.py"))
BATCH_SIZE = _script["BATCH_SIZE"]
UPDATES = _script["UPDATES"]
same_informative_episode = _script["same_informative_episode"]
varied_schedule = _script["varied_schedule"]
oracle_metrics = _script["oracle_metrics"]


def test_varied_cadence_matches_fixed_total_steps_and_informative_examples():
    schedule = varied_schedule()
    assert len(schedule) == UPDATES == 1500
    assert {d: schedule.count(d) for d in (0, 1, 2)} == {0: 500, 1: 500, 2: 500}
    assert schedule == varied_schedule()
    for end in range(300, UPDATES + 1, 300):
        assert sum(schedule[end - 300:end]) == 300
        assert sum((10 + 8 * d) * BATCH_SIZE for d in schedule[end - 300:end]) == 300 * 18 * BATCH_SIZE
    for step in (1, 300, 900, 1500):
        seed = 491000 + 401 * 10000 + step
        fixed = make_source_belief_batch(BATCH_SIZE, np.random.default_rng(seed))
        varied = make_source_belief_batch(BATCH_SIZE, np.random.default_rng(seed),
                                          distractors_per_evidence=schedule[step - 1])
        assert same_informative_episode(fixed, varied)
        assert torch.equal(fixed.final_joint, varied.final_joint)
        before = fixed.final_joint.clone()
        assert oracle_metrics(fixed) == oracle_metrics(varied)
        assert torch.equal(fixed.final_joint, before)
