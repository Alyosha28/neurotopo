import torch
from pathlib import Path
from runpy import run_path


_script = run_path(str(Path(__file__).resolve().parents[1] / "scripts/calibrate_distractor_transfer.py"))
_oracle_metrics = _script["_oracle_metrics"]
_paired_batches = _script["_paired_batches"]


def test_calibration_delays_share_evidence_targets_and_oracle_is_read_only():
    batches = _paired_batches(48, 393001)
    assert list(batches) == [1, 4, 8, 16]
    reference = batches[1]
    before = reference.final_joint.clone()
    scores = [_oracle_metrics(batch) for batch in batches.values()]
    assert torch.equal(before, reference.final_joint)
    assert all(scores[0] == item for item in scores[1:])
    assert [batch.x.shape[1] for batch in batches.values()] == [18, 42, 74, 138]
