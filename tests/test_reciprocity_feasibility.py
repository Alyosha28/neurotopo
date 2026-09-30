from pathlib import Path
from runpy import run_path

import torch

from neurotopo.models import recurrent_mask


witness = run_path(str(Path(__file__).resolve().parents[1] / "scripts/witness_reciprocity_variation.py"))["witness"]


def test_local_witness_preserves_registered_graph_constraints_deterministically():
    mask, _ = recurrent_mask(64, 4, "random", 695000 + 9401)
    before = mask.clone()
    first = witness(mask.numpy().astype(bool), 505000 + 9401, max_attempts=10_000)
    second = witness(mask.numpy().astype(bool), 505000 + 9401, max_attempts=10_000)
    assert first == second
    assert first["found"]
    assert first["degree_diagonal_connectivity_preserved"]
    assert first["original_cross_edges"] == first["changed_cross_edges"]
    assert first["original_reciprocity"] != first["changed_reciprocity"]
    assert torch.equal(mask, before)
