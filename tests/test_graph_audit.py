from pathlib import Path
from runpy import run_path

import pytest

from neurotopo.models import recurrent_mask


graph_measures = run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_existing_graphs.py"))["graph_measures"]


def test_fixed_partition_metrics_detect_multiple_changed_graph_properties():
    block, _ = recurrent_mask(64, 4, "block", 704401)
    random, _ = recurrent_mask(64, 4, "random", 704401)
    a, b = graph_measures(block.numpy()), graph_measures(random.numpy())
    for name in ("nodes", "edges", "self_loops", "in_degree", "out_degree"):
        assert a[name] == b[name]
    assert a["directed_modularity_original_blocks"] == pytest.approx(0.75)
    assert a["cross_original_block_edges"] == 0
    assert a["reciprocal_offdiagonal_edge_fraction"] == 1
    assert a["second_absolute_eigenvalue_row_normalized"] == pytest.approx(1)
    assert not a["strongly_connected"]
    assert b["cross_original_block_edges"] > 0
    assert b["directed_modularity_original_blocks"] < a["directed_modularity_original_blocks"]
    assert b["reciprocal_offdiagonal_edge_fraction"] < 1
    assert b["second_absolute_eigenvalue_row_normalized"] < 1
    assert b["strongly_connected"]
