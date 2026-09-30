from neurotopo.pfstyle_models import build_model
from neurotopo.pfstyle_resources import resource_account


def test_registered_resource_counts_and_conservative_primary_control():
    expected = {
        "K4_score": (13200, 3984),
        "S68_sparse": (14964, 4152),
        "S68_dense": (14964, 14964),
        "S64_score": (13200, 3984),
        "K4_unscored": (14760, 4356),
    }
    rows = {}
    for name, counts in expected.items():
        model = build_model(name, 9900001, 8700000 + 9900001)
        row = resource_account(model, 16, 128)
        assert (row["allocated_parameters"], row["effective_parameters"]) == counts
        assert row["total_dynamic_state_floats"] == 68
        assert row["pre_readout_cache_floats_per_example"] == 4
        rows[name] = row
    candidate = rows["K4_score"]
    control = rows["S68_sparse"]
    assert 1.0 <= control["allocated_parameters"] / candidate["allocated_parameters"] <= 1.15
    assert 1.0 <= control["effective_parameters"] / candidate["effective_parameters"] <= 1.05
    assert 1.0 <= (control["dense_forward_linear_macs_per_example"] /
                   candidate["dense_forward_linear_macs_per_example"]) <= 1.20
