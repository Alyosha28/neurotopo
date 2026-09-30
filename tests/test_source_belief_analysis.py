import pytest

from neurotopo.source_belief_analysis import classify


CONFIG = {"minimum_useful_kl_gain": 0.05, "max_iid_kl_cost": 0.01,
          "max_h_accuracy_cost": 0.01}


@pytest.mark.parametrize("mean,interval,iid_cost,h_cost,matching,adequacy,headroom,expected", [
    (0.08, [0.06, 0.11], 0.00, 0.00, True, True, True, "SUPPORTED"),
    (0.08, [0.01, 0.11], 0.00, 0.00, True, True, True, "PARTIALLY_SUPPORTED"),
    (0.08, [0.06, 0.11], 0.02, 0.00, True, True, True, "PARTIALLY_SUPPORTED"),
    (0.08, [0.06, 0.11], 0.00, -0.02, True, True, True, "PARTIALLY_SUPPORTED"),
    (0.03, [0.01, 0.04], 0.00, 0.00, True, True, True, "FALSIFIED"),
    (0.06, [0.00, 0.10], 0.00, 0.00, True, True, True, "INCONCLUSIVE"),
    (0.07, [0.05, 0.10], 0.00, 0.00, True, True, True, "PARTIALLY_SUPPORTED"),
    (0.05, [0.01, 0.05], 0.00, 0.00, True, True, True, "PARTIALLY_SUPPORTED"),
    (0.08, [0.06, 0.11], 0.00, 0.00, False, True, True, "INCONCLUSIVE"),
    (0.08, [0.06, 0.11], 0.00, 0.00, True, False, True, "INCONCLUSIVE"),
    (0.08, [0.06, 0.11], 0.00, 0.00, True, True, False, "INCONCLUSIVE"),
])
def test_registered_useful_effect_and_boundary_rules(mean, interval, iid_cost, h_cost,
                                                      matching, adequacy, headroom, expected):
    assert classify({"mean": mean, "ci95": interval}, matching=matching, adequacy=adequacy,
                    headroom=headroom, iid_kl_cost=iid_cost, h_accuracy_cost=h_cost,
                    config=CONFIG) == expected
