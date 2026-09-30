"""Conditional normal-theory planning and synthetic decision-boundary cases.

No research observations, candidate pilot, parameter updates, or empirical power.
"""
from __future__ import annotations

import hashlib
import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import mpmath as mp


def classify(lower: list[float], upper: list[float], *, complete: bool = True,
             gates: bool = True) -> str:
    if not complete:
        return "INCONCLUSIVE_EXECUTION"
    if not gates:
        return "INCONCLUSIVE_GATES"
    if len(lower) != 4 or len(upper) != 4 or any(l > u for l, u in zip(lower, upper)):
        raise ValueError("Four ordered bounds required")
    if any((u-l)/2 > .01 for l, u in zip(lower, upper)):
        return "INCONCLUSIVE_PRECISION"
    if all(l > .02 for l in lower):
        return "SUPPORTED_LOCAL_ONLY"
    if any(u <= .02 for u in upper):
        return "FALSIFIED_CONJUNCTION"
    if any(l > .02 for l in lower):
        return "PARTIALLY_SUPPORTED"
    return "INCONCLUSIVE"


def main() -> None:
    mp.mp.dps = 40
    n, m, sd, effect, delta, width = 128, 4, .04, .04, .02, .01
    df = n-1
    alpha = .05/m

    def t_cdf(x: mp.mpf) -> mp.mpf:
        return 1-mp.betainc(mp.mpf(df)/2, mp.mpf(".5"), 0,
                            df/(df+x*x), regularized=True)/2

    lo, hi = mp.mpf(0), mp.mpf(10)
    for _ in range(160):
        middle = (lo+hi)/2
        if t_cdf(middle) < 1-alpha:
            lo = middle
        else:
            hi = middle
    critical = float((lo+hi)/2)
    s_cap = width*sqrt(n)/critical
    mean_failure = NormalDist().cdf((delta+width-effect)*sqrt(n)/sd)
    chi_threshold = df*(s_cap/sd)**2
    precision_failure = float(mp.gammainc(mp.mpf(df)/2, chi_threshold/2, mp.inf)/mp.gamma(mp.mpf(df)/2))
    lower_power = max(0., 1-m*(mean_failure+precision_failure))
    cases = [
        ([.03]*4, [.04]*4, {}, "SUPPORTED_LOCAL_ONLY"),
        ([.01]+[.03]*3, [.02]+[.04]*3, {}, "FALSIFIED_CONJUNCTION"),
        ([.03,.015,.015,.015], [.04,.025,.025,.025], {}, "PARTIALLY_SUPPORTED"),
        ([.015]*4, [.025]*4, {}, "INCONCLUSIVE"),
        ([.021]*4, [.06]*4, {}, "INCONCLUSIVE_PRECISION"),
        ([.03]*4, [.04]*4, {"gates": False}, "INCONCLUSIVE_GATES"),
        ([.03]*4, [.04]*4, {"complete": False}, "INCONCLUSIVE_EXECUTION"),
        ([.02]*4, [.02]*4, {}, "FALSIFIED_CONJUNCTION"),
    ]
    rows=[]
    for low, high, options, expected in cases:
        actual=classify(low, high, **options)
        assert actual == expected, (actual, expected)
        rows.append(dict(lower=low, upper=high, options=options, expected=expected, actual=actual))
    result=dict(
        scope="CONDITIONAL_PLANNING_NOT_EMPIRICAL_POWER", n=n, comparisons=m,
        df=df, family_alpha=.05, critical_t=critical, assumed_sd_upper=sd,
        assumed_all_effects_lower=effect, delta=delta, maximum_halfwidth=width,
        sample_sd_cap=s_cap, mean_failure_upper=mean_failure,
        precision_failure_upper=precision_failure,
        joint_power_lower_bound_under_assumptions=lower_power,
        assumptions=["IID_normal_seed_differences", "each_SD_at_most_0.04", "each_effect_at_least_0.04"],
        assumptions_empirically_verified=False, candidate_samples=0, research_updates=0,
        decision_cases=rows, conditional_check_pass=lower_power >= .8,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    target=Path("research/design/dir0007_power_plan.json")
    if target.exists():
        raise FileExistsError("Preserve planning output; version amendments")
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "decision_cases"},indent=2))
    if lower_power < .8:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
