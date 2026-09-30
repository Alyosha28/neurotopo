"""DIR-0004 step-0 affordance screen: is the required stored precision unbounded in T?

The pre-design's section 4 gate demands a *verifiable numerical form* of the claim
"no bounded-dimension exact solution exists", not the assertion itself: an explicit
curve of the bits a representation must spend as a function of the horizon T, with a
fitted slope.

Model class (inherited verbatim from the audited `dir0004_precision_check.py`; the
recursion is imported, not re-implemented, so the two artifacts cannot drift apart):
a 2-D linear hyperbolic map z_{t+1} = diag(lam_u, lam_s) z_t, lam_u > 1 > lam_s, stored
at a fixed per-coordinate bit budget b so that storage quantisation adds variance
q^2/12 per coordinate per step, q = 2^-b.

Two conditions:
  hidden   : the expanding coordinate is never observed -> the only thing that can keep
             its error below eps is stored precision, so the minimal sufficient b grows
             with T.
  observed : the expanding coordinate is re-localised every step (negative control). The
             posterior variance is capped at ~sigma^2, so the required b is BOUNDED in T
             and no allocation can fail. Round 1 of DIR-0004 found this by accident
             (crossing T = NaN); here it is the screen's own control.

The control is only meaningful when the observation is informative enough to pin the mode
below eps, i.e. sigma <= eps/10 (the pre-design's section 1 says so explicitly). With
sigma > eps the observed arm is infeasible for EVERY bit budget and the control silently
degenerates into the hidden curve -- the first version of this script shipped exactly that
parameterisation (eps = 1e-6, sigma = 1e-3) and produced two identical conditions. The
screen therefore refuses to report a control verdict unless sigma <= eps/10, and records
the feasibility check in `self_checks`.

Reported: minimal b(T) on a frozen T-grid, the least-squares slope with a bootstrap
interval, the empirical endpoint slope, and the max residual in bits. The nonlinear
replication is not re-run: the audited `dir0004_nonlinear_bitcheck.json` is read and
converted to bits per step with its sha256 recorded, so this artifact never claims a
number its own script did not produce or cite.

No training, no learning, no samples, CPU only. This is a task-side screen.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load_precision_check():
    """Import the audited recursion from the sibling script (single source of truth)."""
    path = HERE / "dir0004_precision_check.py"
    spec = importlib.util.spec_from_file_location("dir0004_precision_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def min_bits_for_horizon(error_variance, lam: float, T: int, eps: float,
                         sigma: float, observed: bool, b_max: int = 160) -> int | None:
    """Smallest b whose stored-precision error stays below eps through step T."""
    for b in range(1, b_max + 1):
        var = error_variance(lam, b, T, sigma, observed=observed)
        if var[T - 1] < eps * eps:
            return b
    return None


def fit_slope(ts: np.ndarray, bs: np.ndarray) -> dict:
    slope, intercept = np.polyfit(ts, bs, 1)
    resid = bs - (slope * ts + intercept)
    total = float(np.sum((bs - bs.mean()) ** 2))
    # A perfectly flat fit (the bounded control) has zero total variance; report R^2 = 1
    # rather than dividing by zero, which is what the first version of this script did.
    r_squared = 1.0 if total == 0.0 else float(1.0 - np.sum(resid ** 2) / total)
    return {
        "slope_bits_per_step": float(slope),
        "intercept_bits": float(intercept),
        "max_abs_residual_bits": float(np.max(np.abs(resid))),
        "r_squared": r_squared,
        "degenerate_flat_fit": bool(total == 0.0),
    }


def bootstrap_slope(ts: np.ndarray, bs: np.ndarray, rng: np.random.Generator,
                    draws: int = 2000) -> dict:
    n = ts.size
    slopes = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        if np.unique(ts[idx]).size < 2:
            slopes[i] = np.nan
            continue
        slopes[i] = np.polyfit(ts[idx], bs[idx], 1)[0]
    lo, hi = np.nanpercentile(slopes, [2.5, 97.5])
    return {"draws": draws, "ci95_low": float(lo), "ci95_high": float(hi)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lam-u", type=float, default=2.0)
    # eps = 1e-3 with sigma = 1e-6 keeps the observed control feasible (sigma <= eps/10).
    ap.add_argument("--sigma", type=float, default=1e-6)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--t-min", type=int, default=4)
    ap.add_argument("--t-max", type=int, default=64)
    ap.add_argument("--t-step", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--linear-check", type=Path,
                    default=Path("research/design/dir0004_precision_check.json"))
    ap.add_argument("--nonlinear-check", type=Path,
                    default=Path("research/design/dir0004_nonlinear_bitcheck.json"))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)

    pc = _load_precision_check()
    rng = np.random.default_rng(a.seed)
    ts = np.arange(a.t_min, a.t_max + 1, a.t_step)

    out: dict = {
        "schema": "dir0004_affordance_screen_v1",
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "declared": {
            "lam_u": a.lam_u, "sigma": a.sigma, "eps": a.eps,
            "predicted_bits_per_step_amplitude": float(np.log2(a.lam_u)),
            "predicted_bits_per_step_variance": float(2 * np.log2(a.lam_u)),
        },
        "conditions": {},
        "nonlinear_replication": {},
        "self_checks": {},
    }

    for name, observed in (("hidden", False), ("observed", True)):
        rows, bs = [], []
        for T in ts:
            b = min_bits_for_horizon(pc.error_variance, a.lam_u, int(T), a.eps, a.sigma, observed)
            rows.append({"T": int(T), "min_bits": b})
            if b is None:
                raise SystemExit(f"bit search exhausted for T={T}")
            bs.append(b)
        bs_arr = np.asarray(bs, dtype=float)
        fit = fit_slope(ts.astype(float), bs_arr)
        out["conditions"][name] = {
            "expanding_observed": observed,
            "rows": rows,
            "fit": fit,
            "bootstrap": bootstrap_slope(ts.astype(float), bs_arr, rng),
            "empirical_endpoint_slope": float((bs_arr[-1] - bs_arr[0]) / (ts[-1] - ts[0])),
            "required_bits_span": float(bs_arr.max() - bs_arr.min()),
        }

    # Nonlinear replication: read the audited artifact rather than re-running it.
    nl_path = Path(a.nonlinear_check)
    nl = json.loads(nl_path.read_text(encoding="utf-8"))
    for mapname, block in nl["maps"].items():
        spb = block["steps_per_bit"]
        out["nonlinear_replication"][mapname] = {
            "steps_per_bit": spb,
            "bits_per_step": float(1.0 / spb),
            "source": str(nl_path),
            "source_sha256": sha256_file(nl_path),
        }

    hidden = out["conditions"]["hidden"]
    observed = out["conditions"]["observed"]
    nl_bits = [v["bits_per_step"] for v in out["nonlinear_replication"].values()]
    observation_informative = bool(a.sigma <= a.eps / 10.0)
    control_bounded = bool(
        observation_informative
        and observed["fit"]["slope_bits_per_step"] <= 0.05
        and observed["required_bits_span"] <= 1.0)
    out["self_checks"] = {
        "observation_informative_enough_for_control": observation_informative,
        "control_verdict": ("BOUNDED" if control_bounded
                            else "OBSERVATION_NOT_INFORMATIVE_ENOUGH"
                            if not observation_informative else "UNEXPECTEDLY_UNBOUNDED"),
        "hidden_slope_matches_log2_lam_u": bool(
            abs(hidden["fit"]["slope_bits_per_step"] - np.log2(a.lam_u)) <= 0.05),
        "hidden_relation_linear_within_one_bit": bool(
            hidden["fit"]["max_abs_residual_bits"] <= 1.0),
        "observed_control_slope_bounded": control_bounded,
        "observed_control_bits_span_bounded": bool(
            observation_informative and observed["required_bits_span"] <= 1.0),
        "observed_control_not_unbounded": control_bounded,
        "nonlinear_bits_per_step_within_15pct_of_one": bool(
            all(abs(v - 1.0) <= 0.15 for v in nl_bits)),
        "necessary_condition_hidden_mode": bool(
            control_bounded and hidden["fit"]["slope_bits_per_step"] >= 0.5),
        "screen_valid": bool(
            observation_informative
            and abs(hidden["fit"]["slope_bits_per_step"] - np.log2(a.lam_u)) <= 0.05
            and hidden["fit"]["max_abs_residual_bits"] <= 1.0
            and control_bounded),
        "linear_check_sha256": sha256_file(Path(a.linear_check))
        if Path(a.linear_check).exists() else None,
    }
    payload = json.dumps(out, indent=2, sort_keys=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(a.output),
        "sha256": hashlib.sha256((payload + "\n").encode()).hexdigest(),
        "hidden_fit": hidden["fit"],
        "observed_fit": observed["fit"],
        "self_checks": out["self_checks"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
