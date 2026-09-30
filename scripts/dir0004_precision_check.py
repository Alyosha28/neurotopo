"""DIR-0004 step 1: turn the precision-separation claim into verifiable numbers.

Model class (declared): a 2-D LINEAR hyperbolic map z_{t+1} = diag(lam_u, lam_s) z_t with
lam_u > 1 > lam_s, observed each step as y_t = z_t + eta, eta ~ N(0, sigma^2). Belief is
Gaussian (Kalman recursion); the STATE IS STORED WITH A FIXED BIT BUDGET, and storage
quantisation adds variance q^2/12 per coordinate per step.

Two representations with the SAME total bits:
  symmetric : equal bits per coordinate
  structured: more bits on the expanding coordinate, fewer on the contracting one
Prediction to test: the structured allocation keeps the expanding-coordinate error below
eps for a longer horizon, and the required bits grow linearly in T with slope
2*ln(lam_u)/ln(2) bits per step (i.e. no bounded-dimension exact solution).

No training, no learning; this is the affordance screen of the pre-design, CPU only.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def error_variance(lam: float, bits: int, T: int, sigma: float, observed: bool = True) -> np.ndarray:
    """Posterior variance of one coordinate under quantised storage.

    observed=True : the coordinate is re-localised by an observation every step, so the
                    variance saturates at ~sigma^2 and the bit requirement is BOUNDED in T.
    observed=False: the coordinate is a hidden unstable mode, so its variance grows
                    geometrically and the bits needed to stay below eps grow linearly in T.
    """
    q = 2.0 ** (-bits)
    var = np.empty(T); v = q * q / 12.0   # the mode is stored at the representation's precision
    for t in range(T):
        v = (lam ** 2) * v                                  # predict (expanding)
        if observed:
            v = 1.0 / (1.0 / v + 1.0 / sigma ** 2)          # observation update
        v += q * q / 12.0                                   # storage quantisation
        var[t] = v
    return var

def crossing(var: np.ndarray, eps: float) -> float:
    idx = np.where(var > eps * eps)[0]
    return float(idx[0] + 1) if idx.size else float("nan")

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lam-u", type=float, default=2.0); ap.add_argument("--lam-s", type=float, default=0.5)
    ap.add_argument("--total-bits", type=int, default=16); ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--sigma", type=float, default=1e-3); ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--observe-expanding", action="store_true",
                    help="re-localise the expanding coordinate every step (default: hidden unstable mode)")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    b_sym = a.total_bits // 2
    out = {"schema": "dir0004_precision_check_v1", "config": vars(a) | {"output": str(a.output)},
           "declared": {"lam_u": a.lam_u, "lam_s": a.lam_s, "bits_per_coord_symmetric": b_sym,
                        "total_bits": a.total_bits, "sigma": a.sigma, "eps": a.eps},
           # The two conventions differ by exactly a factor 2 and were conflated in the
           # first version of this artifact, which made `predicted_gain_steps` read 15.5
           # against a measured 31. Amplitude is the convention the ladder uses.
           "theoretical_bits_per_step_amplitude": float(np.log(a.lam_u) / np.log(2)),
           "theoretical_bits_per_step_variance": float(2 * np.log(a.lam_u) / np.log(2)),
           "rows": [], "self_checks": {}}
    for delta in range(0, b_sym):                      # structured: +delta on expanding, -delta on contracting
        bu, bs = b_sym + delta, b_sym - delta
        vu = error_variance(a.lam_u, bu, a.T, a.sigma, observed=a.observe_expanding)
        vs = error_variance(a.lam_s, bs, a.T, a.sigma, observed=True)
        out["rows"].append({"delta": delta, "bits_expanding": bu, "bits_contracting": bs,
                            "crossing_T": crossing(vu, a.eps), "final_var_expanding": float(vu[-1]),
                            "contracting_bounded": bool(vs[-1] < a.eps ** 2)})
    base = error_variance(a.lam_u, b_sym, a.T, a.sigma, observed=a.observe_expanding)
    best = error_variance(a.lam_u, a.total_bits - 1, a.T, a.sigma, observed=a.observe_expanding)
    out["self_checks"] = {
        "symmetric_crossing_T": crossing(base, a.eps),
        "structured_crossing_T": crossing(best, a.eps),
        "horizon_gain_steps": crossing(best, a.eps) - crossing(base, a.eps),
        # Amplitude convention: one extra bit on the hidden mode buys ln2/ln(lam_u) steps.
        "predicted_gain_steps": float((b_sym - 1) * np.log(2) / np.log(a.lam_u)),
        # The bits-needed-versus-T curve required by section 4 of the pre-design is NOT
        # defined on this ladder; it is computed (with its negative control) by
        # scripts/dir0004_affordance_screen.py. An earlier version of this file reported
        # a hand-rolled `fit_bits_vs_T_slope` here (0.0644) that measured something else
        # and is deliberately not carried forward.
    }
    payload = json.dumps(out, indent=2, sort_keys=True); a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"output": str(a.output), "sha256": hashlib.sha256((payload + "\n").encode()).hexdigest(),
                      "theoretical_bits_per_step_amplitude": round(out["theoretical_bits_per_step_amplitude"], 4),
                      "theoretical_bits_per_step_variance": round(out["theoretical_bits_per_step_variance"], 4),
                      "self_checks": {k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in out["self_checks"].items()}}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
