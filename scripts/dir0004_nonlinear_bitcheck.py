"""DIR-0004: does the 'one stored bit buys one step' law survive nonlinear chaos?

The unstable coordinate is stored with b bits (quantised), the map is iterated in float64
as the reference. T*(b) is the first step where |quantised - exact| exceeds eps. Tent map
(lambda = ln 2) predicts T* ~ b + log2(1/eps); logistic r=4 (lambda = ln 2) likewise.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

MAPS = {"tent": lambda x: np.where(x < 0.5, 2 * x, 2 * (1 - x)),
        "logistic": lambda x: 4 * x * (1 - x)}

def horizon(fn, bits, eps, x0, steps):
    q = 2.0 ** (-bits)
    x_exact, x_q = x0, round(x0 / q) * q
    for t in range(1, steps + 1):
        x_exact = float(fn(np.array([x_exact]))[0])
        x_q = round(float(fn(np.array([x_q]))[0]) / q) * q
        if abs(x_q - x_exact) > eps:
            return t
    return float("nan")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--x0", type=float, default=0.3141592653589793)
    ap.add_argument("--output", type=Path, required=True); a = ap.parse_args()
    out = {"schema": "dir0004_nonlinear_bitcheck_v1", "eps": a.eps, "x0": a.x0, "maps": {}}
    for name, fn in MAPS.items():
        rows = [{"bits": b, "T_star": horizon(fn, b, a.eps, a.x0, 400)} for b in (8, 12, 16, 20, 24, 30)]
        ts = [r["T_star"] for r in rows]
        slope = float(np.polyfit([r["bits"] for r in rows], ts, 1)[0])
        out["maps"][name] = {"rows": rows, "steps_per_bit": slope,
                             "predicted": 1.0 / np.log2(np.e) / (np.log(2) / np.log(2)),  # 1/log2(e)/1 = ln2/ln2
                             "note": "tent and logistic r=4 both have Lyapunov exponent ln 2"}
    payload = json.dumps(out, indent=2, sort_keys=True); a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"output": str(a.output), "sha256": hashlib.sha256((payload+"\n").encode()).hexdigest()[:16],
                      "eps": a.eps, "rows": {k: [(r["bits"], r["T_star"]) for r in v["rows"]] for k, v in out["maps"].items()},
                      "steps_per_bit": {k: round(v["steps_per_bit"], 3) for k, v in out["maps"].items()}}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
