"""Protocol-required reference for EXP-0028: a simple explicit Gram re-encoder.

The frozen protocol (experiments/EXP-0028/v1.md, task information check) requires measuring whether a
simple explicit solution explains a condition. This script implements the registered reference without
touching the frozen run source: state = canonical(b) with column 1 = e1 and column 2 = (2b-1)e1; at each
step decode the bit from sign(<col1,col2>) of the received (rotated/leaked/noised) state, apply the actual
command, re-encode canonically, and take sign(<col1,col2>) as the logit. No learning, no history.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neurotopo.arch0001_core import generate


def canonical(b: np.ndarray) -> np.ndarray:
    n = len(b)
    h = np.zeros((n, 3, 2), np.float32)
    h[:, 0, 0] = 1.
    h[:, 0, 1] = np.where(b == 1, 1., -1.)
    return h


def reference(condition: dict, cfg: dict, mode: str = "primary") -> dict:
    ns = cfg["streams"]["eval_namespace"]
    n = cfg["streams"]["eval_examples"]
    length = condition["length"]
    amp = 3.141592653589793 if mode == "primary" else .4
    data = generate(n, length, [ns, length, 1, 0], "MIX", amp, condition["leak"], condition["sigma"])
    targets = data["targets"]
    b = data["b0"].astype(np.int64)
    h = canonical(b)
    step_ok = [float(np.mean(b == targets[:, 0]))]
    flip, hold = [], []
    for t in range(length):
        d = np.einsum("nij,njk->nik", data["common"][:, t + 1], h)
        second = np.einsum("nij,njk->nik", data["leak"][:, t], d[:, :, 1:2])
        d = np.stack((d[:, :, 0], second[:, :, 0]), axis=-1) + data["noise"][:, t]
        inner = (d[:, :, 0] * d[:, :, 1]).sum(axis=1)
        b = (inner > 0).astype(np.int64) ^ data["commands"][:, t]
        h = canonical(b)
        ok = b == targets[:, t + 1]
        step_ok.append(float(ok.mean()))
        flip.extend(ok[data["commands"][:, t] == 1].tolist())
        hold.extend(ok[data["commands"][:, t] == 0].tolist())
    return dict(condition=condition, mode=mode,
                final_accuracy=step_ok[-1], accuracy_all=float(np.mean(step_ok[1:])),
                flip_accuracy=(float(np.mean(flip)) if flip else None),
                hold_accuracy=(float(np.mean(hold)) if hold else None),
                steps_below_1=sum(1 for x in step_ok[1:] if x < 1.))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/exp0028_v1.json").read_text())
    out = []
    for condition in cfg["menu"]:
        entry = dict(condition=condition)
        for mode in ("primary", "secondary"):
            r = reference(condition, cfg, mode)
            entry[mode] = {k: r[k] for k in ("final_accuracy", "accuracy_all", "flip_accuracy", "hold_accuracy",
                                             "steps_below_1")}
        out.append(entry)
    summary = dict(reference="simple explicit Gram re-encoder (no learning, no history)",
                   conditions=out,
                   note=("Protocol-required reference measured after the main run because the automated phases "
                         "omitted it; it does not enter the EXP-0028 decision rule."))
    path = root / "research/evidence/EXP-0028/explicit_reference.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for e in out:
        c = e["condition"]
        print(f"{c['id']:18s} primary all={e['primary']['accuracy_all']:.4f} flip={e['primary']['flip_accuracy']:.4f} "
              f"hold={e['primary']['hold_accuracy']:.4f} steps_below_1={e['primary']['steps_below_1']:3d} | "
              f"secondary all={e['secondary']['accuracy_all']:.4f}")


if __name__ == "__main__":
    main()
