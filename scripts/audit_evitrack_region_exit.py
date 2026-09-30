"""EVT-0001: fixed-parameter, zero-training EviTrack region-exit audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np


SEED = 20260924
PATHS = 100_000
MAX_STEP = 200
TIMES = (0, 5, 10, 20, 30, 80, 140, 170, 200)
BOUNDARY = 2.0
OUTPUT = Path("artifacts/validation/evt0001_region_exit.json")
PROTOCOL = Path("research/design/evt0001_evitrack_region_exit_protocol.md")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    started = time.perf_counter()
    rng = np.random.Generator(np.random.PCG64(SEED))
    position = rng.standard_normal(PATHS)
    first_exit = np.full(PATHS, -1, dtype=np.int16)
    first_exit[np.abs(position) > BOUNDARY] = 0
    for step in range(1, MAX_STEP + 1):
        active = np.flatnonzero(first_exit < 0)
        if not len(active):
            break
        current = position[active]
        updated = current - 0.06 * current * (current * current - 9.0)
        updated += 0.05 * rng.standard_normal(len(active))
        if not np.isfinite(updated).all():
            raise FloatingPointError(f"nonfinite state at step {step}")
        position[active] = updated
        first_exit[active[np.abs(updated) > BOUNDARY]] = step
    elapsed = time.perf_counter() - started
    if len(first_exit) != PATHS:
        raise RuntimeError("path count changed")
    rows = []
    for step in TIMES:
        exited = int(np.count_nonzero((first_exit >= 0) & (first_exit <= step)))
        rows.append({"step": step, "first_exited_count": exited,
                     "first_exited_fraction": exited / PATHS,
                     "not_yet_exited_count": PATHS - exited,
                     "not_yet_exited_fraction": (PATHS - exited) / PATHS})
    remaining_30 = rows[TIMES.index(30)]["not_yet_exited_fraction"]
    if elapsed > 30:
        label = "AUDIT_FAILED"
    elif remaining_30 <= 0.01:
        label = "EARLY_EXIT_PROXY"
    elif remaining_30 >= 0.10:
        label = "REGION_PERSISTENCE_PLAUSIBLE"
    else:
        label = "INCONCLUSIVE_PROXY"
    report = {
        "id": "EVT-0001", "label": label, "seed": SEED, "paths": PATHS,
        "max_step": MAX_STEP, "elapsed_seconds": elapsed,
        "parameters": {"a": 3.0, "V0": 0.06, "delta_t": 1.0,
                       "sigma_z": 0.05, "d": BOUNDARY, "z0": "Normal(0,1)"},
        "rows": rows, "never_exited_by_200": int(np.count_nonzero(first_exit < 0)),
        "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "numpy_version": np.__version__,
        "scope": "First latent exit from even-emission region, not posterior t_DD or paper replication",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"label": label, "remaining_at_30": remaining_30,
                      "elapsed_seconds": elapsed, "output": str(OUTPUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
