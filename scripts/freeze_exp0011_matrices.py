"""Freeze the four fixed 8D dynamics matrices before EXP-0011 task data."""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


OUTPUT = Path("configs/exp0011_matrices.json")
ROOT_SEED = 34017
MODES = 4
DIMENSION = 8
DECAY = 0.97
RUNTIME_ENV_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON number: {value}")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = Path(__file__)
    if subprocess.run(["git", "diff", "--quiet", "--", str(source)], check=False).returncode:
        raise RuntimeError("matrix generator has uncommitted edits")
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--", str(source)], check=False).returncode:
        raise RuntimeError("matrix generator has staged edits")
    children = np.random.SeedSequence(ROOT_SEED).spawn(MODES)
    matrices = []
    checks = []
    for child in children:
        raw = np.random.Generator(np.random.PCG64(child)).standard_normal((DIMENSION, DIMENSION))
        q, r = np.linalg.qr(raw)
        if not np.isfinite(raw).all() or not np.isfinite(q).all() or not np.isfinite(r).all():
            raise ArithmeticError("nonfinite matrix sample or QR factor")
        diagonal = np.diag(r)
        if np.any(diagonal == 0):
            raise ArithmeticError("QR diagonal is zero")
        q = q @ np.diag(np.where(diagonal < 0, -1.0, 1.0))
        signed_determinant = float(np.linalg.det(q))
        if not np.isfinite(signed_determinant):
            raise ArithmeticError("nonfinite QR determinant")
        if signed_determinant < 0:
            q[:, -1] *= -1
        orthogonality_error = float(np.max(np.abs(q.T @ q - np.eye(DIMENSION))))
        determinant = float(np.linalg.det(q))
        if (not np.isfinite(orthogonality_error) or not np.isfinite(determinant)
                or orthogonality_error > 1e-12 or abs(determinant - 1) > 1e-12):
            raise ArithmeticError("invalid proper orthogonal matrix")
        matrices.append(DECAY * q)
        checks.append({"orthogonality_max_abs_error": orthogonality_error,
                       "orthogonal_determinant": determinant})
    array = np.stack(matrices).astype("<f8")
    if not np.isfinite(array).all():
        raise ArithmeticError("nonfinite frozen matrix array")
    build = StringIO()
    with redirect_stdout(build):
        np.show_config()
    report = {
        "schema_version": 1,
        "task": "EXP-0011 fixed four-mode 8D deterministic dynamics",
        "root_seed": ROOT_SEED,
        "bit_generator": "PCG64 from each SeedSequence.spawn child",
        "qr_sign_rule": "make R diagonal positive, then flip final Q column if det(Q)<0",
        "modes": MODES,
        "dimension": DIMENSION,
        "decay": DECAY,
        "numpy_version_at_freeze": np.__version__,
        "numpy_build_config_at_freeze": build.getvalue(),
        "runtime_thread_env_at_freeze": {key: os.environ.get(key) for key in RUNTIME_ENV_KEYS},
        "python_executable_at_freeze": sys.executable,
        "generator_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "worktree_status_before_freeze": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True),
        "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "matrix_le_float64_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "checks": checks,
        "matrices": array.tolist(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    frozen = json.loads(OUTPUT.read_text(encoding="utf-8"),
                        parse_constant=reject_nonstandard_constant)
    reconstructed = np.asarray(frozen["matrices"], dtype="<f8")
    if (not np.isfinite(reconstructed).all()
            or hashlib.sha256(reconstructed.tobytes()).hexdigest() != report["matrix_le_float64_sha256"]):
        raise RuntimeError("matrix write/read-back digest mismatch")
    print(json.dumps({"output": str(OUTPUT),
                      "matrix_le_float64_sha256": report["matrix_le_float64_sha256"]}))


if __name__ == "__main__":
    main()
