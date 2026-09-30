"""Run the preregistered zero-sample analytic EXP-0012 matrix screen."""

from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import distribution
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


OUTPUT = Path("configs/exp0012_matrices.json")
RECEIPT = Path("configs/exp0012_matrices_receipt.json")
SEED = 35017
DIMENSION = 6
DECAY = 0.98
NOISE_STD = 0.8
ANGLES = (0.08, 0.12, 0.16, 0.20)
RUNTIME_ENV_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _atomic_create_json(path: Path, value: dict[str, object]) -> None:
    """Fsync one complete JSON payload, then publish without replacing a prior file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _proper_q() -> np.ndarray:
    raw = np.random.Generator(np.random.PCG64(SEED)).standard_normal((DIMENSION, DIMENSION))
    if not np.isfinite(raw).all():
        raise ArithmeticError("Nonfinite fixed Gaussian matrix")
    q, upper = np.linalg.qr(raw)
    diagonal = np.diag(upper)
    if not (np.isfinite(q).all() and np.isfinite(upper).all()) or np.any(diagonal == 0):
        raise ArithmeticError("Invalid fixed QR decomposition")
    q = q @ np.diag(np.where(diagonal < 0, -1.0, 1.0))
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    determinant = float(np.linalg.det(q))
    if (not np.isfinite(determinant)
            or np.max(np.abs(q.T @ q - np.eye(DIMENSION))) > 1e-12
            or abs(determinant - 1) > 1e-12):
        raise ArithmeticError("Fixed Q is not proper orthogonal")
    return q


def _rotation(angle: float) -> np.ndarray:
    rotation = np.eye(DIMENSION)
    for first, signed in ((0, angle), (2, -angle), (4, angle)):
        cosine, sine = np.cos(signed), np.sin(signed)
        rotation[first, first] = cosine
        rotation[first, first + 1] = -sine
        rotation[first + 1, first] = sine
        rotation[first + 1, first + 1] = cosine
    determinant = float(np.linalg.det(rotation))
    if (not np.isfinite(determinant)
            or np.max(np.abs(rotation.T @ rotation - np.eye(DIMENSION))) > 1e-12
            or abs(determinant - 1) > 1e-12):
        raise ArithmeticError("Fixed rotation is not proper orthogonal")
    return rotation


def expected_information(target: np.ndarray, distractor: np.ndarray) -> dict[str, float]:
    """Expected cumulative conditional KL from the registered impulse-response formula."""
    target_impulse = np.eye(DIMENSION)[:, 0].copy()
    distractor_impulse = target_impulse.copy()
    running_square = 0.0
    cumulative = 0.0
    values: dict[str, float] = {}
    for step in range(1, 33):
        coefficient = float(target_impulse[0] - distractor_impulse[0])
        running_square += coefficient * coefficient
        cumulative += running_square / (2 * NOISE_STD * NOISE_STD)
        if step in (8, 16, 32):
            values[str(step)] = float(cumulative)
        target_impulse = target @ target_impulse
        distractor_impulse = distractor @ distractor_impulse
    return values


def main() -> None:
    if OUTPUT.exists() or RECEIPT.exists():
        raise FileExistsError("EXP-0012 matrix report or receipt already exists")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("EXP-0012 matrix screen requires a clean committed checkout")
    if np.__version__ != "2.3.3":
        raise RuntimeError("EXP-0012 matrix screen requires NumPy 2.3.3")
    source = Path(__file__)
    q_target = _proper_q()
    target = DECAY * q_target
    candidates: list[dict[str, object]] = []
    selected: np.ndarray | None = None
    selected_angle: float | None = None
    for angle in ANGLES:
        distractor = DECAY * (_rotation(angle) @ q_target)
        values = expected_information(target, distractor)
        if not np.isfinite(list(values.values())).all():
            raise ArithmeticError("Nonfinite analytic information value")
        passed = 0.05 <= values["8"] <= 1.0 and 1.5 <= values["32"] <= 4.5
        candidates.append({"angle": angle, "D8": values["8"], "D16": values["16"],
                           "D32": values["32"], "pass": passed})
        if passed and selected is None:
            selected = np.stack((target, distractor)).astype("<f8")
            selected_angle = angle
    build = StringIO()
    with redirect_stdout(build):
        np.show_config()
    numpy_record = distribution("numpy").read_text("RECORD")
    if numpy_record is None:
        raise RuntimeError("Installed NumPy RECORD provenance unavailable")
    report = {
        "schema_version": 1,
        "scope": "zero-sample analytic matrix screen; no task examples or models",
        "protocol": "research/design/exp0012_matrix_screen_protocol_v3.md",
        "protocol_sha256": _sha(Path("research/design/exp0012_matrix_screen_protocol_v3.md")),
        "seed": SEED,
        "bit_generator": "NumPy PCG64",
        "dimension": DIMENSION,
        "decay": DECAY,
        "noise_std": NOISE_STD,
        "cue_reliability_planned": 0.75,
        "angles_ordered": list(ANGLES),
        "selection_rule": "first angle with 0.05<=D8<=1.0 and 1.5<=D32<=4.5",
        "generator_assumptions": "z0=0, no process noise, independent N(0,1) u and observation epsilon across time/streams, H uniform, cue true=.75 else each wrong=.25/3",
        "qr_sign_rule": "first make intermediate R diagonal positive; then flip final Q column if det(Q)<0",
        "rotation_rule": "column-vector 2D blocks [[cos,-sin],[sin,cos]] at (0,1)=+angle,(2,3)=-angle,(4,5)=+angle, left multiply Q",
        "candidates": candidates,
        "status": "SELECTED" if selected is not None else "NO_MATRIX",
        "selected_angle": selected_angle,
        "matrices": selected.tolist() if selected is not None else None,
        "matrix_le_float64_sha256": sha256(selected.tobytes(order="C")).hexdigest()
        if selected is not None else None,
        "numpy_version_at_screen": np.__version__,
        "numpy_build_config_at_screen": build.getvalue(),
        "numpy_record_sha256": sha256(numpy_record.encode("utf-8")).hexdigest(),
        "numpy_core_binary_sha256": _sha(Path(np._core._multiarray_umath.__file__)),
        "python_version_at_screen": sys.version,
        "python_executable_at_screen": sys.executable,
        "python_executable_sha256": _sha(Path(sys.executable).resolve()),
        "runtime_thread_env_at_screen": {key: os.environ.get(key) for key in RUNTIME_ENV_KEYS},
        "generator_source_sha256": _sha(source),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "lock_sha256": _sha(Path("uv.lock")),
    }
    _atomic_create_json(OUTPUT, report)
    frozen = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if selected is not None:
        roundtrip = np.asarray(frozen["matrices"], dtype="<f8")
        if (not np.isfinite(roundtrip).all() or roundtrip.shape != (2, DIMENSION, DIMENSION)
                or sha256(roundtrip.tobytes(order="C")).hexdigest()
                != frozen["matrix_le_float64_sha256"]):
            raise RuntimeError("EXP-0012 selected matrix digest mismatch")
    report_sha = _sha(OUTPUT)
    receipt = {"id": "EXP-0012_MATRIX_SCREEN_RECEIPT",
               "report_sha256": report_sha,
               "generator_source_sha256": report["generator_source_sha256"],
               "source_commit": report["source_commit"],
               "matrix_le_float64_sha256": report["matrix_le_float64_sha256"]}
    _atomic_create_json(RECEIPT, receipt)
    print(json.dumps({"status": report["status"], "output": str(OUTPUT),
                      "selected_angle": selected_angle,
                      "matrix_le_float64_sha256": report["matrix_le_float64_sha256"],
                      "report_sha256": report_sha, "receipt": str(RECEIPT)}))


if __name__ == "__main__":
    main()
