"""Freeze EXP-0013 matrices and analytic gamma without drawing task samples."""

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


OUTPUT = Path("configs/exp0013_matrices.json")
RECEIPT = Path("configs/exp0013_matrices_receipt.json")
PROTOCOL = Path("research/design/exp0013_matrix_gamma_freeze_v2.md")
SEED = 36017
DIMENSION = 6
DECAY = np.float64(0.98)
NOISE_STD = np.float64(0.8)
TARGET_INFORMATION = np.float64(3.0)
RUNTIME_ENV_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _atomic_create_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        Path(temporary).unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _proper_q(child: np.random.SeedSequence) -> tuple[np.ndarray, str]:
    raw = np.random.Generator(np.random.PCG64(child)).standard_normal((DIMENSION, DIMENSION))
    if not np.isfinite(raw).all():
        raise ArithmeticError("Nonfinite fixed Gaussian matrix")
    q, upper = np.linalg.qr(raw)
    diagonal = np.diag(upper)
    if not (np.isfinite(q).all() and np.isfinite(upper).all()) or np.any(diagonal == 0):
        raise ArithmeticError("Invalid fixed QR decomposition")
    q = q @ np.diag(np.where(diagonal < 0, -1.0, 1.0))
    determinant = float(np.linalg.det(q))
    if not np.isfinite(determinant):
        raise ArithmeticError("Nonfinite QR determinant")
    if determinant < 0:
        q[:, -1] *= -1
    determinant = float(np.linalg.det(q))
    if (not np.isfinite(determinant)
            or np.max(np.abs(q.T @ q - np.eye(DIMENSION))) > 1e-12
            or abs(determinant - 1) > 1e-12):
        raise ArithmeticError("Fixed Q is not proper orthogonal")
    return q, sha256(raw.astype("<f8").tobytes(order="C")).hexdigest()


def _base_information(target: np.ndarray, distractor: np.ndarray) -> tuple[float, dict[str, float]]:
    target_impulse = np.eye(DIMENSION, dtype=np.float64)[:, 0].copy()
    distractor_impulse = target_impulse.copy()
    running_square = np.float64(0.0)
    cumulative = np.float64(0.0)
    values: dict[str, float] = {}
    denominator = np.float64(2.0) * NOISE_STD * NOISE_STD
    for step in range(1, 49):
        coefficient = np.float64(target_impulse[0] - distractor_impulse[0])
        running_square = np.float64(running_square + coefficient * coefficient)
        if step >= 17:
            cumulative = np.float64(cumulative + running_square / denominator)
        if step in (8, 16, 32, 48):
            values[str(step)] = float(cumulative)
        target_impulse = target @ target_impulse
        distractor_impulse = distractor @ distractor_impulse
    return float(cumulative), values


def main() -> None:
    if OUTPUT.exists() or RECEIPT.exists():
        raise FileExistsError("EXP-0013 matrix report or receipt already exists")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("EXP-0013 matrix freeze requires a clean committed checkout")
    if np.__version__ != "2.3.3":
        raise RuntimeError("EXP-0013 matrix freeze requires NumPy 2.3.3")
    children = np.random.SeedSequence(SEED).spawn(2)
    fixed = [_proper_q(child) for child in children]
    q_array = np.stack((fixed[0][0], fixed[1][0])).astype("<f8")
    matrices = (DECAY * q_array).astype("<f8")
    dbase, diagnostic = _base_information(matrices[0], matrices[1])
    gamma = (np.sqrt(TARGET_INFORMATION / np.float64(dbase))
             if np.isfinite(dbase) and dbase > 0 else np.float64("nan"))
    selected = bool(np.isfinite(gamma) and 0.1 <= gamma <= 1.5)
    gamma_array = np.asarray([gamma], dtype="<f8") if np.isfinite(gamma) else None
    failure_reason = (None if selected else
                      "NONFINITE_D_BASE" if not np.isfinite(dbase) else
                      "NONPOSITIVE_D_BASE" if dbase <= 0 else
                      "NONFINITE_GAMMA" if not np.isfinite(gamma) else
                      "GAMMA_OUT_OF_WINDOW")
    build = StringIO()
    with redirect_stdout(build):
        np.show_config()
    numpy_record = distribution("numpy").read_text("RECORD")
    if numpy_record is None:
        raise RuntimeError("Installed NumPy RECORD provenance unavailable")
    report = {
        "schema_version": 1,
        "scope": "zero-sample fixed-matrix and analytic-signal freeze",
        "status": "SELECTED" if selected else "NO_TASK",
        "failure_reason": failure_reason,
        "protocol": str(PROTOCOL),
        "protocol_sha256": _sha(PROTOCOL),
        "root_seed": SEED,
        "child_rule": "np.random.SeedSequence(36017).spawn(2), one PCG64(child).standard_normal((6,6)) per child in order",
        "qr_sign_rule": "make intermediate R diagonal positive, then flip final Q column if det(Q)<0",
        "dimension": DIMENSION,
        "decay": float(DECAY),
        "noise_std": float(NOISE_STD),
        "switch_step": 16,
        "target_expected_kl_at_48": float(TARGET_INFORMATION),
        "gamma_window": [0.1, 1.5],
        "raw_matrix_sha256": [fixed[0][1], fixed[1][1]],
        "q_matrices": q_array.tolist(),
        "matrices": matrices.tolist(),
        "matrix_le_float64_sha256": sha256(matrices.tobytes(order="C")).hexdigest(),
        "D_base_48": dbase if np.isfinite(dbase) else None,
        "D_base_float64_bits_hex": np.asarray([dbase], dtype="<f8").tobytes(order="C").hex(),
        "D_base_nonfinite_repr": str(dbase) if not np.isfinite(dbase) else None,
        "D_base_diagnostic": {key: value if np.isfinite(value) else None
                              for key, value in diagnostic.items()},
        "gamma": float(gamma) if gamma_array is not None else None,
        "gamma_float64_bits_hex": np.asarray([gamma], dtype="<f8").tobytes(order="C").hex(),
        "gamma_nonfinite_repr": str(gamma) if not np.isfinite(gamma) else None,
        "gamma_le_float64_sha256": sha256(gamma_array.tobytes(order="C")).hexdigest()
        if gamma_array is not None else None,
        "generator_source_sha256": _sha(Path(__file__)),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "lock_sha256": _sha(Path("uv.lock")),
        "python_executable_at_freeze": sys.executable,
        "python_executable_sha256": _sha(Path(sys.executable).resolve()),
        "python_version_at_freeze": sys.version,
        "numpy_version_at_freeze": np.__version__,
        "numpy_record_sha256": sha256(numpy_record.encode("utf-8")).hexdigest(),
        "numpy_core_binary_sha256": _sha(Path(np._core._multiarray_umath.__file__)),
        "numpy_build_config_at_freeze": build.getvalue(),
        "runtime_thread_env_at_freeze": {key: os.environ.get(key) for key in RUNTIME_ENV_KEYS},
    }
    _atomic_create_json(OUTPUT, report)
    frozen = json.loads(OUTPUT.read_text(encoding="utf-8"))
    roundtrip = np.asarray(frozen["matrices"], dtype="<f8")
    if (roundtrip.shape != (2, DIMENSION, DIMENSION) or not np.isfinite(roundtrip).all()
            or sha256(roundtrip.tobytes(order="C")).hexdigest()
            != frozen["matrix_le_float64_sha256"]):
        raise RuntimeError("EXP-0013 matrix write/read-back digest mismatch")
    if gamma_array is not None:
        roundtrip_gamma = np.asarray([frozen["gamma"]], dtype="<f8")
        if (not np.isfinite(roundtrip_gamma).all()
                or sha256(roundtrip_gamma.tobytes(order="C")).hexdigest()
                != frozen["gamma_le_float64_sha256"]):
            raise RuntimeError("EXP-0013 gamma write/read-back digest mismatch")
    report_sha = _sha(OUTPUT)
    receipt = {"id": "EXP-0013_MATRIX_GAMMA_RECEIPT",
               "report_sha256": report_sha,
               "generator_source_sha256": report["generator_source_sha256"],
               "source_commit": report["source_commit"],
               "matrix_le_float64_sha256": report["matrix_le_float64_sha256"],
               "gamma_le_float64_sha256": report["gamma_le_float64_sha256"]}
    _atomic_create_json(RECEIPT, receipt)
    print(json.dumps({"status": report["status"], "report": str(OUTPUT),
                      "receipt": str(RECEIPT), "report_sha256": report_sha,
                      "D_base_48": report["D_base_48"], "gamma": report["gamma"]},
                     allow_nan=False))


if __name__ == "__main__":
    main()
