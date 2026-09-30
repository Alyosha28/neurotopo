"""Independently recompute and verify EXP-0011's frozen task matrices."""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


SOURCE = Path("configs/exp0011_matrices.json")
GENERATOR = Path("scripts/freeze_exp0011_matrices.py")
REPORT = Path("artifacts/validation/exp0011_matrix_verify.json")
ROOT_SEED = 34017
MODES = 4
DIMENSION = 8
DECAY = 0.97
RUNTIME_ENV_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON number: {value}")


def recompute() -> np.ndarray:
    children = np.random.SeedSequence(ROOT_SEED).spawn(MODES)
    matrices = []
    for child in children:
        raw = np.random.Generator(np.random.PCG64(child)).standard_normal((DIMENSION, DIMENSION))
        q, r = np.linalg.qr(raw)
        diagonal = np.diag(r)
        if not np.isfinite(q).all() or not np.isfinite(r).all() or np.any(diagonal == 0):
            raise ArithmeticError("nonfinite QR or zero diagonal")
        for col in range(DIMENSION):
            if diagonal[col] < 0:
                q[:, col] = -q[:, col]
        determinant = float(np.linalg.det(q))
        if not np.isfinite(determinant):
            raise ArithmeticError("nonfinite determinant")
        if determinant < 0:
            q[:, -1] = -q[:, -1]
        matrices.append(DECAY * q)
    return np.stack(matrices).astype("<f8")


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    started = time.monotonic()
    report = {
        "id": "EXP-0011_MATRIX_VERIFY",
        "status": "RUNNING",
        "source_path": str(SOURCE),
        "python_executable": sys.executable,
        "argv": sys.argv,
        "cwd": str(Path.cwd().resolve()),
    }
    try:
        build = StringIO()
        with redirect_stdout(build):
            np.show_config()
        report.update(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                            text=True).strip(),
                      worktree_status_at_verify=subprocess.check_output(
                          ["git", "status", "--porcelain"], text=True),
                      verifier_sha256=sha256(Path(__file__)),
                      generator_sha256=sha256(GENERATOR),
                      numpy_version=np.__version__,
                      numpy_build_config=build.getvalue(),
                      runtime_thread_env={key: os.environ.get(key) for key in RUNTIME_ENV_KEYS})
        for source in (GENERATOR, Path(__file__)):
            if subprocess.run(["git", "diff", "--quiet", "--", str(source)],
                              check=False).returncode:
                raise RuntimeError(f"uncommitted source edits: {source}")
            if subprocess.run(["git", "diff", "--cached", "--quiet", "--", str(source)],
                              check=False).returncode:
                raise RuntimeError(f"staged source edits: {source}")
        frozen = json.loads(SOURCE.read_text(encoding="utf-8"),
                            parse_constant=reject_nonstandard_constant)
        if (frozen["schema_version"] != 1 or frozen["root_seed"] != ROOT_SEED
                or frozen["modes"] != MODES or frozen["dimension"] != DIMENSION
                or frozen["decay"] != DECAY
                or frozen["qr_sign_rule"] != "make R diagonal positive, then flip final Q column if det(Q)<0"
                or frozen["bit_generator"] != "PCG64 from each SeedSequence.spawn child"):
            raise ValueError("frozen matrix metadata differs from protocol")
        if (frozen["numpy_version_at_freeze"] != np.__version__
                or frozen["numpy_build_config_at_freeze"] != build.getvalue()
                or frozen["runtime_thread_env_at_freeze"] != report["runtime_thread_env"]
                or frozen["python_executable_at_freeze"] != sys.executable
                or frozen["lock_sha256"] != sha256(Path("uv.lock"))):
            raise RuntimeError("ENVIRONMENT_REPLAY_MISMATCH: pinned NumPy/BLAS/lock differs")
        if frozen["generator_source_sha256"] != sha256(GENERATOR):
            raise RuntimeError("GENERATOR_SOURCE_MISMATCH: committed generator bytes differ")
        if not isinstance(frozen["source_commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", frozen["source_commit"]):
            raise ValueError("invalid freeze source commit identifier")
        frozen_source = subprocess.check_output(
            ["git", "show", f"{frozen['source_commit']}:scripts/freeze_exp0011_matrices.py"])
        if hashlib.sha256(frozen_source).hexdigest() != frozen["generator_source_sha256"]:
            raise RuntimeError("GENERATOR_COMMIT_MISMATCH: frozen commit lacks generator bytes")
        if not isinstance(frozen["worktree_status_before_freeze"], str):
            raise ValueError("missing freeze worktree snapshot")
        if len(frozen["checks"]) != MODES:
            raise ValueError("incorrect number of generator checks")
        for check in frozen["checks"]:
            error = float(check["orthogonality_max_abs_error"])
            determinant = float(check["orthogonal_determinant"])
            if (not np.isfinite(error) or not np.isfinite(determinant)
                    or error > 1e-12 or abs(determinant - 1) > 1e-12):
                raise ValueError("invalid frozen generator checks")
        actual = np.asarray(frozen["matrices"], dtype="<f8")
        if actual.shape != (MODES, DIMENSION, DIMENSION) or not np.isfinite(actual).all():
            raise ValueError("frozen matrix shape or finite check failed")
        independent = recompute()
        if not np.isfinite(independent).all():
            raise ArithmeticError("independent matrices are nonfinite")
        actual_bytes = actual.tobytes()
        recomputed_bytes = independent.tobytes()
        actual_digest = hashlib.sha256(actual_bytes).hexdigest()
        if actual_digest != frozen["matrix_le_float64_sha256"]:
            raise ValueError("frozen file's array digest is wrong")
        if actual_bytes != recomputed_bytes:
            raise RuntimeError("RECOMPUTE_ENV_UNDETERMINED: seed/QR bytes differ despite recorded environment")
        errors = []
        determinants = []
        for matrix in actual:
            q = matrix / DECAY
            error = float(np.max(np.abs(q.T @ q - np.eye(DIMENSION))))
            determinant = float(np.linalg.det(q))
            if not np.isfinite(error) or not np.isfinite(determinant):
                raise ArithmeticError("nonfinite orthogonality check")
            errors.append(error)
            determinants.append(determinant)
        if max(errors) > 1e-12 or max(abs(value - 1) for value in determinants) > 1e-12:
            raise ValueError("proper orthogonal matrix check failed")
        report.update(status="PASS", matrix_json_sha256=sha256(SOURCE),
                      matrix_le_float64_sha256=actual_digest,
                      independently_recomputed_sha256=hashlib.sha256(recomputed_bytes).hexdigest(),
                      matrix_shape=list(actual.shape), orthogonality_max_abs_errors=errors,
                      determinants=determinants)
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        report["failure_kind"] = ("ENVIRONMENT_REPLAY_MISMATCH" if "ENVIRONMENT_REPLAY_MISMATCH" in str(error)
                                  else "RECOMPUTE_ENV_UNDETERMINED" if "RECOMPUTE_ENV_UNDETERMINED" in str(error)
                                  else "SOURCE_OR_MATRIX_VALIDATION_FAILURE")
        raise
    finally:
        report["elapsed_seconds_before_report_fsync"] = time.monotonic() - started
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        with REPORT.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "report": str(REPORT),
                      "matrix_le_float64_sha256": report["matrix_le_float64_sha256"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
