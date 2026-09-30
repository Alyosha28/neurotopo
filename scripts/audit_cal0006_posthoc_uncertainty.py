"""Post-hoc, fixed-checkpoint uncertainty audit; never changes CAL-0006 v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.nn import functional as F

from neurotopo.dynamic_filter_task import make_dynamic_filter_batch
from neurotopo.models import RecurrentModel
from neurotopo.run import configure


ROOT = Path(".")
REPORT = ROOT / "artifacts/calibration/CAL-0006_v3.json"
REPRO = ROOT / "artifacts/calibration/CAL-0006_v3_reproduction.json"
ARRAYS = ROOT / "artifacts/calibration/CAL-0006_seed603_dense_posthoc.npz"
OUTPUT = ROOT / "research/reflections/CAL-0006/diagnostic.json"
PROTOCOL = ROOT / "research/reflections/CAL-0006/diagnostic_protocol.md"
SOURCE_ROOT = Path("/private/tmp/neurotopo-cal0006-clean-f170e9b")
INTERPRETER = Path("/private/tmp/neurotopo-venv/bin/python")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bootstrap_means(values: np.ndarray, *, seed: int, draws: int = 5000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty(draws, dtype=np.float64)
    size = len(values)
    for start in range(0, draws, 50):
        stop = min(start + 50, draws)
        index = rng.integers(0, size, size=(stop - start, size))
        result[start:stop] = values[index].mean(1)
    return result


def main() -> None:
    if ARRAYS.exists() or OUTPUT.exists():
        raise FileExistsError("Post-hoc output already exists")
    if (Path.cwd().resolve() != SOURCE_ROOT or Path(sys.executable) != INTERPRETER
            or Path(sys.prefix) != INTERPRETER.parent.parent):
        raise RuntimeError("Post-hoc diagnostic requires the frozen checkout and interpreter")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if commit != "f170e9b3acea0ad7611746f5a61c9e1f86e980a4" or dirty:
        raise RuntimeError("Post-hoc diagnostic checkout is not frozen and clean")
    for name in ("neurotopo.dynamic_filter_task", "neurotopo.models", "neurotopo.run"):
        if not Path(sys.modules[name].__file__).resolve().is_relative_to(SOURCE_ROOT):
            raise RuntimeError(f"Post-hoc diagnostic imported a nonfrozen module: {name}")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    reproduction = json.loads(REPRO.read_text(encoding="utf-8"))
    sidecar = json.loads(REPORT.with_name("CAL-0006_v3_publication.json").read_text(encoding="utf-8"))
    if (report["git_commit"] != commit or digest(REPORT.read_bytes()) != sidecar["report_sha256"]
            or digest(REPORT.read_bytes()) != reproduction["report_sha256"]
            or digest(REPORT.with_name("CAL-0006_v3_publication.json").read_bytes())
            != reproduction["publication_sidecar_sha256"]):
        raise ValueError("Frozen report/source provenance mismatch")
    expected_pairs = {(seed, arm) for seed in (601, 602, 603) for arm in ("dense", "random")}
    observed_pairs = [(item["seed"], item["arm"]) for item in reproduction["checkpoint_replay"]]
    if len(observed_pairs) != 6 or set(observed_pairs) != expected_pairs:
        raise ValueError("Independent reproduction does not cover six unique model records")
    row = next(record for record in report["model_results"] if record["seed"] == 603)
    arm = row["arms"]["dense"]
    checkpoint = Path(arm["checkpoint"])
    if digest(checkpoint.read_bytes()) != arm["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint digest mismatch")
    configure(603, 1)
    model = RecurrentModel(4, 64, 4, groups=1, topology="dense")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    batch = make_dynamic_filter_batch(8192, 12, 6_400_001)
    if digest(batch.x.numpy().tobytes()) != report["evaluation"]["ood_input_sha256"]:
        raise ValueError("Frozen OOD evaluation input mismatch")
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(batch.x), 256):
            _, logits = model(batch.x[start:start + 256], batch.pre_index)
            rows.append(F.softmax(logits, dim=1))
    p = torch.cat(rows).numpy().astype(np.float64)
    q = batch.final_joint.numpy().astype(np.float64)
    kl = (q * (np.log(np.clip(q, 1e-12, 1)) - np.log(np.clip(p, 1e-12, 1)))).sum(1)
    changed = batch.pre_joint.numpy().argmax(1) != batch.final_joint.numpy().argmax(1)
    mean = float(kl.mean())
    revision_mean = float(kl[changed].mean())
    if not (np.isfinite(kl).all() and abs(mean - arm["ood12"]["final_joint_kl_to_oracle"]) < 1e-12
            and abs(revision_mean - arm["ood12"]["oracle_revision_subset_final_kl"]) < 1e-12
            and int(changed.sum()) == 3381):
        raise ValueError("Frozen per-example KL does not reproduce the report")
    overall_draws = bootstrap_means(kl, seed=6_600_001)
    revision_draws = bootstrap_means(kl[changed], seed=6_600_002)
    paired_differences = []
    for replay in reproduction["checkpoint_replay"]:
        original = next(rec for rec in report["model_results"] if rec["seed"] == replay["seed"])["arms"][replay["arm"]]
        paired_differences.append({"seed": replay["seed"], "arm": replay["arm"],
                                   "iid_kl_difference": replay["iid8_final_kl_nats"] - original["iid8"]["final_joint_kl_to_oracle"],
                                   "ood_kl_difference": replay["ood12_final_kl_nats"] - original["ood12"]["final_joint_kl_to_oracle"]})
    np.savez_compressed(ARRAYS, kl=kl, natural_revision=changed)
    value = {
        "kind": "POST_HOC_FIXED_CHECKPOINT_DIAGNOSTIC_NOT_FORMAL_RECLASSIFICATION",
        "formal_status_unchanged": report["status"],
        "source_commit": report["git_commit"],
        "protocol_sha256": digest(PROTOCOL.read_bytes()),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "checkpoint_sha256": arm["checkpoint_sha256"],
        "per_example_array_sha256": digest(ARRAYS.read_bytes()),
        "per_example_kl_bytes_sha256": digest(kl.tobytes()),
        "evaluation_examples": len(kl),
        "natural_revision_examples": int(changed.sum()),
        "overall": {"mean_kl": mean, "mean_minus_0_08": mean - 0.08,
                    "bootstrap_draws": len(overall_draws), "bootstrap_seed": 6_600_001,
                    "bootstrap_95_percentile": np.percentile(overall_draws, [2.5, 97.5]).tolist(),
                    "bootstrap_ge_0_08_fraction": float((overall_draws >= 0.08).mean()),
                    "per_example_quantiles": np.percentile(kl, [0, 25, 50, 75, 100]).tolist()},
        "natural_revision_slice": {"mean_kl": revision_mean,
                                   "bootstrap_draws": len(revision_draws), "bootstrap_seed": 6_600_002,
                                   "bootstrap_95_percentile": np.percentile(revision_draws, [2.5, 97.5]).tolist(),
                                   "threshold_applicable": False},
        "six_frozen_checkpoint_replay_differences": paired_differences,
        "optimizer_retrained": False,
        "uncertainty_scope": "Evaluation-example resampling conditional on seed603 dense weights; no training-seed uncertainty",
    }
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(OUTPUT), "overall": value["overall"],
                      "revision": value["natural_revision_slice"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
