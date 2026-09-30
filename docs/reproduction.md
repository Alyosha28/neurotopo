# Reproduction

Reference: Apple Silicon macOS, Python 3.13, PyTorch 2.8.0, NumPy 2.3.3, deterministic CPU float32, one thread. Full dependency graph and wheel hashes: `uv.lock`. Exact hardware, build flags and package versions are saved in each manifest. Determinism is checked on this platform, not promised across hardware/releases.

```sh
uv sync --frozen --extra dev
uv run --frozen python scripts/verify.py
uv run --frozen python -m neurotopo.run --config configs/exp0001_v2.json
uv run --frozen python -m neurotopo.analyze --run artifacts/EXP-0001/v2
```

The formal runner requires a clean committed tree and source-hashed passing validation. Commit intentional validation updates before a new formal run. Existing run directories are never overwritten. To reproduce elsewhere use `--output /path/to/new/run`. `--seed 0` reproduces a registered subset and is marked `REPRODUCTION_SUBSET`; it cannot issue a full-study decision. `--allow-dirty` marks engineering-only work and cannot support a scientific decision.

For the EXP-0009 `source_belief` workflow, run from a clean checkout of the frozen commit with its matching `artifacts/validation/suite.json` and `exp0009_preflight.json`. `python -m neurotopo.loop_worker --root <original repo>/artifacts/auto_research/live --config configs/exp0009_v1.json --output <original repo>/artifacts/EXP-0009/v1` keeps the source checkout clean while the real loop ledger and immutable run output live in the project workspace. The `--output` override must be absolute. A single-seed replay uses `python -m neurotopo.source_belief_study --config configs/exp0009_v1.json --seed 9401 --output <fresh directory>`; it is not a full scientific rerun.

## This workspace's external volume
Initial `.venv` installation on this mounted volume failed because AppleDouble sidecars were interpreted as wheel data (`Unknown wheel data type: ._data`). Use an environment and cache on the local filesystem; this does not change dependencies or source:

```sh
export UV_CACHE_DIR=/private/tmp/neurotopo-uv-cache
export UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-venv
uv sync --frozen --extra dev
uv run --frozen python scripts/verify.py
```

The temporary environment is disposable and can be recreated with the lock. `._*` sidecars are ignored by Git and source fingerprints. No global Python packages or global skills are changed.

## Independent check
Use a clean archive/checkout of the execution commit in a fresh path, install from its lock and run seed 0 with a fresh output directory. Compare every reported predictive metric, logit hash, training loss, mask/data hash and resource count to the original seed. Exclude wall-clock timings and serialized checkpoint archive byte identity; compare loaded tensor contents if testing checkpoints. A one-seed reproduction proves only that subset, not the whole ten-seed distribution.
