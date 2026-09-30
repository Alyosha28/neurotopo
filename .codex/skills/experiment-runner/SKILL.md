---
name: experiment-runner
description: Execute an already registered NeuroTopo experiment with immutable configurations, validation gates, provenance and resource records; use for formal runs and explicit reproduction subsets.
---

# Experiment runner

Read the active experiment, `docs/reproduction.md` and `docs/experiment_protocol.md`. Confirm the experiment is permitted by `research/stage_gates.md` and prior-art inspection is recorded.

- Run required equivalence/leakage/determinism/overfit/accounting checks with `python scripts/verify.py`. Formal execution requires the passing report to match current source hashes and a clean committed tree.
- Use `python -m neurotopo.run --config <registered-config>`. Preserve existing output directories. Use `--output` for new runs; `--seed` is a declared reproduction subset, not a replacement for planned seeds. `--allow-dirty` results remain engineering-only.
- Record commit, source/config/lock hashes, seeds, hardware, packages, dataset/generator hashes, runtime, model state and parameters, active vs dense executed compute, latency and available memory accounting. Label estimates and unavailable measurements.
- Preserve every failure and planned seed. No silent retries with new seeds, metric changes, early stopping, test-set tuning or checkpoint selection. Protocol changes require a new version before running.
- Analyze with the registered procedure; update experiment registry, evidence/negative-results records and stage gate only after review. A successful command or tiny overfit is infrastructure evidence.

For mounted-volume wheel errors, use the documented local temporary environment and the same `uv.lock`; do not change global Python or ignore installation errors.

Reference selection: `docs/skill_reference_audit.md`. This skill is a NeuroTopo-owned workflow over the repository runner, not a copied external runtime.
