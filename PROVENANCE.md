# Provenance of this public snapshot

2026-09-29. This repository is a curated snapshot of a working research workspace, not a git-history
preservation of it. What was done to build it:

**Included.** `src/`, `tests/`, `scripts/`, `configs/`, `docs/`, `experiments/`, `research/`, `paper/`,
`.codex/skills/`, `pyproject.toml`, `uv.lock`, `README.md`, `AGENTS.md`, plus the complete artifact trees of
EXP-0028..0034 and `artifacts/validation/`.

**Excluded on purpose.**
- Raw large artifacts from EXP-0015..0027 and DYN-0001 (multi-hundred-MB `.npz` trajectory/state dumps; the
  workspace copy alone is ~6.7 GB). Their scripts and configs are included and regenerate them.
- Local-only material: `.venv/`, `.pytest_cache/`, macOS AppleDouble side files (`._*`), `.dsh-recovery/`,
  a raw assistant-session transcript, an obsolete/duplicate git metadata directory, and an unrelated motion
  artifact directory.
- Nothing under `artifacts/auto_research/live/` (local loop ledger).

**Rewritten content.** All absolute local paths were rewritten: the workspace path prefix was made relative,
and the home-directory prefix was replaced by `~/`. 78 text files were affected. Recorded SHA-256 hashes in
the affected records still refer to the pre-rewrite bytes of the files they hash; re-verification of those
hash chains requires the pre-rewrite copies.

**Restored files.** Three working-tree files in `research/iterations/ITER-0021/` and
`research/reflections/ITER-0021/` were corrupted in the source workspace (binary garbage / misplaced
content). For this snapshot they were restored from the last committed version of the same paths.

**Not included / not verified here.** The original workspace's git metadata lives outside the workspace;
the session that produced the most recent experiments could not create commits (sandbox restriction). This
snapshot therefore has no continuous history: it carries a single initial commit and relies on the recorded
hashes, protocols and verification scripts for auditability.
