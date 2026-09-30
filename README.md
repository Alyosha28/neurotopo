# NeuroTopo — architecture-candidate qualification toolkit and evidence base

This repository contains the code, protocols, results and audit records of the NeuroTopo project's
architecture-candidate line (ARCH-0001) together with the qualification tooling built while running it.

**Status: negative and closed, honestly.** The candidate (`SC-REL-v3/v4`: a recurrent core that performs two
hypothetical next-command rollouts and applies a bounded state correction) showed **no replicable advantage**
over a compute-matched strong control on the tasks tested, and the route was closed in the tested scope. The
project's stage gates contain no positive architecture claim. What this repository *does* contain is a
reproducible record of that outcome plus a set of checks that repeatedly caught the project's own false
positives.

## Headline results (all reproducible from the included artifacts)

| Result | Verdict |
|---|---|
| Candidate learning qualification, 15 seeds, fixed budget | `PARTIALLY_SUPPORTED` — 14/15 passed (one `HOLD_COLLAPSE` failure retained) |
| Old one-bit command-response task family | Not discriminative — strong control at 100% on 4/8 pre-registered conditions, and an explicit memoryless rule solves it exactly |
| `DELAY-2` memory-task carrier qualification | `INCONCLUSIVE` — the pre-registered compute-matched control misses the learnability gate (5-seed mean 0.6455 vs 0.65) |
| Exploratory candidate-vs-control probe | `SUPPORTED / EXPLORATORY` (+11.26pp), _not_ A2-creditable: the control used 1.82x the candidate's compute |
| Independent, compute-matched replication (fresh seeds) | `FALSIFIED / NO_REPLICATED_ADVANTAGE` — candidate 0.6637 vs control 0.7261; the no-correction arm ties the candidate |

Two positive signals were produced and both were retracted by pre-registered checks; the full chain
(EXP-0030 -> EXP-0031, EXP-0032 v1 -> v2, EXP-0033 -> EXP-0034) is preserved under `research/evidence/`
and `research/decisions/`.

## Repository layout

| Path | Contents |
|---|---|
| `src/neurotopo/` | Package: task generators, models, qualification tooling, loop/runner utilities |
| `tests/`, `scripts/` | 368-test acceptance suite, verification/replay scripts, experiment drivers |
| `configs/`, `experiments/` | Frozen run configs and pre-registered protocols (EXP-0001 .. EXP-0034) |
| `artifacts/EXP-0028..0034`, `artifacts/validation` | Full records for the six most recent experiments plus the acceptance suite |
| `research/` | Programme: hypotheses, prior-art audits, design docs, decisions (0001-0109), evidence reports, stage gates, packet-only reflection records |
| `paper/` | Contribution dossier and venue assessment for the write-up |
| `docs/` | Workflow, reproduction and protocol documentation |

Large raw artifacts from earlier experiments (hundreds of MB of `.npz` state/trajectory dumps for
EXP-0015..0027 and DYN-0001) are **not** included; the scripts and configs that regenerate them are.
See `PROVENANCE.md`.

## Running the checks

```sh
uv sync --frozen --extra dev
uv run --frozen --extra dev python -m pytest -q
uv run --frozen --extra dev python scripts/verify.py
```

The verification suite writes `artifacts/validation/suite.json`; the qualification runners refuse to start
unless that suite passes and its recorded source hashes match the working tree.

## What this repository does not claim

- no positive architecture effect, no Transformer/Mamba comparison, no scaling, language or biological claim;
- no claim that the qualification checklist items are novel phenomena (see `research/prior_art/` for the
  audits and the explicit overlap list);
- the scoped structural result (the strong control's function class contains the candidate's on the tested
  family) is a statement about that model family and those tasks only.

## License and citation

Code: MIT (`LICENSE`). Documents, protocols, results and audit records: CC BY 4.0 (`LICENSE-DATA`).
Please cite the repository and the specific decision/evidence files you rely on; no paper exists yet.
