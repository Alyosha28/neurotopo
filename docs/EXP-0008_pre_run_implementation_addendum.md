# EXP-0008 v1 pre-run implementation addendum

The independent design review approved `experiments/EXP-0008/v1.md`; a separate static source review found that the initial ridge-fit MAC proxy counted one of two probe heads. Before any official evaluation, the runner was corrected to report both A and B fits, and the analysis gained absolute-budget and M3/C0 parity gates. A distinct test alters the recorded MAC proxy and requires INCONCLUSIVE.

The reviewer also noted that pair grouping was implied by the audited generator and tests rather than rechecked at runtime. The runner now checks **every** train A/B, validation and test pair for identical pre-evidence prefixes and suffix nuisance events, a confirm selector equal to the prior, a revise selector different from it, confirm label equal to the prior answer, and a revised label different from it. Failed checks stop execution before a completed model record is written. These are implementation/provenance checks under the same registered v1 task, outcome and thresholds.

No EXP-0008 outcome was inspected while making these corrections. Targeted source tests cover frozen final-state/readout identity, paired splitting, fixed heads, all four decision branches, and the corrupted resource proxy. Full source-hashed verification and a clean commit remain prerequisites to the formal run.
