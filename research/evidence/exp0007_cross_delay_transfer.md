# A short-delay linear readout transfers less well in the block model

EXP-0006 found that candidate labels remain linearly readable from the frozen pre-evidence state even at delay32, while the models' original task outputs are less reliable. EXP-0007 v2 asks whether a **fixed prior-answer linear mapping fitted at delay8** can be applied to longer-delay states. It fits equal-sized ridge heads to frozen model states, using one standardizer computed from the delay8 A training states for every head/test within that model seed. Independent B heads at source and target delays quantify refitting variability. All inputs use the original full-noise task distribution and separate preregistered train/validation/test streams.

| Delay32, ten trained model seeds | Block M3 | Matched random C0 | Contextual wide GRU |
|---|---:|---:|---:|
| Short-delay H8A head on the long test | 66.99% | 76.36% | 93.26% |
| Long-delay H32A head on the **same** test | 97.77% | 98.22% | 99.06% |
| Transfer loss (long-fitted minus short-fitted) | 30.78 pp | 21.87 pp | 5.80 pp |
| Original pretrained task head, pre answer | 72.23% | 80.41% | 92.20% |

The preregistered difference in transfer loss, `(M3 long−short)−(C0 long−short)`, is **+8.91 pp**. The paired trained-seed percentile-bootstrap 95% CI is **[+1.57,+15.78] pp**. All source/target same-domain validation, behavior-gap, resource and independent head-refit stability guards passed. The automated decision is **PARTIALLY_SUPPORTED**: a positive direction is supported, while the interval does not confirm a minimum +5 pp excess. H8A and H8B source probes both validate near100%; H32A/B target heads validate near98% for the primary models. The model's original head and the H8 probe show similar aggregate long-delay trends, but they were trained on different objectives and cannot be treated as the same mapping.

The result is a property of the fixed ridge probe and the tested distributions. Delay8→32 changes noise events and recurrent updates together; shared scaling removes an avoidable head-specific normalization change, but transfer loss can still arise from mean/scale shift, other state geometry, finite data or regularization. Better performance from a head fitted using *new labelled long-delay data* does not establish why the pretrained multi-output head failed. Candidate memory capacity, semantic trajectory specialization and biological mechanism remain untested. Gate1 and earlier EXP-0004/0006 decisions are unchanged.

Complete seed/head records, fit coefficients, data/source hashes and secondary delay64 results are under `artifacts/EXP-0007/v2/`. Independent scientific review recomputed the paired result and confirmed PARTIALLY_SUPPORTED in `research/iterations/ITER-0005/review.md`. A clean-checkout numerical subset is documented separately in `docs/EXP-0007_reproduction.md`; it covers seed9101 only and does not add an independent trained seed.
