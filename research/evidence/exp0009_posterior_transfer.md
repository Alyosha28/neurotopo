# EXP-0009: matched recurrent topology comparison had insufficient d=4 headroom

2026-09-24. The frozen `EXP-0009/v1` full run completed at source commit `c32b07b071ace8c3c57aa2e5b01d9d238d374ff8`: ten fresh paired training/graph seeds9401–9410, three models per seed, identical varied d=0/1/2 schedule and per-seed data, 192,000 examples and3,456,000 input tokens per arm. All30 result files, checkpoints and predictions were retained. Independent science review (`research/iterations/ITER-0007/review.md`) recomputed all90 model×seed×d prediction metrics and the paired bootstrap; seed9401 was exactly rerun for all three models from a clean checkout (`docs/EXP-0009_reproduction.md`).

| Evaluation d | Block M3 final joint KL | Matched random C0 | Dense H64 context |
|---:|---:|---:|---:|
| 1 | 0.003112 | 0.003307 | 0.004240 |
| **4 (sole primary)** | **0.013819** | **0.012463** | **0.015780** |
| 8 (descriptive) | 0.08127 | 0.06606 | 0.09288 |

At d=4, the registered effect `Δ=KL(C0)−KL(M3)` averaged **−0.001356 nat**, sample SD0.002657, ten-paired-seed bootstrap95%CI **[−0.002990,+0.000120]**. Both primary models passed d=1 learnability, their effective/allocated parameters (5,704/14,920), H64 state, degree16, 64 self-loops, training data/tokens and dense-execution proxy matched; M3/C0 initial parameter values were paired. The wide dense GRU has14,920 **effective** parameters and is contextual only. No graph, source, test-stream or resource mismatch was found. The CI conditions on one frozen 4,096-episode test stream, not arbitrary future task draws.

**Registered decision: INCONCLUSIVE** for the `>0.05 nat` useful block advantage. The control headroom guard failed: C0 d=4 mean KL0.012463 versus required≥0.07, and0/10 seeds had KL≥0.05 versus required8/10. Since KL is nonnegative, the maximum possible improvement over this C0 mean on this test stream is only0.012463 nat; the stated useful gain could not be demonstrated at this endpoint. The effect CI upper bound being below0.05 does **not** override the predeclared guard or justify a post-hoc `FALSIFIED` label. The d=8 difference is secondary, variable and cannot replace d=4.

The immediate design failure was ours: CAL-0004 qualified d=4 using a dense H64 baseline with KL around0.02, but the later primary control C0 was not calibrated independently before freezing a0.05-nat useful-gain target. The no-outcome preflight verified resources and streams yet did not test **effect-size feasibility** against the actual matched control. This is separate from the genuine EXP-0004 negative result on an older lookup-style task. Neither this near-zero structural comparison nor one clean seed replay says that every modular or particle-based architecture is useless. It establishes no candidate-specific hypotheses, attractor escape, SAE, biological effect or Gate 1 passage.

The action is to stop treating all-input independent blocks as evidence for the broader H3 multi-hypothesis claim. The benchmark-design skill and experiment protocol now require primary-control oracle-relative calibration on disjoint streams before selecting a useful-effect threshold. PF-RNN, in addition to RIMs, has been added to the prior-art boundary for any future candidate-specific trajectory mechanism. No d=8 endpoint promotion or mechanism added to rescue this result.
