# ITER-0007 — matched recurrent structure on posterior transfer

One question: **under the same varied d=0/1/2 training schedule, does four-block independent recurrence reduce d=4 exact-posterior error relative to a degree-, self-loop-, state-, effective-parameter- and compute-matched random-mask GRU?**

Tested link: recurrent topology → learned dynamics → computed joint-posterior representation/readout → nuisance-duration transfer. CAL-0004 qualified the *training distribution* using only a dense single-state GRU; it did not inspect block or random-mask outcomes. EXP-0004's prior negative block effect on a lookup-style delayed task remains intact. The strongest rejection is that a d=4 difference could reflect optimization under one schedule, graph mixing or training stability rather than semantic competing hypotheses. The wide dense H64 GRU has more effective parameters and is contextual, not the primary causal control. Gate 1 remains NOT_PASSED before this run.
