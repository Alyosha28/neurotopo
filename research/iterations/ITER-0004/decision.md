# ITER-0004 decision

**FALSIFIED** for the preregistered hypothesis that M3 has a >=5pp loss in linearly decodable *uncued* candidate labels relative to degree/self-loop-matched C0 at delay32. Ten paired trained seeds yield−0.48pp (95% seed bootstrap CI[−1.19,+0.14]pp). Probe positive controls, source checkpoint/data/resource matches and the original-task behavioral gap guard passed. Independent scientific review reproduced the paired analysis. A detached clean checkout reproduced seed9101 for all three models and both delays exactly, including six probe coefficient files.

The new finding is that uncued values remain highly linearly accessible in both frozen states despite a substantial difference in their own task readout performance on the same full-noise instances. This excludes a large deficit **under this fixed decoder and accuracy metric**; the probe is near ceiling, and no causal stage of the original computation is identified. Original EXP-0004 falsification and Gate1 closure remain.

Registered next action for this bounded diagnostic: STOP_MECHANISM_ESCALATION. After explicit user continuation, a new frozen-weight observational direction may test cross-delay readout transfer without altering the models. That would need its own prior-art check, registration, new test streams and review. No existing protocol or endpoint is retuned.

Evidence: `artifacts/EXP-0006/v2/analysis.json`, `research/iterations/ITER-0004/review.md`, `docs/EXP-0006_reproduction.md`, `research/evidence/exp0006_uncued_linear_access.md`.
