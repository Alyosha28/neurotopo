# Ordered research questions

1. Does maintaining multiple recurrent latent trajectories improve delayed-evidence hypothesis revision relative to single-trajectory models when parameter count, state capacity, training budget and inference compute are controlled?
2. Does adaptive inhibition improve escape from dominant invalid states more than perturbation, more compute or explicit diversity?
3. Do trajectory interventions show hypothesis-specific functional specialization rather than redundant copies?
4. Do extracted biological topology principles improve dynamics beyond degree-, module-, motif- and compute-matched synthetic graphs?

Current operational scope remains Q1, but tested implementations are narrow: EXP-0004 falsified a minimum block benefit on lookup-style delayed evidence; EXP-0009 was inconclusive on exact-posterior nuisance transfer because the matched control reached a d=4 ceiling. Neither establishes semantic candidate trajectories, SAE or biology. See `research/evidence/exp0009_posterior_transfer.md` and `research/decisions/0002_gate1_benchmark_strategy.md`.
