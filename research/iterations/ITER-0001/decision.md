# ITER-0001 decision

SUPPORTED for baseline curriculum recovery. Independent review: review.md; evidence: artifacts/EXP-0003/v2/analysis.json. All12 registered runs completed. No architectural stage gate advances.

The noise curriculum achieves mean joint success99.35% at delay8; direct training0.10%. The blank-direct condition is unstable across seeds, so removing input payload alone does not reliably resolve the floor. Training path is the accepted practical intervention; its algorithmic mechanism is unresolved.

Next branch: EXP-0004 tests long-delay transfer after the unchanged curriculum on all five model variants, ten new seeds, and fresh final test streams. IID8 checks learning; OOD32 is primary and64 robustness to avoid the dense calibration ceiling at8/16. This design choice uses calibration evidence and is frozen before the new study. No adaptation or biological graph.
