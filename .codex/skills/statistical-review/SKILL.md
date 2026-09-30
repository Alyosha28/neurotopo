---
name: statistical-review
description: Audit NeuroTopo seed-level comparisons, effect sizes, uncertainty, resource fairness and preregistered decisions before accepting an experimental finding or stage-gate change.
---

# Statistical review

Read the frozen protocol before examining the outcome. Check manifest completeness, source-hashed validation, full seed coverage, dataset identity and the exact resource-matching claim.

- Identify the replication unit. EXP-0001 pairs model/graph/training seeds; thousands of test examples do not become independent architecture replications. Shared test-set uncertainty is separate from seed variability.
- Recompute paired differences, mean, sample SD, CI and the registered minimum useful effect. Preserve paired sampling in bootstrap. Report every seed and failed run, not just the best checkpoint/model/delay.
- Verify code implements the registered decision branches, confirm/learning guardrails and primary-vs-secondary distinction. A robustness conjunction is not permission to select whichever endpoint succeeds.
- Inspect optimizer/tuning fairness, self-connections and other graph confounds, state capacity, effective and allocated weights, useful and executed compute, inference steps and wall time. Any remaining mismatch limits interpretation.
- No significance means neither equivalence nor falsification. Inadequate learning or uncertain effects are INCONCLUSIVE; a FALSIFIED label applies only to the exact prediction whose registered meaningful effect has been excluded.
- Distinguish task performance, representation semantics, causal mechanism, biology and transfer. Record the simplest remaining explanation and a four-way decision in the repository.

Reference selection: `docs/skill_reference_audit.md` (K-Dense statistical-analysis/power principles). Do not import default independent-sample tests into paired-seed studies.
