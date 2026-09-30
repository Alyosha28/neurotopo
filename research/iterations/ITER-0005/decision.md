# ITER-0005 decision

**PARTIALLY_SUPPORTED** for the *direction* of excess H8→32 fixed-ridge prior-answer readout transfer loss in M3 relative to C0. The preregistered paired-seed mean is+8.91pp with 95%CI[+1.57,+15.78]pp. The lower interval bound exceeds zero but not the registered +5pp practical threshold. All thirty model-seed runs, model/probe source checks, original-score replay, four adequacy/stability/behavior guards, independent statistics review and the seed9101 clean-checkout numerical subset passed.

With one A8 train-only scaler per model/seed, H8A→32 accuracy was M3 66.99% and C0 76.36%; H32A→32 on the same states was97.77% and98.22%. The source and target B heads differed by less than the registered 2pp mean-absolute refit limit. The original pretrained head's pre-answer scores on the same new task instances were72.23% and80.41%. This supports a fixed-probe transfer difference, not a causal explanation of the original multi-output head or a pure elapsed-time effect: longer delays also add random distractor inputs.

Gate1 remains closed; EXP-0004 and EXP-0006 verdicts stand. Registered next branch for this diagnostic: STOP_MECHANISM_ESCALATION. A separate frozen-state diagnostic of the **revise** branch's final selected-answer accessibility would test a different unresolved representation link without changing any recurrent mechanism; it requires its own protocol, data and decision. No secondary endpoint from EXP-0007 changes its primary verdict.

Evidence: `artifacts/EXP-0007/v2/analysis.json`, `research/iterations/ITER-0005/review.md`, `docs/EXP-0007_reproduction.md` and `research/evidence/exp0007_cross_delay_transfer.md`.
