# Frozen models lose accuracy when irrelevant payload is removed

EXP-0005 v2 tests a specific explanation for the long-delay loss seen in EXP-0004. Ten pairs of already trained block and random-mask GRUs, plus a contextual dense GRU, received the same new counterfactual instances at delay32 and64. All weights, event positions and output labels were held fixed; only the task-irrelevant distractor candidate/value channels changed. Thirty input checkpoints were copied byte-for-byte from the frozen earlier run and pinned before this test. Every checkpoint passed exact replay of its original delay32/64 scores before entering the new test.

At the preregistered delay32 endpoint, removing all distractor payload brought the block and random models closer in **relative** score, but both became worse in **absolute** score:

| Frozen model | Original full payload | Blank payload | Change |
|---|---:|---:|---:|
| Independent block GRU | 58.15% | 40.40% | −17.75 pp |
| Degree/self-loop matched random GRU | 67.50% | 41.29% | −26.21 pp |
| Contextual wide dense GRU | 90.42% | 67.00% | −23.42 pp |

The block-minus-random gap moved from −9.35 to −0.89 pp. Its seed-paired gap interaction was +8.46 pp, but the 95% bootstrap interval was [−0.30,+17.48] pp. More importantly, the random control's blank score fell by 26.21 pp, beyond the preregistered 3 pp validity guard. The block model also lost 17.75 pp, violating the registered absolute-recovery requirement. The analysis therefore returns **INCONCLUSIVE** for the hypothesis that removing payload meaningfully alleviates a block-specific deficit.

The equal-amplitude `repeat` condition reduced the full-input block gap from −9.35 to −6.29 pp at delay32; its exploratory interaction interval [−1.25,+6.77] pp spans zero. Repetition keeps each distractor's candidate/value two-hot amplitude while changing its temporal correlation, so this result cannot cleanly isolate input magnitude or internal memory decay. ID-only and value-only conditions were exploratory and also changed the training distribution. The registered comparisons use ten paired trained-seed units on one fixed fresh test stream; test-instance sampling uncertainty is outside their confidence interval.

This diagnostic finds a substantial response to an out-of-training input transformation across **all three** models. The observed gap closure under blank input is driven partly by worse performance of the matched control and does not explain EXP-0004's negative architecture result. Gate1 remains closed. Source: `artifacts/EXP-0005/v2/results/analysis.json` and all thirty per-model results. Independent scientific review confirmed the decision in `research/iterations/ITER-0003/review.md`. A clean-checkout reproduction exactly matched all three seed9101 model records and 120 prediction arrays (`docs/EXP-0005_reproduction.md`); it does not add training-seed evidence.
