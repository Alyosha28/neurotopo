# Auto-research: training path restored learning; independent blocks did not add transfer value

2026-09-23. Two registered research cycles distinguish a training failure from an architectural result. The fixed curriculum first made the original delayed-evidence task learnable. With that prerequisite satisfied, the resource-matched comparison rejected the proposed useful gain from independent recurrent blocks on its prespecified primary transfer endpoint.

## Training conditions

EXP-0003 used four conditions and three paired seeds: original or zeroed distractor payload, each trained directly at delay8 or through the fixed0→2→4→8 schedule. Every run used6000 updates and identical dense H64 GRU settings. On fresh original-noise delay8 tests, curriculum joint success averaged99.35%, versus0.10% with direct training. Each curriculum seed passed the registered pre/joint learning thresholds and gained at least10pp. Independent review accepted SUPPORTED for this bounded recovery claim.

Removing distractor payload alone was insufficient across seeds: blank_direct learned in only one of three seeds. The successful curriculum processed5,248,000 tokens versus6,912,000 for direct training; improvement therefore did not come from more executed linear operations. Training distribution and target-delay exposure changed, so this is a training-path result, not a matched-compute efficiency claim or proof of a particular gradient mechanism. Curriculum learning itself is established prior art.

## Architectural comparison

EXP-0004 froze that curriculum for all five models, used ten fresh paired seeds and fresh final tests. Delay8 checked learning, delay32 was the primary transfer endpoint and64 the robustness endpoint; these were chosen before the run to avoid the calibrated near-ceiling scores at8/16. M3 and C0 each had6212 effective parameters,15428 allocated parameters,64 state values,64 self-loops and identical training/evaluation counts. Initial parameter hashes and common test hashes matched. Their ten IID pre/joint accuracies were all100%.

| Model | Joint success, delay8 | delay16 | Primary delay32 | Robustness delay64 |
|---|---:|---:|---:|---:|
| Small dense GRU, H16 | 66.92% | 53.80% | 32.90% | 20.33% |
| Wide dense GRU, H64 | 100.00% | 99.79% | 90.34% | 63.69% |
| Sparse tanh RNN | 0.01% | 0.03% | 0.03% | 0.04% |
| Independent block GRU, H64 | 100.00% | 93.19% | 57.94% | 33.77% |
| Matched random-mask GRU, H64 | 100.00% | 98.08% | 67.04% | 35.13% |

At delay32, block minus random was−9.10pp (paired-seed95% bootstrap CI−14.90 to−3.60pp). This excludes the registered minimum useful gain of+3pp and shows a disadvantage on this endpoint. At64, the difference was−1.36pp (−10.18 to+6.73pp), leaving the sign uncertain. The conventional wide GRU also exceeded the block model, although that contextual comparison has more effective parameters and cannot isolate the same structural factor.

Independent review recomputed the paired results and accepted **FALSIFIED** for the stated fixed-budget, task-specific ≥3pp primary improvement prediction. It does not falsify every possible multi-trajectory architecture. The small GRU and tanh RNN did not meet the learning standard and serve only as contextual outcomes. Ten seeds cover coupled graph/initialization/training variability; the intervals condition on the fixed held-out sample sets.

## Scientific decision

Gate1 remains closed. The registered branch is STOP_MECHANISM_ESCALATION: this negative result does not license adding adaptation, communication, DARC or biological wiring as a rescue. Candidate caching plus selection still explains the task, and no semantic hypothesis specialization was tested. Prior EXP-0001/0002 decisions are retained; improved training here does not rewrite them.

Evidence: `artifacts/EXP-0003/v2/`, `artifacts/EXP-0004/v1/`, both protocols and `research/iterations/ITER-0001/review.md`, `ITER-0002/review.md`. Subset reproduction is documented separately in `docs/EXP-0004_reproduction.md`; it is an engineering repeatability check, not another ten-seed experiment.
