# Final selected answer is readable, but the model contrast is unresolved

EXP-0008 v1 tests frozen recurrent states after a late selector and final query, using the original full-noise task. Each counterfactual pair stays wholly within train, validation or test; both confirm and revise siblings train a fixed 260-coefficient ridge head, while **one revise row per held-out pair** defines the primary score. The probe's labelled training and train-only scaler do not update the original recurrent model. Ten original trained/graph seeds are paired across the block and random-mask models; their state/parameter/resource counts remain matched.

| Delay32 revise final answer | Independent block M3 | Matched random C0 | Contextual wide GRU |
|---|---:|---:|---:|
| Newly fitted final-state linear head | 94.33% | 95.76% | 99.09% |
| Original pretrained shared readout | 92.25% | 95.12% | 98.94% |

M3−C0 probe accuracy differs by **−1.43 percentage points** on the new final test. The paired-seed 95% bootstrap interval is **[−3.30,+0.31] pp**, so it includes both zero and the registered −3pp meaningful-deficit threshold. All provenance, validation-positive-control, independent refit stability and original revise-behavior guards passed. The automatic registered decision is **INCONCLUSIVE** for a ≥3pp final-state linear-access deficit. Delay64 is secondary: M3/C0 probe revise accuracy 86.43%/86.92%, paired interval for their difference [−4.15,+3.58] pp.

The fitted head reads the selected answer from both frozen states much better than 25% class chance, and M3's fitted score exceeds its pretrained revise accuracy on these instances by about 2.09pp. C0's corresponding difference is about0.63pp. These are descriptives under **additional labelled delay32 probe fitting**, not a matched-budget change to the original architecture. The registered M3-vs-C0 contrast remains uncertain; a new head's high accuracy cannot identify whether the pretrained head failed through calibration, cue binding, nonlinear geometry or another part of the original computation. The earlier EXP-0004 FALSIFIED benefit prediction, EXP-0006 FALSIFIED large pre-evidence linear-access deficit and EXP-0007 PARTIALLY_SUPPORTED probe-transfer direction all retain their own scopes. Gate1 is still closed.

All model-seed/delay outputs, coefficients and source/model/data hashes are in `artifacts/EXP-0008/v1/`. An independent result review and clean-checkout seed9101 reproduction must be included when ITER-0006 closes. No favorable confirm or delay64 endpoint replaces the preregistered revise delay32 test.
