# Gate 1 benchmark and baseline audit

2026-09-24. Trigger: the concern that a near-perfect single-trajectory baseline leaves no room for a multi-trajectory effect. This is a post-result design audit, not an amendment to EXP-0004 v1 and not a new scientific run. The question is whether that experiment could detect its registered structural-transfer effect and whether its task measures competing hypotheses. Causal link: recurrent connectivity → computation on a synthetic delayed-evidence task; no representation-level specialization claim follows.

## Which comparison had headroom?

| Model | Role in EXP-0004 | IID delay 8 joint success | Primary OOD delay 32 | Robustness delay 64 | Interpretation |
|---|---|---:|---:|---:|---|
| Small dense GRU, H16 | Contextual low-capacity reference | 66.92% | 32.90% | 20.33% | Failed the IID learning standard; low long-delay score is not a useful causal comparator. |
| Wide dense GRU, H64 | Strong conventional context | 100.00% | 90.34% | 63.69% | At/near ceiling on short delays; more effective parameters than M3, so not an exactly matched architecture contrast. |
| Random-mask GRU, H64 | **Prespecified primary control** | 100.00% | 67.04% | 35.13% | Same 64 state values, 6,212 effective and 15,428 allocated parameters, recurrent degree/self-loops and dense executed operations as M3. Substantial primary-endpoint headroom. |
| Independent-block GRU, H64/K4 | Intervention | 100.00% | 57.94% | 33.77% | Lost 9.10 percentage points to the primary control at delay 32. |

EXP-0004 selected delay 32 before its outcomes because delay 8/16 calibration exposed a ceiling. Its ten paired-seed delay-32 M3−C0 interval was [−14.90, −3.60] percentage points. A positive 3-point effect was measurable in principle: C0 was 32.96 points below perfect accuracy, and the observed difference was appreciably negative. Therefore a ceiling cannot explain the registered negative M3-versus-C0 result. Delay 64 did not resolve the sign. The wide GRU's 90.34% at delay 32 leaves only 9.66 points to perfect accuracy and is an important practical benchmark, but its extra effective weights prevent attributing the M3−M1 gap solely to recurrent structure. The small GRU and sparse tanh RNN are not adequate learned alternatives.

There is a second baseline-design issue: EXP-0004 labels the wide GRU contextual, yet its `SUPPORTED` rule also requires M3 to beat that larger-effective-parameter model by 3 points at both long delays, and its next-branch rule considers only `SUPPORTED` for Gate 1. That combines a matched causal contrast with a separate practical-performance bar. The threshold stays frozen for EXP-0004, whose primary matched comparison already failed. A new protocol should state these as separate claims and justify any requirement to beat a baseline with more active weights; a matched structural effect and broad practical superiority need not share one gate decision.

## What the task fails to require

`make_batch` presents each of four candidate-to-value bindings once, with the four values drawn as a permutation. A prior cue selects an early answer; late evidence directly names either the same candidate or a different one. A solver can cache the small mapping and retrieve the selected value. This is a sufficient solution without maintaining distinct evolving hypotheses. The task therefore measures learned binding, distractor retention and overt answer switching; it cannot by itself test whether four recurrent blocks are four competing interpretations. EXP-0006 found that a newly fitted linear probe could read uncued values from both frozen states at approximately 99% at its measured timepoint, consistent with the availability of a simple lookup solution. That probe does not establish how the original head computes its answer.

M3's four independent recurrent blocks receive the same input and feed one shared readout; no candidate assignment, competition or trajectory-specific intervention was part of EXP-0004. Calling this a test of generic block topology is accurate. Treating it as a decisive test of semantic multi-hypothesis reasoning would exceed both the architecture and benchmark. The strongest rejection of a new positive result on this task would remain: a state encodes the four-value table and applies the late selector, with no hypothesis maintenance.

## Design correction before another Gate 1 architecture run

Keep EXP-0004's FALSIFIED decision for its registered ≥3-point block advantage and keep Gate 1 NOT_PASSED. Do not rescue it by changing endpoints or thresholds. A future, separately registered benchmark must first specify latent hypotheses and evidence updates with a nontrivial full-information oracle, plus a simple cache-and-select oracle that can reveal the shortcut. It must use disjoint calibration and final streams, preselect a primary difficulty slice before final model results, and require a strong conventional single-state baseline to learn while retaining enough error for the prespecified useful-effect size. Report baseline and Bayes/algorithmic oracle scores on every difficulty slice; retain all slices, including ceilings and floors. Pair a capacity/compute-matched single-state control with a topology-matched control; disclose tradeoffs when exact matching is impossible. Test candidate-specific interventions only after a genuine task benefit is established, and do not infer specialization from vector separation or a fresh probe alone.

This design correction is not evidence that any multi-trajectory model will win. It identifies an unresolved construct-validity problem for the broader Gate 1 claim while preserving the existing narrow negative result.

Sources: `experiments/EXP-0004/v1.md`, `research/evidence/auto_research_cycles.md`, `artifacts/EXP-0004/v1/analysis.json`, `src/neurotopo/task.py`, `src/neurotopo/models.py`, and `research/evidence/representation_diagnostics_synthesis.md`.
