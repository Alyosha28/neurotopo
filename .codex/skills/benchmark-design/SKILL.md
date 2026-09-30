---
name: benchmark-design
description: Design or audit NeuroTopo synthetic cognitive benchmarks and preregistered model comparisons, including ground truth, shortcuts, matched controls, OOD splits and falsification.
---

# Benchmark design

Read `docs/experiment_protocol.md`, the current stage gate and `research/research_questions.md`. Work on one question and identify the causal-chain link it tests.

1. Derive a simple uninteresting solution first. If caching plus late selection solves a task, label it retrieval/observable revision until stronger evidence exists. For probabilistic tasks, write the smallest sufficient state and an exact or bounded algorithmic oracle; compare its size and update rule to the single-state baseline before adding complexity.
2. Define latent ground truth, inputs available at each time, target generation, nuisance variables, independent RNG streams and split shifts. Build shared-prefix changed-evidence and shared-evidence changed-memory controls. Verify chance/shortcut/full-information oracles without leaking targets. Distinguish Bayes-limited true-label accuracy from model headroom **relative to the oracle posterior**; noisy labels alone do not prevent a ceiling.
3. Specify a strong conventional baseline, closest prior-art mechanism, matched control and clean self-ablation. **Calibrate the actual primary control**, not only a dense contextual baseline, on disjoint streams before selecting a minimum useful improvement: for a nonnegative loss, the maximum possible gain over that control cannot exceed its error to the oracle. Separate benchmark qualification, the observed fixed-task effect bound, and the broader architecture claim in reports; a failed headroom gate is not a general falsification. Distinguish effective/allocated parameters, recurrent state, input tokens, optimizer updates, internal steps and actual executed operations. Degree matching must also inspect self-feedback and other changed graph properties.
4. Freeze primary and secondary outcomes, replication unit, seeds, thresholds, uncertainty method, failure handling and budget in `experiments/EXP-XXXX/vN.md` plus config/registry before implementation or outcome inspection. A protocol amendment gets a new version and reason; never overwrite original thresholds.
5. Before a formal architecture comparison, qualify the benchmark on disjoint calibration streams using a strong conventional single-state model at a fixed budget. Inspect proper scores against the oracle, learning adequacy, shortcut gaps, held-out difficulty and time-resolved/counterfactual update behavior. Store all failed calibrations; do not select a favorable condition using architecture outcomes. If repeated configurations saturate because their sufficient state fits easily in the baseline, stop increasing distractors or hidden-variable counts and revisit the scientific question.
6. Plan leakage, equivalence, gradient, deterministic, tiny-overfit and resource tests. Keep calibration/sanity data independent of final scientific evaluation.

Deliver a discriminating registered experiment within the current gate. Do not introduce adaptation, DARC, biology or LM to rescue an unresolved earlier effect.

References and adaptation choices: `docs/skill_reference_audit.md`, especially K-Dense experimental design. External agent-skill benchmark rubrics do not substitute for scientific model comparisons.
