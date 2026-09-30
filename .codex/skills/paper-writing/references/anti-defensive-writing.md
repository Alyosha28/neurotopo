# anti-defensive-writing — evidence-preserving NeuroTopo adaptation

Based on inspection of [Kiterlin/anti-defensive-writing](https://github.com/Kiterlin/anti-defensive-writing/blob/4b1938cdff9f7796e26065357f6ccbee5fd552ce/skill/anti-defensive-writing/SKILL.md), MIT, inspected 2026-09-23. The project audit also compares a higher-star alternative whose result-selection advice is excluded. Full provenance: `docs/skill_reference_audit.md`. The following workflow is newly written for this project.

For each potentially defensive sentence, record its location, function, necessary information, and retain/rewrite/delete decision. Remove repetitions and imagined-objection language that adds neither evidence nor precision. Replace vague hedges with a concrete population, task, effect estimate or source of uncertainty. Lead a paragraph with what was learned; keep limitations where they affect that claim.

Preserve restrictions affecting validity, causal identification, applicability, methods, ethics or interpretation. Consolidation is allowed; suppressing an unfavorable result is not. Never convert an uncertain/falsified experimental decision into support, remove an inconvenient baseline, select a new metric after seeing results, or call a synthetic behavioral effect a biological mechanism.

Examples:

- Repetitive: “We do not claim a universal architecture advantage; this should not be interpreted as general superiority.”
  Direct: “The comparison estimates the effect of block recurrence on delayed-evidence retrieval under a fixed training budget.”
- Vague: “This may possibly suggest that independent trajectories could be useful.”
  Precise: “Across ten seeds, block recurrence changed joint accuracy by X percentage points (95% CI [L, U]).” Fill only from verified results.
- Necessary restriction: “The task is solvable by retaining candidate values and selecting after the cue.” Preserve when interpreting internal belief-revision claims; repeated copies elsewhere can be removed.

Final check: unchanged facts and claim strength; all material negative evidence retained; main finding stated early; uncertainty specific; no repeated apology or fabricated certainty. This is an editorial check, not an AI-detection guarantee.
