# Independent scientific review record

2026-09-23. A separate GPT-6 Sol worker configured with high reasoning reviewed design, implementation before execution, and final interpretation. The main agent integrated and verified corrections. Review judgments are checks on evidence, not independent empirical replications or majority votes.

## Design and pre-run findings

- Endpoint-only supervision allowed the prior cue to be ignored. The frozen task requires a pre-evidence answer and a post-evidence answer, while explicitly retaining caching-and-selection as a simpler explanation.
- Unconditional counterfactual joint success avoids defining the primary evaluation subset by each model's own early correctness. Conditional revision accuracy remains secondary.
- V1's random graph changed diagonal self-feedback. V2 freezes all self-connections, preserves in/out degrees, tests those invariants and checks them in analysis. V1 had no scientific run.
- The wider dense GRU criterion in analysis originally included a CI requirement absent from registration. It was corrected before execution to the registered mean threshold.
- Engineering checks are now fingerprinted in each formal manifest, preventing a scientific decision without matching source validation.

## Outcome review

EXP-0001 v2's INCONCLUSIVE decision follows the failed 90% early-answer requirement. A small positive IID paired interval near the performance floor cannot establish benefit; excluding a 3 pp effect there also cannot bypass the learning requirement to assert falsification.

EXP-0002's PARTIALLY_SUPPORTED decision applies only to baseline feasibility at delay0. Added delay events change both distractor exposure and recurrent sequence length. No unique causal optimization explanation is established. Gate 1 remains closed.

The reviewer required the independent reproduction file to exist before the final Gate 0 status is marked passed; this is resolved by the separate report in `docs/independent_reproduction.md`. That reproduction covers one full seed across all models; no ten-seed or cross-hardware reproduction is claimed.

## Collaboration scope

Two GPT-6 Luna workers configured with max reasoning performed prior-art and GitHub skill-source audits. A separate GPT-6 Luna max context performed clean-checkout reproduction. All writing/code changes were integrated by the main agent; there was no DeepSeek or Astra worker and no vote-based scientific conclusion. Upstream paper code was inspected but not reproduced.
