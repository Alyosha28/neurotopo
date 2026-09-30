---
name: prior-art-audit
description: Audit NeuroTopo mechanism proposals and novelty claims against primary papers and source code; update the project prior-art evidence matrix before proposing a contribution.
---

# Prior-art audit

Read `research/stage_gates.md`, the active registration, and the relevant entries in `research/prior_art/`. Start from one proposed causal mechanism and its strongest alternative explanation.

- Search exact terms, mechanism aliases, older work and counterexamples. Cover FlyVis, conn2res, TRNN, CTM, RIMs, ncps, connectome reservoirs and BPU when the mechanism overlaps; record unresolved identity instead of merging similarly named projects.
- Use primary papers and official repositories. Record the query date, search boundary, direct source, inspected commit/file, license, tests and maintenance evidence. Separate author-reported results, inspected code, locally reproduced behavior and inference.
- Update `evidence_matrix.md`, `novelty_boundary.md`, `mandatory_baselines.md`, `reusable_projects.md` and open questions as relevant. Record what each source establishes and leaves untested along the topology-to-capability chain.
- Check whether the mechanism exists under another name, whether the difference is architectural, and whether a registered ablation can measure it. A search miss does not establish novelty.
- Before reuse, verify scientific assumptions and license, wrap the dependency, pin it and reproduce a minimum behavior. Do not call code inspection replication.

Output the closest overlap, strongest baseline, reusable component, unresolved rejection reason and measurable discriminating experiment. Do not promote a candidate to a contribution from terminology alone.

Adaptation provenance: `docs/skill_reference_audit.md` (K-Dense experimental/literature workflows and DeerFlow review workflows). Local protocol overrides external workflow assumptions; no arXiv-only restriction, fixed search cap or mandatory delegation.
