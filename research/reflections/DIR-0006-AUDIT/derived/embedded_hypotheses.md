# Hypotheses

| ID | Hypothesis | Required causal evidence | Current status |
|---|---|---|---|
| H1 | Topological properties support identifiable computations | Feature interventions with matched graph controls | UNTESTED; EXP-0004/0009 block-vs-random graphs change several coupled properties, and fixed-partition Q is algebraically tied to cross-edge count; see decisions/0004_h1_h2_evidence_scope.md |
| H2 | Dynamics interact with topology to improve computation | Matched interventions and dynamics ablations | UNTESTED; existing comparisons use one GRU dynamics family without a topology×dynamics factorial or adaptation ablation |
| H3 | Multiple recurrent trajectories improve delayed-evidence revision | Matched state/parameters/compute, OOD and causal checks | Broad H3 unresolved; EXP-0004 FALSIFIED the minimal independent-block >=3pp transfer prediction |
| H3-semantic | Trajectories encode distinct hypotheses | Decoding + selective lesions beyond RIM/PF-RNN prior art | UNTESTED; current all-input blocks have no candidate assignment, and the current implementation design is NO-GO pending a distinct falsifiable prediction |
| H-adaptation | Adaptation beats noise/diversity/extra compute at useful escape | Gate 2 experiment after Gate 1 | UNTESTED |
| H-SAE | Distance increases without relevance/usefulness collapse | Ground-truth association benchmark and controls | UNTESTED |

Statuses here describe evidence coverage. Completed experiment decisions use SUPPORTED, PARTIALLY_SUPPORTED, INCONCLUSIVE or FALSIFIED. A bounded task failure does not universally falsify H3.
