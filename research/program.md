# Research program

用户研究章程的持久化摘要；2026-09-23 初始化。项目目标是隔离可迁移的拓扑与动力学计算原则，不是果蝇脑模拟器，也不预设 NeuroTopo 有优势。

## Causal chain and boundaries
Biological connectome → topological principles → dynamical principles → artificial architecture → capability. Experiments locate the tested link in Topology → Dynamics → Representation → Computation → Capability. Mechanisms already explored include biological sparse wiring, trainable connectome dynamics, reservoirs, adaptation/transients, multiple recurrent modules and internal computation time. Integrated novelty remains unestablished.

## Hypotheses and candidate mechanisms
H1: identify which sparsity, modularity, hierarchy, loops, degree heterogeneity, hubs, sparse bridges, asymmetry, clustering, inhibition, rich clubs and expansion/compression structures causally support which computations.
H2: study topology × recurrence, persistence, adaptation, inhibition, competition, transient activity, attractors/escape, heterogeneous timescales, neuromodulation and local plasticity. No dynamics mechanism is assumed helpful.
H3: persistent context C plus K candidate states Z and adaptation states A may maintain competing computational hypotheses. Require candidate decoding and hypothesis-specific causal lesions; vector divergence is insufficient.

Structured Associative Exploration (SAE) requires increased associative distance/novelty with preserved task relevance and downstream usefulness. Entropy/noise/more samples alone do not establish SAE.
DARC (divergence → adaptation → recombination → competition) is a candidate hypothesis. Adaptation must beat noise, dropout, stochastic perturbation, temperature, extra compute and explicit diversity where applicable. State averaging is not conceptual recombination. Evidence consistency, relevance, predictive usefulness and coherence must ground competition.

## Sequence of work
Current claim rules are `ARCH-GATES-v1` in `research/stage_gates.md`, authorized by decision0101. First specify a complete architecture candidate with state, update, readout, training, resource path, a differentiating prediction, a defining ablation and credible strong controls. Then verify that it runs and learns; test a registered architecture effect; identify its causal computation; evaluate new tasks, training instances and resources; assess biological attribution only if claimed; and separately test natural sequence, language-model or scale claims. These are claim levels, not a ban on bounded prototype or component research before a prior gate passes. A local component result such as EXP0027 can guide candidate design without counting as a complete architecture or a gate passage.

Generic tasks may include delayed evidence, distractor resistance, associative bridges, rule switching, analogy and compositional recombination. GRU, wider/dense, sparse, multi-trajectory and relevant prior architectures are candidate controls according to the actual claim. Inspect TRNN, RIMs, conn2res, NCP, CTM and other directly relevant prior work; inspection does not equal replication. For H3, distinguish the proposed computation from simple filtering and PF-GRU/RIMs before claiming a new mechanism. Biological source data require immutable provenance and suitable graph nulls; visual similarity is insufficient.

Future formal A2 comparisons use `ARCH-MATCH-v1` (decision0101): match the declared, pilot-verified primary budget axis; for `SUPPORTED`, match or operator-profile executed compute; disclose all other axes and include strong controls. The complete candidate may establish a scoped architecture package effect; its defining ablation and causal component attribution belong to A3. Historical project Gate 1 registrations retain `G1-MATCH-v3` and decision0069 on their original terms; earlier outcomes are not reclassified. Preregister hypotheses, controls, thresholds and resource budgets before seeing effects.

## Separate claims
Sequence modeling feasibility, specific benchmark superiority, efficient scaling and general competition with Transformer/Mamba are four separate claims. Prior-dominance errors, memory retrieval/interference, reasoning/belief revision and deception are separate research programs. Topology alone does not establish safety or solve deception.

## Tool reuse and collaboration
Inspect FlyVis, conn2res, TRNN, CTM, RIMs, ncps and primary connectome-reservoir/BPU work. Evaluate ReservoirPy, BrainPy, NeuroGym only when assumptions fit. Engineering repositories such as fly.ai/wetware/connectome-bpu are not scientific evidence by themselves. Inspect license, maintenance, tests, paper/code alignment, reproduce a minimum behavior and pin the commit before adoption.
Delegate bounded prior-art, implementation, independent criticism and reproduction when useful; integration remains with the orchestrator. Votes are not evidence. Do not use Astra without explicit authorization or send private code to an additional provider without authorization. Writing follows stable evidence.

## Iteration
One question → prior art → strongest rejection → frozen specification → minimum implementation → sanity/overfit/leakage/accounting checks → registered run → uncertainty/cost analysis → adversarial review → four-way decision → repository update. Expected information gain and discriminating experiments outrank added complexity. Negative results are first-class records.
