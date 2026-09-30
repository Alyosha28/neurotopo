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
A: generic mechanism on delayed evidence, distractor resistance, associative bridge, rule switching, analogy and compositional recombination. M0 GRU; M1 wider capacity control; M2 sparse recurrence; M3 multi-trajectory; M4 adaptation; M5 communication; M6 DARC. Only implement mechanisms allowed by current gate. For a new H3 candidate, write its falsifiable difference from simple filtering and relevant PF-GRU/RIMs mechanisms before implementation. Before a Gate 1-motivated task qualification, first specify that candidate or a named reference K factor, its ablation, the actual proposed matched primary control, and the decision changed by each qualification outcome; STR-0002 stops candidate-free headroom hunting without banning independently registered benchmark science. After qualifying the actual primary control, validate only the relevant strong baselines' defining behavior and resource accounts as preflight; see `research/design/gate1_baseline_dependency_audit.md`. Prior-art validation alone cannot pass Gate 1, and relative-prior-art novelty is a separate claim from its current generic-effect rule.
B: after a qualifying generic effect, more complete representative expected-behavior validation of TRNN, RIMs, conn2res, NCP and CTM where practical. Inspection does not equal replication; do not implement unrelated baselines solely to complete this list.
C: biological topology after generic validation. Consider fafbseg-py/NAVis; immutable raw/processed/derived data with provenance and explicit filtering.
Compare biological, ER, degree-, module-, motif-matched and learned graphs as applicable; control nodes, edges, parameters, state, recurrence scale and initialization. Evaluate BCTpy/established null models before custom graph infrastructure.
**资源匹配口径：判据版本 `G1-MATCH-v3`（决定0069，取代 `G1-MATCH-v2`/决定0068 的两层口径）**——**单层，无自动降级，无数值上限**。Gate 1 主张只须匹配**一条**轴：机制实际消耗的轴，注册时声明并经 pilot 实测的超出消耗 argmax 验证；给 `SUPPORTED` 判词须 **compute matched-or-operator-profiled**。**state 匹配不再必须**；state 及其余各轴（分配/有效参数、样本、更新、内部步数、时延）一律在 manifest 中按**实测比值披露**，匹配与否均须披露。时延为受计费报告指标，**除非被声明为消耗轴，不单独阻断 Gate 1 判词**。每条 Gate 1 判词**仅就所声明的轴条件化成立**；`state` 未匹配的比较**不得**被读作分离出机制的组织结构。整包比较走带资源未匹配标志的 `PACKAGE_EFFECT` 轨，禁止聚合或作为 Gate 1 证据引用。决定0067 §5 的**日落条款继续有效**。每个注册存储 `matching_criterion_version`。**预注册、控制不得弱化、全轴披露、筛查阈值冻结、其余全部阶段门不变。** 见 `research/decisions/0069_gate1_matching_relaxation_v3.md`。
Extract compact synthetic principles and perturb individual properties. Later optimize task graphs and quantify similarity with biology; visual similarity is insufficient.
Only after replication and intervening gates consider language-like tasks, independent LM, then scaling.

## Separate claims
Sequence modeling feasibility, specific benchmark superiority, efficient scaling and general competition with Transformer/Mamba are four separate claims. Prior-dominance errors, memory retrieval/interference, reasoning/belief revision and deception are separate research programs. Topology alone does not establish safety or solve deception.

## Tool reuse and collaboration
Inspect FlyVis, conn2res, TRNN, CTM, RIMs, ncps and primary connectome-reservoir/BPU work. Evaluate ReservoirPy, BrainPy, NeuroGym only when assumptions fit. Engineering repositories such as fly.ai/wetware/connectome-bpu are not scientific evidence by themselves. Inspect license, maintenance, tests, paper/code alignment, reproduce a minimum behavior and pin the commit before adoption.
Delegate bounded prior-art, implementation, independent criticism and reproduction when useful; integration remains with the orchestrator. Votes are not evidence. Do not use Astra without explicit authorization or send private code to an additional provider without authorization. Writing follows stable evidence.

## Iteration
One question → prior art → strongest rejection → frozen specification → minimum implementation → sanity/overfit/leakage/accounting checks → registered run → uncertainty/cost analysis → adversarial review → four-way decision → repository update. Expected information gain and discriminating experiments outrank added complexity. Negative results are first-class records.
