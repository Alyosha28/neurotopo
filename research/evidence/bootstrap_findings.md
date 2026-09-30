# Bootstrap findings — 2026-09-23

NeuroTopo now has a frozen CPU reference, five recurrent baseline variants, a counterfactual delayed-evidence task, registered execution and resource accounts. The first comparison is **INCONCLUSIVE** because the task-learning prerequisite failed. A separate diagnostic identifies sensitivity to the tested combination of extra distractor events and a longer recurrent sequence under the fixed schedule.

## What ran

EXP-0001 v2 used five models and ten paired seeds, with 400 updates per model. Block and random-mask GRUs each have 64 dynamic state values, 6,212 effective parameters, 15,428 allocated parameters, 64 self-connections, and matched in/out degree and dense executed operations. For an 18-event evaluation sequence, each executes 266,624 linear MACs per example, with 100,736 nominal active MACs. Dense execution means sparse weights do not save those kernel operations.

The main models' IID early-answer accuracies averaged 25.17% and 24.97%, respectively; the requirement was 90%. Their joint counterfactual success difference was +0.137 percentage points (paired-seed bootstrap 95% CI +0.029 to +0.244 pp), against a minimum useful effect of 3 pp. OOD difference was −0.020 pp (−0.098 to +0.059 pp). With the learning gate failed, these floor-level differences cannot adjudicate the architectural hypothesis. All seeds and final checkpoints were retained.

EXP-0002 used only the standard H64 dense GRU, three fresh diagnostic seeds, delays0/8 and 6,000 updates. Every delay0 run reached 100% joint success by the measured update2000 checkpoint and remained there. At delay8, final pre accuracy was 26.27%, 24.12% and 27.05%; joint success was 1.46%, 0% and 0%. This supports short-delay feasibility while leaving delayed-task learnability unresolved. Delay0 versus delay8 changes both distractor exposure and recurrent sequence length; it does not isolate pure elapsed time. This neither revises EXP-0001's verdict nor identifies a unique optimization cause.

## Strongest remaining rejection

The experiment presently has a learning floor, so an architecture difference would be uninterpretable. Even a successful future run can use candidate caching and late selection; observing an answer change does not establish semantic hypothesis trajectories. These are separate identification problems: first achieve adequate delayed-task learning under a newly registered protocol, then design causal trajectory tests.

## Engineering evidence and boundary

Twelve unit tests cover PyTorch GRU equivalence, independent-block equivalence, masking gradients, degree/self-connection invariants, task labels and counterfactual prefixes, absence of future leakage, deterministic training and exact resource accounting. All five model variants reached perfect joint success on the independent eight-pair overfit check. These are implementation checks; generator learning is measured separately above.

Prior-art audit covered all eight requested families and a recent hierarchical modular reservoir paper. It found strong component overlap; no novelty claim is established. Source inspection is recorded separately from replication. GitHub skill audit covered nine repositories before the five local skills were created. Writing incorporates evidence-preserving anti-defensive-writing and remains gated by stable evidence.

## Decision

Gate 0 infrastructure is operational, with independent subset reproduction recorded in `docs/independent_reproduction.md`. Gate 1 has not passed. This bootstrap closes with preserved inconclusive/negative evidence and a precise unresolved learning problem; adaptation, DARC, biological graphs and language modeling remain outside the active evidence-supported stage.
