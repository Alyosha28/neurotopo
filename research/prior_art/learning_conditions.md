# Learning-condition references

- [Bengio, Louradour, Collobert & Weston, Curriculum Learning, ICML 2009](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf): controlled ordering of training examples/difficulty is established prior art. We use a fixed length schedule as an optimization diagnostic; curriculum is not a NeuroTopo contribution.
- [Pascanu, Mikolov & Bengio, On the difficulty of training recurrent neural networks, ICML 2013](https://arxiv.org/abs/1211.5063): analyzes recurrent optimization difficulties and gradient-clipping remedies. A long-delay learning failure is compatible with optimization difficulty but does not identify its cause.
- Existing `evidence_matrix.md` covers RIMs, TRNN and conn2res. The current blank/noise task intervention adds a local diagnostic, not a claim of novel delayed-memory tasks.

Reviewed 2026-09-23. Papers provide methodological context; no paper result is reproduced by EXP-0003. No new research dependency or copied upstream code is required.
