# DEC-0001 — Minimal generic mechanism test

2026-09-23. Start with the user's Q1, not biological data or an LM. A minimal block-masked GRU is mathematically equivalent to independent GRU cells plus joint readout; it is a controlled modular baseline, not a novel architecture. Reuse PyTorch optimization/autograd; inspect conn2res but do not add a fixed-reservoir framework to a BPTT experiment. No external research code has been copied or claimed reproduced.

The independent design review identified the cache-and-selector solution. We retain the lightweight benchmark and explicitly limit its claim to retrieval and observable answer revision. Supervise pre/post outputs and evaluate counterfactual pair success. More elaborate belief-integration tasks require a separately registered experiment.

Independent pre-run source review identified diagonal self-feedback as an uncontrolled graph property. Archive v1 without a scientific run, register v2 fixing diagonal edges while preserving row/column degrees, and record the amendment. Do not silently rewrite the original experiment.

Primary pair has exact state/parameter/dense-operation equality. Dense small/wide and tanh controls provide context but do not each match every dimension. Infer neither semantics nor topology-specific biological utility from this comparison.
