# ITER-0005 — Does a short-delay linear readout transfer to long-delay state?

EXP-0006 excluded a large difference in same-delay linear decoding of uncued candidate values, while the frozen trained task heads remained much less accurate at long delays. This new direction keeps model weights, graph masks and full-noise task semantics fixed. It tests the **transfer of a labelled linear mapping** between delay8 (within the trained range) and delay32 (outside it).

EXP-0007 v2 fits equal-sized prior-answer ridge heads at short and target delays under the **same A8 train-only normalization** within each model/seed. It compares each head on the exact same fresh delay32 hidden states, and includes independent source and target refits to reveal training-sample instability. A larger M3 transfer loss would show a decoder-family mismatch across these conditions; it would not isolate pure time, input interference or the mechanism of the original multi-output task readout.

This is a distinct observational diagnostic. Gate1 stays closed, EXP-0004/0006 decisions remain immutable, and no recurrent architectural mechanism is added.
