# Code and environment provenance

All NeuroTopo source in `src/neurotopo/` was written for this registered protocol. No upstream research implementation was vendored. Prior-art inspection, adoption status and source pins are recorded separately in `research/prior_art/`.

The GRU follows the reset/update/new convention of [PyTorch GRUCell](https://docs.pytorch.org/docs/2.8/generated/torch.nn.GRUCell.html); equivalence to the installed PyTorch 2.8.0 GRUCell and to independent block cells is tested. Numerical determinism uses [PyTorch reproducibility controls](https://docs.pytorch.org/docs/2.8/notes/randomness.html). Recurrent masks execute through ordinary dense linear operations; no sparse-kernel efficiency is claimed.

The random synthetic control uses directed degree-preserving double-edge switches. It fixes the diagonal and verifies in/out degree and strong connectivity. Mixing count is a reproducible heuristic, not proof of uniform graph sampling. This is not an implementation of a validated biological network null model; biological claims require the `topology-null-models` workflow and dedicated library/method review.

Environment: Python 3.13 and uv lock; PyTorch 2.8.0 and NumPy 2.3.3 are deliberately fixed reference versions, not claimed latest. Tests use pytest 8.4.2. Pinning enables repeatability; future upgrades require repeating reference checks.

The established [NetworkX directed_edge_swap source](https://networkx.org/documentation/stable/_modules/networkx/algorithms/swap.html) was also inspected. Its swaps select directed paths of three edges; starting from disconnected complete blocks, these paths remain inside a block and cannot introduce inter-block edges. It therefore cannot directly randomize this initial graph. The constrained double-edge switch here explicitly permits cross-block swaps, with deterministic invariant tests and no uniform-sampling claim. A biological null-model study remains out of scope.
