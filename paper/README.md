# Reproduction package: matched-capacity identifiability study

Everything in `paper/` is generated from the scripts below. Machine outputs are frozen JSON with byte
hashes listed in `dir0003_repro_manifest.json`; no number in the manuscript is entered by hand.

## Environment

Per `docs/reproduction.md`, this mounted workspace needs an environment and cache on the local filesystem:

```sh
export UV_CACHE_DIR=/private/tmp/neurotopo-uv-cache
export UV_PROJECT_ENVIRONMENT=/private/tmp/neurotopo-venv
uv sync --frozen --extra dev
```

No plotting or statistics dependency is used: the figures are emitted as hand-written SVG.

## Steps (all deterministic, CPU only, no model training)

```sh
# 1. exact infinite-horizon capacity functional + full matched-allocation enumeration
uv run --frozen python scripts/dir0003_v1_exact_check.py \
    --output research/design/dir0003_v1_exact_check.json

# 2. specification curve (12 Gaussian-weight seeds x 3 readout ridges) with the R1 control
uv run --frozen python scripts/dir0003_spec_curve.py --seeds 12 \
    --output research/design/dir0003_spec_curve.json

# 3. out-of-sample cells: separated bands at phi=0.5 and phi=0.3, adjacent bands at phi=0.7
uv run --frozen python scripts/dir0003_spec_curve.py --seeds 8 --phi 0.7 \
    --label heldout_phi --output research/design/dir0003_spec_curve_heldout_phi.json
uv run --frozen python scripts/dir0003_spec_curve.py --seeds 8 --label heldout_bands \
    --bands "0.10,0.30;0.50,0.70;0.85,0.95" \
    --output research/design/dir0003_spec_curve_heldout_bands.json
uv run --frozen python scripts/dir0003_spec_curve.py --seeds 8 --phi 0.3 --label heldout_both \
    --bands "0.10,0.30;0.50,0.70;0.85,0.95" \
    --output research/design/dir0003_spec_curve_heldout_both.json

# 4. R2: nested re-matching inside the static-nonlinearity core
uv run --frozen python scripts/dir0003_tanh_matched.py --seeds 3 \
    --output research/design/dir0003_tanh_matched.json

# 5. figures
uv run --frozen python scripts/dir0003_figures.py
```

Historical (superseded but retained, because the manuscript's provenance section refers to them):

```sh
uv run --frozen python scripts/dir0003_capacity_precheck.py --blocks 4 \
    --output research/design/dir0003_capacity_precheck.json
uv run --frozen python scripts/dir0003_ar1_transfer_check.py \
    --output research/design/dir0003_capacity_precheck_ar1.json
```

## Reading order for the evidence

1. `dir0003_v1_exact_check.json` — self-checks (closed-form Gram vs direct summation; `phi = 0` cross-check
   against an independent truncated implementation) and the matched-rung census.
2. `dir0003_spec_curve.json` and the three `heldout_*` files — the specification curve per cell.
3. `dir0003_heldout_summary.json` — the 2x2 table with the pre-fixed verdict rule applied.
4. `dir0003_tanh_matched.json` — the nested nonlinear control.

## Rules this package follows

- Exact untruncated totals only, under the task input law; the only truncated quantity is the nonlinear
  control's lag sum, whose tail fraction is reported with it.
- Contrasts are reported as distributions over a specification curve, never as a single point.
- The artifact band is measured in the same run as the contrast it is compared against.
- Negative and superseded results are retained: the two independent review packets
  (`research/reflections/DIR-0003-PREDESIGN/`, `research/reflections/DIR-0003-REDESIGN/`) and the decisions
  `research/decisions/0074`–`0079` record what was rejected and why, including three implementation or
  specification errors found by self-checks and review.

## What this package does not claim

No capability claim, no biological claim, no novelty claim, and no Gate 1 claim for any architecture
comparison. Gate 1 of the host project remains `NOT_PASSED` and is untouched by these artifacts.
