#!/usr/bin/env bash
# DIR-0005 prior-art fetch: is any "organisation for trainability" operation still open?
#
# The audit's binding constraint is K7 (novelty at body level). C7 admits only
# optimisation-dynamics effects at matched capacity, so the question is whether the
# organisation operations that plausibly produce such effects are already covered:
# orthogonal/unitary and antisymmetric recurrences, dynamical isometry, timescale
# separation, and structured state-space initialisation.
#
# Read-only w.r.t. the repository except for its own output directory. Raw responses kept
# verbatim; arXiv is expected to rate-limit (as recorded for DIR-0004).
set -uo pipefail

out=research/prior_art/dir0005_raw
mkdir -p "$out"
log="$out/fetch_log.txt"
: > "$log"

openalex() {
  local name=$1 query=$2 code
  code=$(curl -sS --max-time 30 -G "https://api.openalex.org/works" \
    --data-urlencode "search=$query" --data-urlencode "per-page=6" \
    --data-urlencode "mailto=neurotopo@example.invalid" \
    -o "$out/openalex_$name.json" -w "%{http_code}")
  echo -e "openalex\t$name\t$code\t$query" >> "$log"
  sleep 2
}

crossref() {
  local name=$1 query=$2 code
  code=$(curl -sS --max-time 30 -G "https://api.crossref.org/works" \
    --data-urlencode "query.bibliographic=$query" --data-urlencode "rows=6" \
    -o "$out/crossref_$name.json" -w "%{http_code}")
  echo -e "crossref\t$name\t$code\t$query" >> "$log"
  sleep 2
}

openalex orth1 'orthogonal recurrent neural network vanishing gradient long range'
openalex anti1 'antisymmetric recurrent neural network long-term dependency'
openalex iso1 'dynamical isometry trainability deep recurrent networks initialization'
openalex timescale1 'multiple timescales recurrent network long-term memory organization'
openalex ssm1 'structured state space model initialization long range dependency'
crossref orth1 'orthogonal recurrent neural networks gradient flow'
crossref anti1 'antisymmetric recurrent neural networks'
crossref iso1 'dynamical isometry and a mean field theory of RNNs'
crossref timescale1 'clockwork recurrent neural network timescales'
crossref ssm1 'efficiently modeling long sequences with structured state spaces'

echo "done; responses in $out"
