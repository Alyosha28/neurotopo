#!/usr/bin/env bash
# DIR-0004 prior-art fetch: precision of chaotic trajectory tracking, unstable-subspace
# assimilation / observer design, reservoir prediction horizon, finite-precision chaos.
#
# The design's section 9 records that theorem-level sources for the Lyapunov precision
# bound were NOT retrieved in the DIR-0003 seven-region audit (region G covered capacity
# and memory, not Lyapunov accuracy). This script closes that bookkeeping gap.
#
# Read-only w.r.t. the repository except for its own output directory. Raw responses are
# kept verbatim so that every title quoted later can be re-checked byte for byte.
set -uo pipefail

out=research/prior_art/dir0004_raw
mkdir -p "$out"
log="$out/fetch_log.txt"
: > "$log"

arxiv() {
  local name=$1 query=$2 code
  code=$(curl -sS --max-time 45 -G "https://export.arxiv.org/api/query" \
    --data-urlencode "search_query=$query" \
    --data-urlencode "start=0" --data-urlencode "max_results=8" \
    -o "$out/arxiv_$name.xml" -w "%{http_code}")
  echo "arxiv	$name	$code	$query" >> "$log"
  sleep 3
}

openalex() {
  local name=$1 query=$2 code
  code=$(curl -sS --max-time 45 -G "https://api.openalex.org/works" \
    --data-urlencode "search=$query" --data-urlencode "per-page=8" \
    --data-urlencode "mailto=neurotopo@example.invalid" \
    -o "$out/openalex_$name.json" -w "%{http_code}")
  echo "openalex	$name	$code	$query" >> "$log"
  sleep 2
}

crossref() {
  local name=$1 query=$2 code
  code=$(curl -sS --max-time 45 -G "https://api.crossref.org/works" \
    --data-urlencode "query.bibliographic=$query" --data-urlencode "rows=8" \
    -o "$out/crossref_$name.json" -w "%{http_code}")
  echo "crossref	$name	$code	$query" >> "$log"
  sleep 2
}

# --- unstable-subspace data assimilation / observer design for chaotic systems ---------
arxiv aus1 'all:"assimilation in the unstable subspace"'
arxiv aus2 'all:"unstable subspace" AND all:"data assimilation"'
arxiv aus3 'all:"Lyapunov vectors" AND all:"data assimilation"'
openalex aus1 'assimilation in the unstable subspace'
openalex aus2 'unstable subspace observer chaotic system'
crossref aus1 'assimilation of observations in the unstable subspace'

# --- precision / horizon limits of chaotic prediction ---------------------------------
arxiv prec1 'all:"finite precision" AND all:"chaotic" AND all:"prediction horizon"'
arxiv prec2 'all:"predictability" AND all:"Lyapunov exponent" AND all:"error growth"'
openalex prec1 'finite precision chaotic prediction horizon bits'
crossref prec1 'predictability time Lyapunov exponent error doubling'

# --- reservoir computing / RNN on chaotic systems -------------------------------------
arxiv rc1 'all:"reservoir computing" AND all:"chaotic" AND all:"prediction"'
arxiv rc2 'all:"echo state network" AND all:"Lyapunov" AND all:"prediction"'
openalex rc1 'reservoir computing chaotic attractor prediction horizon'
crossref rc1 'reservoir computing prediction chaotic systems Lyapunov time'

# --- learned latent dynamics with unstable modes / structured state -------------------
arxiv lat1 'all:"latent dynamics" AND all:"unstable" AND all:"state space model"'
arxiv lat2 'all:"Koopman" AND all:"unstable subspace" AND all:"control"'
openalex lat1 'structured state space model unstable dynamics long horizon tracking'
crossref lat1 'model predictive control unstable subspace tracking horizon'

echo "done; responses in $out"
