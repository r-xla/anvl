#!/usr/bin/env bash
## ---------------------------------------------------------------------------
## Pipes anvl's stream (gen.R) into a battery and files the report under
## results/.
##
##   ./run.sh smallcrush                          seed 42, ui32 words
##   ./run.sh crush --seed 7
##   ./run.sh bigcrush --streams 64 --chunk 1     seeds 42..105 interleaved
##   ./run.sh smallcrush --mode unif              nv_runif() f64 doubles
##   TESTU01_OPTS=-r ./run.sh smallcrush          bit-reversed words
##   ./run.sh practrand                           PractRand up to 1 GB
##   PRACTRAND_MAX=64GB ./run.sh practrand --streams 16
##
## The first argument is the battery (smallcrush | crush | bigcrush |
## practrand); the rest go to gen.R. Writes results/<battery>-<label>.txt (the
## battery's report), .tsv (TestU01's p-values) and .log (stderr, timings).
## ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
battery="${1:?usage: run.sh smallcrush|crush|bigcrush|practrand [gen.R options]}"
shift

label="$(printf '%s' "$*" | sed 's/--//g; s/ /_/g')"
label="${label:-default}${TESTU01_OPTS:+_${TESTU01_OPTS//[ -]/}}"
out="$HERE/results/$battery-$label-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$HERE/results"

opts=(${TESTU01_OPTS:-})
[[ " $* " == *" --mode unif "* ]] && opts+=(-d)

start=$(date +%s)
case "$battery" in
  smallcrush | crush | bigcrush)
    set +e
    Rscript "$HERE/gen.R" "$@" 2>>"$out.log" |
      "$HERE/testu01" "${opts[@]}" -n "anvl $* ${TESTU01_OPTS:-}" -p "$out.tsv" "$battery" \
        >"$out.txt" 2>>"$out.log"
    status=${PIPESTATUS[1]}
    set -e
    ;;
  practrand)
    if [[ " $* " == *" --mode unif "* ]]; then
      echo "PractRand reads raw bits; a double's exponent bits are not random" >&2
      exit 2
    fi
    set +e
    word=stdin32
    [[ " $* " == *" --dtype ui64 "* ]] && word=stdin64
    Rscript "$HERE/gen.R" "$@" 2>>"$out.log" |
      "$HERE/RNG_test" "$word" -tlmax "${PRACTRAND_MAX:-1GB}" -multithreaded \
        >"$out.txt" 2>>"$out.log"
    status=${PIPESTATUS[1]}
    set -e
    ;;
  *) echo "unknown battery: $battery" >&2; exit 2 ;;
esac

echo "wall time: $(( $(date +%s) - start )) s, exit status: $status" >>"$out.log"
cat "$out.log" >&2
echo "report: $out.txt" >&2
exit "$status"
