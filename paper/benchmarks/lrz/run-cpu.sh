#!/bin/bash
# CPU runtime benchmark. Runs inside the paper image with a 16-core allocation;
# the single-core configurations are pinned to the first allowed core.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."
export ANVL_DEFAULT_DEVICE=cpu
mkdir -p results
OUT=results/cpu.csv
rm -f "$OUT" results/overhead.csv
CORE=$(taskset -cp $$ | awk -F': ' '{print $2}' | cut -d, -f1 | cut -d- -f1)
lscpu > results/lscpu.txt
Rscript bench-stan.R compile
taskset -c "$CORE" Rscript bench-anvl.R cpu1 "$OUT" single
taskset -c "$CORE" Rscript bench-stan.R cpu1 "$OUT" single 1
Rscript bench-anvl.R cpu16 "$OUT" single
taskset -c "$CORE" Rscript bench-anvl.R cpu1 "$OUT" chains 64
taskset -c "$CORE" Rscript bench-stan.R cpu1 "$OUT" chains 1 64
Rscript bench-anvl.R cpu16 "$OUT" chains 256
Rscript bench-stan.R cpu16 "$OUT" chains 16 256
Rscript overhead.R results/overhead.csv
