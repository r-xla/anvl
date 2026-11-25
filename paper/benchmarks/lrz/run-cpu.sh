#!/bin/bash
# CPU runtime benchmark. Runs inside the paper image with a 16-core allocation;
# the single-core configurations are pinned to the first allowed core.
# "cpu16" splits the chains over 16 PJRT CPU devices (one program per device),
# "cpu16prog" runs all chains as one program on 16 cores.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."
export ANVL_DEFAULT_DEVICE=cpu
mkdir -p results
OUT=results/cpu.csv
rm -f "$OUT" results/overhead.csv results/cpu-tmb.csv results/overhead-tmb.csv
CORE=$(taskset -cp $$ | awk -F': ' '{print $2}' | cut -d, -f1 | cut -d- -f1)
lscpu > results/lscpu.txt
Rscript bench-stan.R compile
taskset -c "$CORE" Rscript bench-anvl.R cpu1 "$OUT" single
taskset -c "$CORE" Rscript bench-stan.R cpu1 "$OUT" single 1
Rscript bench-anvl.R cpu16prog "$OUT" single
taskset -c "$CORE" Rscript bench-anvl.R cpu1 "$OUT" chains 64
taskset -c "$CORE" Rscript bench-stan.R cpu1 "$OUT" chains 1 64
Rscript bench-anvl.R cpu16prog "$OUT" chains 256
PJRT_CPU_DEVICE_COUNT=16 Rscript bench-anvl.R cpu16 "$OUT" chains 1024
Rscript bench-stan.R cpu16 "$OUT" chains 16 1024
TORCH_THREADS=1 taskset -c "$CORE" Rscript bench-torch.R cpu1 "$OUT" single
TORCH_THREADS=1 taskset -c "$CORE" Rscript bench-torch.R cpu1 "$OUT" chains 64
TORCH_THREADS=16 Rscript bench-torch.R cpu16 "$OUT" single
TORCH_THREADS=16 Rscript bench-torch.R cpu16 "$OUT" chains 256
TMB_THREADS=1 taskset -c "$CORE" Rscript bench-tmb.R cpu1 results/cpu-tmb.csv single 1
TMB_THREADS=1 taskset -c "$CORE" Rscript bench-tmb.R cpu1 results/cpu-tmb.csv chains 1 16
TMB_THREADS=1 Rscript bench-tmb.R cpu16 results/cpu-tmb.csv chains 16 256
Rscript overhead.R results/overhead.csv
