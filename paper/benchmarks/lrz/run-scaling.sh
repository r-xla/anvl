#!/bin/bash
# Strong scaling on the CPU: the same program on 1, 2, 4, 8 and 16 physical
# cores (one hardware thread per core). anvl runs one program, Stan uses
# reduce_sum() for a single chain and one process per chain for many chains.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."
export ANVL_DEFAULT_DEVICE=cpu
mkdir -p results
OUT=results/scaling.csv
rm -f "$OUT"
lscpu > results/lscpu.txt
taskset -cp $$ > results/affinity.txt
# First allowed hardware thread of each physical core.
ALLOWED=$(taskset -cp $$ | awk -F': ' '{print $2}' | tr ',' '\n' | awk -F- '{b = ($2 == "") ? $1 : $2; for (i = $1; i <= b; i++) print i}')
CPUS=($(lscpu -p=CPU,CORE,SOCKET | grep -v '^#' | awk -F, -v allowed="$(echo $ALLOWED)" '
  BEGIN { n = split(allowed, a, " "); for (i = 1; i <= n; i++) ok[a[i]] = 1 }
  ok[$1] && !seen[$3 "-" $2]++ { print $1 }'))
echo "physical cores: ${#CPUS[@]} (${CPUS[*]})" | tee results/cores.txt
Rscript bench-stan.R compile
taskset -c "${CPUS[0]}" Rscript bench-stan.R stan-1 "$OUT" scaling 1
for c in 1 2 4 8 16; do
  [ "$c" -le "${#CPUS[@]}" ] || break
  set=$(IFS=,; echo "${CPUS[*]:0:$c}")
  taskset -c "$set" Rscript bench-anvl.R "anvl-$c" "$OUT" scaling
  taskset -c "$set" Rscript bench-stan.R "stan_threads-$c" "$OUT" scaling "$c" threads
  TORCH_THREADS=$c taskset -c "$set" Rscript bench-torch.R "torch-$c" "$OUT" scaling
  TMB_THREADS=$c taskset -c "$set" Rscript bench-tmb.R "tmb-$c" "$OUT" scaling 1
  taskset -c "$set" Rscript bench-anvl.R "anvl-$c" "$OUT" scaling_chains
  taskset -c "$set" Rscript bench-stan.R "stan-$c" "$OUT" scaling_chains "$c"
done
