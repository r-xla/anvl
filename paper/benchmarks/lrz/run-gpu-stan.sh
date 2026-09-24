#!/bin/bash
# GPU runtime benchmark for Stan, which evaluates bernoulli_logit_glm() via OpenCL.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."
export ANVL_DEFAULT_DEVICE=cuda TORCH_DEVICE=cuda
# pjrt.cuda keeps the CUDA libraries in its R_user_dir() cache, which the base
# image populated under root's default cache directory.
if [ ! -e "$R_USER_CACHE_DIR/R/pjrt.cuda" ]; then
  src=$(find /root /opt /usr/local -maxdepth 6 -type d -name pjrt.cuda -path '*cache*' 2>/dev/null | grep -v "^$R_USER_CACHE_DIR" | head -1)
  mkdir -p "$R_USER_CACHE_DIR/R"
  ln -s "$src" "$R_USER_CACHE_DIR/R/pjrt.cuda"
fi
mkdir -p results
OUT=results/gpu-stan.csv
rm -f "$OUT" results/scaling-gpu-stan.csv
clinfo -l > results/clinfo.txt
STAN_COMPILE_OPENCL=1 Rscript bench-stan.R compile
Rscript bench-stan.R opencl "$OUT" single 1 opencl
Rscript bench-stan.R opencl results/scaling-gpu-stan.csv scaling 1 opencl
Rscript bench-stan.R opencl "$OUT" chains 8 opencl 64
