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
# The CUDA base image ships NVIDIA's forward-compatibility libraries (libcuda, the
# PTX JIT compiler and NVVM of a newer driver), while the container runtime injects
# the host driver's OpenCL library. OpenCL crashes or fails to build kernels with
# the newer libraries, so the Stan runs use the host driver's own ones.
HOSTV=$(readlink /lib/x86_64-linux-gnu/libnvidia-opencl.so.1 | sed 's/libnvidia-opencl.so.//')
mkdir -p /tmp/host-driver
for l in libcuda libnvidia-ptxjitcompiler; do
  ln -sf "/lib/x86_64-linux-gnu/$l.so.$HOSTV" "/tmp/host-driver/$l.so.1"
done
ln -sf "/lib/x86_64-linux-gnu/libnvidia-nvvm.so.$HOSTV" /tmp/host-driver/libnvidia-nvvm.so.4
export LD_LIBRARY_PATH="/tmp/host-driver:${LD_LIBRARY_PATH:-}"
mkdir -p results
OUT=results/gpu-stan.csv
rm -f "$OUT" results/scaling-gpu-stan.csv
clinfo -l > results/clinfo.txt
STAN_COMPILE_OPENCL=1 Rscript bench-stan.R compile
Rscript bench-stan.R opencl "$OUT" single 1 opencl
Rscript bench-stan.R opencl results/scaling-gpu-stan.csv scaling 1 opencl
Rscript bench-stan.R opencl "$OUT" chains 8 opencl 64
