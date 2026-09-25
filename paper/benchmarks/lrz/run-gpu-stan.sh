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
# OpenCL needs NVIDIA's user-space libraries in exactly the version of the host's
# kernel driver. The container runtime injects only some of them, and the CUDA base
# image ships newer forward-compatibility versions of others, so we extract the
# complete set from the driver installer of the host's version.
HOSTV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
NV=/tmp/nvidia-$HOSTV
if [ ! -d "$NV" ]; then
  curl -fsSL -o /tmp/nvidia.run \
    "https://download.nvidia.com/XFree86/Linux-x86_64/$HOSTV/NVIDIA-Linux-x86_64-$HOSTV.run"
  sh /tmp/nvidia.run --extract-only --target "$NV" > /dev/null
  rm /tmp/nvidia.run
fi
ln -sf "libnvidia-opencl.so.$HOSTV" "$NV/libnvidia-opencl.so.1"
ln -sf "libnvidia-ptxjitcompiler.so.$HOSTV" "$NV/libnvidia-ptxjitcompiler.so.1"
ln -sf "libnvidia-nvvm.so.$HOSTV" "$NV/libnvidia-nvvm.so.4"
ln -sf "libcuda.so.$HOSTV" "$NV/libcuda.so.1"
echo "$NV/libnvidia-opencl.so.1" > "$NV/nvidia.icd"
export OCL_ICD_VENDORS="$NV/nvidia.icd"
export LD_LIBRARY_PATH="$NV:${LD_LIBRARY_PATH:-}"
# Stan's OpenCL kernels use double precision without enabling cl_khr_fp64, which
# NVIDIA's OpenCL compiler rejects (the build fails or crashes). We prepend the
# pragma to every kernel Stan compiles; this does not change the computations.
KCL="$CMDSTAN/stan/lib/stan_math/stan/math/opencl/kernel_cl.hpp"
if ! grep -q sources_fp64 "$KCL"; then
  sed -i 's|^  cl::Program program(opencl_context.context(), sources);|  std::vector<std::string> sources_fp64{"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\\n"};\n  sources_fp64.insert(sources_fp64.end(), sources.begin(), sources.end());\n  cl::Program program(opencl_context.context(), sources_fp64);|' "$KCL"
fi
grep -q sources_fp64 "$KCL"
mkdir -p results
OUT=results/gpu-stan.csv
rm -f "$OUT" results/scaling-gpu-stan.csv
clinfo -l > results/clinfo.txt
STAN_COMPILE_OPENCL=1 Rscript bench-stan.R compile
Rscript bench-stan.R opencl "$OUT" single 1 opencl
Rscript bench-stan.R opencl results/scaling-gpu-stan.csv scaling 1 opencl
Rscript bench-stan.R opencl "$OUT" chains 8 opencl 64
