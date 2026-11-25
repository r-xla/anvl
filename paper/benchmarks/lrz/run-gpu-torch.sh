#!/bin/bash
# GPU runtime benchmark for torch. Runs inside the paper image with one GPU.
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
rm -f results/gpu-torch.csv results/scaling-gpu-torch.csv
Rscript bench-torch.R gpu results/gpu-torch.csv single
Rscript bench-torch.R gpu results/gpu-torch.csv chains
Rscript bench-torch.R gpu results/scaling-gpu-torch.csv scaling
