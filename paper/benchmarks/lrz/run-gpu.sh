#!/bin/bash
# GPU runtime benchmark. Runs inside the paper image with one GPU.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."
export ANVL_DEFAULT_DEVICE=cuda
mkdir -p results
OUT=results/gpu.csv
rm -f "$OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > results/gpu.txt
Rscript bench-anvl.R gpu "$OUT" single
Rscript bench-anvl.R gpu "$OUT" chains
