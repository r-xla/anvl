#!/bin/bash
#SBATCH --job-name=anvl-paper-gpu
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH -q mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -o %x-%j.out
set -euo pipefail
: "${IMAGE_DIR:?set IMAGE_DIR}"
: "${PAPER_DIR:?set PAPER_DIR}"
srun --container-image="$IMAGE_DIR/anvl-paper.sqsh" --container-remap-root \
  --container-mounts="$PAPER_DIR:/paper" \
  --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES \
  bash /paper/benchmarks/lrz/run-gpu.sh
