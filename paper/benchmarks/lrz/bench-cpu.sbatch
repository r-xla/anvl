#!/bin/bash
#SBATCH --job-name=anvl-paper-cpu
#SBATCH -p lrz-cpu
#SBATCH -q cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH -o %x-%j.out
set -euo pipefail
: "${IMAGE_DIR:?set IMAGE_DIR}"
: "${PAPER_DIR:?set PAPER_DIR}"
srun --container-image="$IMAGE_DIR/anvl-paper.sqsh" --container-remap-root \
  --container-mounts="$PAPER_DIR:/paper" \
  bash /paper/benchmarks/lrz/run-cpu.sh
