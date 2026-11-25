#!/bin/bash
# Builds the paper image on the LRZ AI Systems: imports the anvl-cuda base image
# and runs ../../docker/install.sh inside it, i.e. the steps of the Dockerfile.
#SBATCH --job-name=anvl-paper-image
#SBATCH -p lrz-cpu
#SBATCH -q cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o %x-%j.out
set -euo pipefail
: "${IMAGE_DIR:?set IMAGE_DIR}"
: "${PAPER_DIR:?set PAPER_DIR}"
cd "$IMAGE_DIR"
rm -f base.sqsh anvl-paper.sqsh
enroot import -o base.sqsh "docker://sebffischer/anvl-cuda:latest"
enroot create --force --name anvl-paper-build base.sqsh
enroot start --root --rw --mount "$PAPER_DIR/docker:/build" anvl-paper-build bash /build/install.sh
enroot export -o anvl-paper.sqsh anvl-paper-build
enroot remove -f anvl-paper-build
rm -f base.sqsh
