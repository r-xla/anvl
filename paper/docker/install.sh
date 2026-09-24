#!/usr/bin/env bash
# Installs everything the paper needs on top of the anvl-cuda image.
# Used by the Dockerfile and, on the LRZ AI Systems, run inside an Enroot
# container created from the same base image (see ../benchmarks/lrz/).
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends gfortran libcurl4-openssl-dev pandoc
rm -rf /var/lib/apt/lists/*

# Pinned versions of the r-xla packages and quickr.
R -q -e "remotes::install_github(c(
  'r-xla/xlamisc@b2249754ef1c9febcf577e81b4684d250395f1b6',
  'r-xla/tengen@31357e3d71f6a8ee66fff2cd4e1a1ec1ac68076f',
  'r-xla/stablehlo@302ba5498e5dd1ce551debcec69256cc2c746fbe',
  'r-xla/pjrt@89256c0d80a278339d000395b7f91000b43c02db',
  't-kalinowski/quickr@edd403766e612ef4867ad817ead3dc366dc15d77',
  'r-xla/anvl@9d26362b8df4b526df98135fa30005e83ece5f53'
), upgrade = 'never', force = TRUE)"

R -q -e "install.packages(c('withr', 'ggplot2', 'rmarkdown', 'rjtools', 'knitr', 'safetensors'))"
R -q -e "install.packages('cmdstanr', repos = c('https://stan-dev.r-universe.dev', getOption('repos')))"
mkdir -p /opt/cmdstan
R -q -e "cmdstanr::install_cmdstan(dir = '/opt/cmdstan', version = '2.40.0', cores = 8, quiet = TRUE)"

# Download the PJRT plugins at build time, into a cache that does not depend on $HOME.
export R_USER_CACHE_DIR=/opt/r-cache
PJRT_INSTALL=1 R -q -e "options(timeout = 600); pjrt:::plugin_path('cpu'); pjrt:::plugin_path('cuda')"
