# Environment of the paper: R 4.5.3 with the packages of renv.lock, on NVIDIA's CUDA base image.
# The image also runs on machines without a GPU; anvl and torch then use the CPU.
#
# Build from the paper directory:
#   docker build -f envs/Dockerfile -t anvl-jss .
# Run from the paper directory, on the CPU or, with the NVIDIA Container Toolkit, on the GPU:
#   docker run -it --rm -v "$PWD":/mnt/data/paper anvl-jss
#   docker run -it --rm --gpus all -v "$PWD":/mnt/data/paper anvl-jss
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    TZ=Etc/UTC

# System requirements: compilers for the packages that are built from source (pjrt needs C++20 and
# protobuf, quickr a Fortran compiler), pandoc for rmarkdown, and OpenCL for Stan on the GPU, with
# NVIDIA's OpenCL implementation, which the container runtime injects together with the driver.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git locales tzdata \
      build-essential gfortran cmake pkg-config \
      libcurl4-openssl-dev libssl-dev libxml2-dev libfontconfig1-dev libfreetype6-dev \
      libharfbuzz-dev libfribidi-dev libpng-dev libjpeg-dev libtiff5-dev \
      libprotobuf-dev protobuf-compiler pandoc \
      ocl-icd-opencl-dev opencl-headers clinfo \
 && locale-gen en_US.UTF-8 \
 && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
 && rm -rf /var/lib/apt/lists/*

# R, in the version that renv.lock records, as Posit's build for Ubuntu 24.04.
ARG R_VERSION=4.5.3
RUN ARCH=$(dpkg --print-architecture) \
 && curl -fsSLO "https://cdn.posit.co/r/ubuntu-2404/pkgs/r-${R_VERSION}_1_${ARCH}.deb" \
 && apt-get update && apt-get install -y --no-install-recommends "./r-${R_VERSION}_1_${ARCH}.deb" \
 && rm "r-${R_VERSION}_1_${ARCH}.deb" && rm -rf /var/lib/apt/lists/* \
 && ln -s "/opt/R/${R_VERSION}/bin/R" /usr/local/bin/R \
 && ln -s "/opt/R/${R_VERSION}/bin/Rscript" /usr/local/bin/Rscript

# The packages of renv.lock, installed into the library of R, so that the container does not need
# renv at run time (.Rprofile does not activate renv in the container). The packages of r-xla, quickr,
# and cmdstanr are pinned to commits on GitHub; the others come from CRAN, as binaries of the Posit
# Public Package Manager where available.
COPY renv.lock /tmp/renv.lock
RUN R -q -e "install.packages('renv', repos = 'https://cloud.r-project.org')" \
 && R -q -e "options(renv.config.repos.override = c(CRAN = 'https://packagemanager.posit.co/cran/__linux__/noble/latest')); renv::restore(lockfile = '/tmp/renv.lock', library = .libPaths()[1L], prompt = FALSE)" \
 && rm /tmp/renv.lock

# LibTorch and Lantern for torch, built for CUDA 12.8; they also run on the CPU.
RUN CUDA=12.8 R -q -e "torch::install_torch()"

# CmdStan for the Stan benchmarks.
RUN R -q -e "cmdstanr::install_cmdstan(dir = '/opt/cmdstan', version = '2.40.0', cores = 4, quiet = TRUE)"
ENV CMDSTAN=/opt/cmdstan/cmdstan-2.40.0

# The PJRT plugins for the CPU and CUDA and the CUDA libraries (pjrt.cuda), downloaded at build time
# into a cache that does not depend on $HOME, so that the container never downloads at run time.
ENV R_USER_CACHE_DIR=/opt/r-cache
RUN mkdir -p /opt/r-cache \
 && PJRT_INSTALL=1 R -q -e "options(timeout = 600); anvl::install_anvl(cuda = TRUE)"
ENV PJRT_INSTALL=0

# Check the versions of the paper and that anvl runs.
RUN R -q -e "stopifnot(packageVersion('anvl') == '0.6.0', packageVersion('pjrt') == '0.7.0', packageVersion('stablehlo') == '0.6.0', packageVersion('torch') == '0.17.0'); library(anvl); print(nv_array(1:3) + 1L)"

WORKDIR /mnt/data/paper
CMD ["bash"]
