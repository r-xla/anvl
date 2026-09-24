# Environment of the paper image; enroot does not keep the Dockerfile's ENV lines
# for changes made inside a container, so the job scripts set them explicitly.
export CMDSTAN=/opt/cmdstan/cmdstan-2.40.0
export R_USER_CACHE_DIR=/opt/r-cache
export PJRT_INSTALL=0
export STAN_BUILD_DIR=/tmp/stan-build
export OMP_NUM_THREADS=1
