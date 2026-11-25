# Image used to produce all results of the paper.
# Build from the repository root:  docker build -f paper/docker/Dockerfile -t anvl-paper paper/docker
# The base image is pinned by digest; install.sh pins the r-xla packages by commit.
FROM sebffischer/anvl-cuda@sha256:0b04fd332b411604c1725da0fe1e9b83f6e71a9fb77687fc51ce498ede9370a8

COPY install.sh /tmp/install.sh
RUN bash /tmp/install.sh && rm /tmp/install.sh

ENV CMDSTAN=/opt/cmdstan/cmdstan-2.40.0
ENV PJRT_INSTALL=0
ENV R_USER_CACHE_DIR=/opt/r-cache
WORKDIR /paper
CMD ["bash"]
