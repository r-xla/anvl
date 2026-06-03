# Installation

Currently, {anvl} is not available on CRAN, so you either have to
install it via [r-universe](https://r-xla.r-universe.dev/) or from
[GitHub](https://github.com/r-xla/anvl).

## System Dependencies

The system library required during runtime is `libprotobuf`. Source
installation requires a `C++20` compiler and `protoc` (protobuf
compiler).

## CPU Installation

You can install the latest release from GitHub

``` r

pak::pak("r-xla/anvl@*release")
```

You can install the latest release from r-universe (prebuilt binary).

``` r

install.packages("anvl",repos = c("https://cloud.r-project.org", "https://r-xla.r-universe.dev"))
```

To confirm that your CPU installation is working, run:

``` r

library(anvl)
nv_scalar(1, device = "cpu")
```

The development version can be installed via:

``` r

pak::pak("r-xla/anvl")
```

## GPU Installation

Running {anvl} with GPU support currently only works on Linux
(amd64/x86-64) or via WSL2 on Windows (experimental).

The recommended way to use CUDA there is to install the {cuda12.8} R
package, which only requires a compatible driver to be installed. You
can install it from GitHub or r-universe:

``` r

pak::pak("mlverse/cudatoolkit/cuda12.8")
install.packages("cuda12.8", repos = "https://mlverse.r-universe.dev")
```

When the {cuda12.8} package is not installed, the correct runtime
libraries need to be installed on the system and discoverable via
`LD_LIBRARY_PATH`. The specific versions of the CUDA runtime libraries
provided with {cuda12.8} are listed
[here](https://github.com/mlverse/cudatoolkit/blob/main/cuda12.8/inst/components.tsv).

**Troubleshooting**

To trouble-shoot the CUDA installation, run the following in a new R
session for maximum debug output.

``` r

Sys.setenv(PJRT_DEBUG = "1", TF_CPP_MIN_LOG_LEVEL = "0")
anvl::nv_scalar(1, device = "cuda")
```

Note that if another package is using a different cudatoolkit package
(e.g. when using {torch}), there might be some issues. In this case, use
separate R processes, e.g. via {mirai}.

## Docker

Prebuilt Docker images are available in
[r-xla/docker](https://github.com/r-xla/docker). This includes a CUDA
and CPU build for amd64/x86-64 architecture:

### Available Images

| Image       | Description                          |
|-------------|--------------------------------------|
| `anvl-cpu`  | CPU support, based on `rocker/r-ver` |
| `anvl-cuda` | GPU support with CUDA 12.8           |

Note that running the GPU container requires the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
to be installed on the host. Once installed (and the Docker daemon
restarted), pass `--gpus all` to `docker run` to expose the host GPUs to
the container:

``` bash
docker run --rm -it --gpus all ghcr.io/r-xla/anvl-cuda:latest R
```

You can verify that the GPU is visible inside the container by running
`nvidia-smi`, or from R:

``` r

anvl::nv_scalar(1, device = "cuda")
```

### Tags

Each image is available with two tags:

| Tag        | Description                                  |
|------------|----------------------------------------------|
| `:latest`  | Built from the `main` branch (rebuilt daily) |
| `:release` | Built from the latest release                |
