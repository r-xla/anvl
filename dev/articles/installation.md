# Installation

Currently, {anvl} is not available on CRAN, so you either have to
install it via [r-universe](https://r-xla.r-universe.dev/) or from
[GitHub](https://github.com/r-xla/anvl).

The system library required during runtime is `libprotobuf`. Source
installation requires a `C++20` compiler and `protoc` (protobuf
compiler).

You can install the latest release from r-universe (prebuilt binary).

``` r

install.packages("anvl", repos = c("https://cloud.r-project.org", "https://r-xla.r-universe.dev"))
```

The development version can be installed via:

``` r

pak::pak("r-xla/anvl")
```

Afterwards, you need to install additional dependencies via:

``` r

anvl::install_anvl()
```

If you do not run this, interactive use of {anvl} will ask you for
confirmation to download the additional dependencies. You can opt into
always downloading the additional required dependencies by configuring
the `PJRT_INSTALL` variable:

| Value | Effect |
|----|----|
| `1` | Always download, without asking. Use this in CI, scripts and Docker builds. |
| `0` | Never download; abort with instructions instead. |
| unset | Ask in an interactive session, abort in a non-interactive one. |

## CUDA Setup

The additional dependencies that {anvl} installs includes the
{pjrt.cuda} R package, which only requires a CUDA 13.3-compatible driver
to be installed.

When the {pjrt.cuda} package is not installed, the correct runtime
libraries need to be installed on the system and discoverable via
`LD_LIBRARY_PATH`. The specific versions of the CUDA runtime libraries
provided with {pjrt.cuda} are listed
[here](https://github.com/r-xla/pjrt.cuda/blob/main/inst/components.tsv).

To trouble-shoot the CUDA installation, run the following in a new R
session for maximum debug output.

``` r

Sys.setenv(PJRT_DEBUG = "1", TF_CPP_MIN_LOG_LEVEL = "0")
anvl::nv_scalar(1, device = "cuda")
```

Note that if another package (such as {torch}) is using different CUDA
versions, there might be some issues. In this case, use separate R
processes, e.g. via {mirai}.

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
