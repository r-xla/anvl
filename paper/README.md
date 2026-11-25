# anvl: Composable Code Transformations for Array Programs in R

Source of the JSS article on {anvl}.

| File | Content |
|---|---|
| `main.tex`, `jss.cls`, `jss.bst` | Source of the JSS article (`make`) |
| `extract.R`, `paper_code.R` | Extracts the R code of `main.tex` into the replication script `paper_code.R` |
| `references.bib` | Bibliography |
| `figures/overview.tex` | TikZ source of the overview figure (`make figures/overview.pdf`) |
| `figures/style.tex` | Style shared by the TikZ figures: fonts, colors, boxes, numbered notes |
| `figures/runtime.R` | Writes the runtime figures of `main.tex` in the same font and colors (`Rscript figures/runtime.R`) |
| `figures/hmc.R` | Runs the HMC sampler of `main.tex` and plots its draws (`figures/hmc.pdf`, needs anvl 0.6.0; `Rscript figures/hmc.R`) |
| `renv.lock`, `renv/`, `.Rprofile`, `.renvignore` | R package versions of the paper (anvl 0.6.0) for {renv}; `.Rprofile` activates renv outside the container |
| `envs/Dockerfile` | Image with R 4.5.3 and the packages of `renv.lock` on NVIDIA's CUDA base image, for CPUs and GPUs |
| `docker/` | `Dockerfile` and `install.sh` of the image in which the runtime benchmark was run (a development version of anvl from before 0.6.0) |
| `benchmarks/hmc-anvl.R` | HMC sampler of the use case, as benchmarked |
| `benchmarks/hmc-torch.R` | The same sampler with torch (autograd, eager execution) |
| `benchmarks/logistic.stan`, `logistic-threads.stan` | The model in Stan, and with `reduce_sum()` for multithreading |
| `benchmarks/bench-*.R`, `overhead.R` | Runtime benchmark and compilation overhead |
| `benchmarks/lrz/` | Slurm scripts used to run the benchmark on the LRZ AI Systems |
| `benchmarks/results/` | Raw benchmark results used in the article |
| `benchmarks/plots.R` | Figures and tables of the runtime evaluation |

## Reproducing the results

The appendix "Computational details" of the article describes the hardware and the software
versions.

### Computational environment

To reproduce the results, you can either use the Docker image of `envs/Dockerfile` or recreate
the renv environment that `renv.lock` describes. `renv.lock` records the versions of all R
packages that the code of `main.tex`, the figures, and the benchmark scripts use: anvl 0.6.0,
stablehlo 0.6.0, and pjrt 0.7.0, which are pinned to their commits on GitHub, and the other
packages from CRAN.

To work with the renv environment, go into the `paper` directory and start an interactive R
session, which will bootstrap the `renv` package. Then, restore the environment by running:

```r
renv::restore()
```

It will ask you whether you want to proceed installing the missing packages, which you have to
confirm. Afterwards, download the PJRT plugins (and, on Linux machines with an NVIDIA GPU, the
CUDA plugin and libraries), LibTorch for torch, and, for the Stan benchmarks, CmdStan:

```r
anvl::install_anvl()
torch::install_torch()
cmdstanr::install_cmdstan(version = "2.40.0")
```

The `Dockerfile` in `envs/` installs the same packages into an image based on NVIDIA's CUDA base
image (`nvidia/cuda`), together with LibTorch, CmdStan 2.40.0, and the PJRT plugins for the CPU
and CUDA. The image also runs on machines without a GPU. Build it from the `paper` directory:

```bash
make image   # or: docker build -f envs/Dockerfile -t anvl-jss .
```

To start the container on the CPU, run:

```bash
# from the paper directory
docker run -it --rm -v "$PWD":/mnt/data/paper anvl-jss
# the working directory of the container is /mnt/data/paper
```

To use the GPU, start it with the command below, which requires the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker run -it --rm --gpus all -v "$PWD":/mnt/data/paper anvl-jss
```

Note that the `.Rprofile` file in `paper` ensures that R programs run from the `paper` directory
use the renv environment, unless they run in the container, where the packages are installed into
the library of R and renv is not used. It detects the container via the file `/.dockerenv` or via
the mount point `/mnt/data/paper`. When the image is run with something other than Docker (e.g.,
Podman, Apptainer, or Enroot) and the `paper` directory is not mounted at `/mnt/data/paper`,
neither heuristic applies, so renv is activated and the packages installed in the image are not
found. In that case, adjust the condition in `.Rprofile` (or start R with `--no-init-file`).

### Running the code of the article

`paper_code.R` contains all R code shown in `main.tex`, in the order of the article. Render it
from the `paper` directory, which runs the code and shows its output next to it:

```r
knitr::spin("paper_code.R")
```

The examples run on the CPU; on a machine with an NVIDIA GPU, the arrays of the HMC sampler can
be placed on the GPU via `nv_array(X, device = "cuda")`. One chunk shows an error on purpose
(the promotion rules refuse to round a double-precision input), which `knitr::spin()` displays
and continues.

## Software environment of the runtime benchmark

The image is based on `sebffischer/anvl-cuda`, pinned by digest, and `docker/install.sh`
installs the r-xla packages and {quickr} at fixed commits, CmdStan 2.40.0, torch with the CUDA build of LibTorch,
the OpenCL loader for Stan's GPU support, and the R
packages needed to render the article. It contains both the CPU and the CUDA PJRT plugin,
so the same image is used for all experiments.

```bash
cd paper
docker build -t anvl-paper docker
```

## Building the article

`make` builds the JSS article `main.pdf` (with `latexmk`) and the replication script
`paper_code.R`, which contains the R code of `main.tex` (render it with
`knitr::spin("paper_code.R")`). The figures are rebuilt when their sources change:
`figures/hmc.R` runs the sampler with anvl, `figures/runtime.R` reads `benchmarks/results/`.
`make image` builds the image of `envs/Dockerfile`, and `make shell` opens a shell in it with
this directory mounted.

## Runtime benchmark

The benchmark compares the HMC sampler to CmdStan's static HMC sampler (on the CPU, with
`reduce_sum()` threads, and with OpenCL on the GPU) and to the same sampler written with torch.
It consists of

- `run-cpu.sh`: one core and 16 cores, single chain for n = 10^3, 10^4, 10^5 and
  1 to 256 chains for n = 10^4 (1 to 64 on one core), plus `overhead.R`;
- `run-gpu.sh`, `run-gpu-torch.sh`, `run-gpu-stan.sh`: the same configurations on one GPU for
  {anvl}, torch, and Stan with OpenCL, including the strong-scaling configuration. On the LRZ
  nodes, torch did not finish 1024 chains and Stan (OpenCL) did not finish 64 chains within the
  one-hour limit. `run-gpu-stan.sh` extracts NVIDIA's user-space libraries of the host driver's
  version from the driver installer (the nodes lack a matching `libnvidia-gpucomp`, and the CUDA
  base image ships newer forward-compatibility libraries), and enables `cl_khr_fp64` in Stan's
  OpenCL kernels, which NVIDIA's compiler requires for double precision;
- `run-scaling.sh`: strong scaling of a single chain (n = 10^5, p = 100) on 1 to 16 physical
  cores, for {anvl} and for Stan with `reduce_sum()` (`logistic-threads.stan`), plus 64 chains
  as a single {anvl} program versus one Stan process per chain.

On 16 cores, the multi-chain runs of {anvl} split the chains over 16 PJRT CPU devices
(`PJRT_CPU_DEVICE_COUNT=16`); `cpu16prog` is the same experiment as a single program.

Locally, with Docker, in the image of `docker/`:

```bash
docker build -t anvl-paper docker
# restrict the container to 16 cores, e.g. via --cpuset-cpus, to match the article
docker run --rm -v "$PWD:/paper" anvl-paper bash /paper/benchmarks/lrz/run-cpu.sh
# requires the NVIDIA Container Toolkit
docker run --rm --gpus all -v "$PWD:/paper" anvl-paper bash /paper/benchmarks/lrz/run-gpu.sh
```

For the article, we ran the scripts on the LRZ AI Systems, where Enroot imports the
Docker image and Slurm runs the jobs via Pyxis:

```bash
export IMAGE_DIR=<directory for the squashfs image> PAPER_DIR=<checkout>/paper
sbatch --export=ALL,IMAGE_DIR,PAPER_DIR benchmarks/lrz/build-image.sbatch
sbatch --export=ALL,IMAGE_DIR,PAPER_DIR benchmarks/lrz/bench-cpu.sbatch
sbatch --export=ALL,IMAGE_DIR,PAPER_DIR benchmarks/lrz/bench-gpu.sbatch
sbatch --export=ALL,IMAGE_DIR,PAPER_DIR benchmarks/lrz/bench-gpu-torch.sbatch
sbatch --export=ALL,IMAGE_DIR,PAPER_DIR benchmarks/lrz/bench-gpu-stan.sbatch
sbatch --export=ALL,IMAGE_DIR,PAPER_DIR benchmarks/lrz/bench-scaling.sbatch
```

The `lrz-cpu` partition was congested when we ran the benchmark, so we submitted the image
build and the CPU jobs to `mcml-hgx-a100-80x4-mig` instead (overriding `-p`, `-q` and adding
`--gres=gpu:1` on the `sbatch` command line); the CPU benchmarks do not use the GPU slice.

`build-image.sbatch` runs the steps of the Dockerfile (`docker/install.sh`) inside an
Enroot container of the base image. Results are written to `benchmarks/results/`, together
with the output of `lscpu` and `nvidia-smi`. Runtimes depend on the hardware and on other
jobs running on the same node, so they will not be reproduced exactly.
