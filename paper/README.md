# anvl: Composable Code Transformations for Array Programs in R

Source of the R Journal article on {anvl}.

| File | Content |
|---|---|
| `paper.Rmd` | Article source (rendered with `rjtools::rjournal_pdf_article()`) |
| `references.bib`, `preamble.tex`, `RJournal.sty` | Bibliography, LaTeX macros, journal style |
| `docker/` | `Dockerfile` and `install.sh` of the image in which all results were produced |
| `benchmarks/hmc-anvl.R` | HMC sampler; its sections are read into the article via `knitr::read_chunk()` |
| `benchmarks/logistic.stan` | The same model in Stan |
| `benchmarks/bench-*.R`, `overhead.R` | Runtime benchmark and compilation overhead |
| `benchmarks/lrz/` | Slurm scripts used to run the benchmark on the LRZ AI Systems |
| `benchmarks/results/` | Raw benchmark results used in the article |
| `benchmarks/plots.R` | Figures and tables of the runtime evaluation |

## Software environment

The image is based on `sebffischer/anvl-cuda`, pinned by digest, and `docker/install.sh`
installs the r-xla packages and {quickr} at fixed commits, CmdStan 2.40.0, and the R
packages needed to render the article. It contains both the CPU and the CUDA PJRT plugin,
so the same image is used for all experiments.

```bash
cd paper
docker build -t anvl-paper docker
```

## Rendering the article

Inside the image (or any environment with the same packages and a LaTeX installation):

```bash
docker run --rm -it -v "$PWD:/paper" anvl-paper bash
cd /paper
make pdf
```

`make pdf` knits `paper.Rmd`, which runs all code shown in the article and reads the
benchmark results from `benchmarks/results/`, and then compiles `RJwrapper.tex` to
`RJwrapper.pdf`. The examples run on the CPU.

## Runtime benchmark

The benchmark compares the HMC sampler to CmdStan's static HMC sampler. It consists of

- `run-cpu.sh`: one core and 16 cores, single chain for n = 10^3, 10^4, 10^5 and
  1 to 256 chains for n = 10^4 (1 to 64 on one core), plus `overhead.R`;
- `run-gpu.sh`: the same configurations for {anvl} on one GPU, with up to 4096 chains, and the
  strong-scaling configuration on the GPU;
- `run-scaling.sh`: strong scaling of a single chain (n = 10^5, p = 100) on 1 to 16 physical
  cores, for {anvl} and for Stan with `reduce_sum()` (`logistic-threads.stan`), plus 64 chains
  as a single {anvl} program versus one Stan process per chain.

On 16 cores, the multi-chain runs of {anvl} split the chains over 16 PJRT CPU devices
(`PJRT_CPU_DEVICE_COUNT=16`); `cpu16prog` is the same experiment as a single program.

Locally, with Docker:

```bash
make bench-cpu   # restrict the container to 16 cores, e.g. via --cpuset-cpus, to match the article
make bench-gpu   # requires the NVIDIA Container Toolkit
```

For the article, we ran the scripts on the LRZ AI Systems, where Enroot imports the
Docker image and Slurm runs the jobs via Pyxis:

```bash
export IMAGE_DIR=<directory for the squashfs image> PAPER_DIR=<checkout>/paper
sbatch --export=ALL,IMAGE_DIR,PAPER_DIR benchmarks/lrz/build-image.sbatch
sbatch --export=ALL,IMAGE_DIR,PAPER_DIR benchmarks/lrz/bench-cpu.sbatch
sbatch --export=ALL,IMAGE_DIR,PAPER_DIR benchmarks/lrz/bench-gpu.sbatch
sbatch --export=ALL,IMAGE_DIR,PAPER_DIR benchmarks/lrz/bench-scaling.sbatch
```

The `lrz-cpu` partition was congested when we ran the benchmark, so we submitted the image
build and the CPU jobs to `mcml-hgx-a100-80x4-mig` instead (overriding `-p`, `-q` and adding
`--gres=gpu:1` on the `sbatch` command line); the CPU benchmarks do not use the GPU slice.

`build-image.sbatch` runs the steps of the Dockerfile (`docker/install.sh`) inside an
Enroot container of the base image. Results are written to `benchmarks/results/`, together
with the output of `lscpu` and `nvidia-smi`. Runtimes depend on the hardware and on other
jobs running on the same node, so they will not be reproduced exactly.
