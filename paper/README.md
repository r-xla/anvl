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
- `run-gpu.sh`: the same configurations for {anvl} on one GPU, with up to 4096 chains.

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
```

`build-image.sbatch` runs the steps of the Dockerfile (`docker/install.sh`) inside an
Enroot container of the base image. Results are written to `benchmarks/results/`, together
with the output of `lscpu` and `nvidia-smi`. Runtimes depend on the hardware and on other
jobs running on the same node, so they will not be reproduced exactly.
