
<!-- README.md is generated from README.Rmd. Please edit that file -->

# anvl <img src="man/figures/logo.png" align="right" width = "120" />

Package website: [release](https://r-xla.github.io/anvl/) \|
[dev](https://r-xla.github.io/anvl/dev/)

<!-- badges: start -->

![R-CMD-check](https://github.com/r-xla/anvl/actions/workflows/R-CMD-check.yaml/badge.svg)
[![CRAN
status](https://www.r-pkg.org/badges/version/anvl)](https://CRAN.R-project.org/package=anvl)
[![codecov](https://codecov.io/gh/r-xla/anvl/branch/main/graph/badge.svg)](https://codecov.io/gh/r-xla/anvl)
[![r-universe](https://r-xla.r-universe.dev/badges/anvl)](https://r-xla.r-universe.dev/anvl)
![CUDA 13.3](https://img.shields.io/badge/CUDA-13.3-green.svg)
<!-- badges: end -->

Accelerated array computing and code transformations for R, allowing you
to run numerical programs at the speed of light. The package supports
JIT compilation for very fast execution and reverse-mode automatic
differentiation. Programs can run on CPU and NVIDIA GPU.

## Installation

``` r
install.packages("anvl", repos = c("https://r-xla.r-universe.dev", getOption("repos")))
```

The PJRT plugins anvl runs on are downloaded separately, on demand. To
download them right away instead of when they are first needed:

``` r
anvl::install_anvl()
```

CUDA support is only available on Linux (x86_64 and ARM). There, this
also installs the CUDA plugin when an NVIDIA GPU is detected; pass
`cuda = TRUE` to install it regardless.

See the [installation
guide](https://r-xla.github.io/anvl/articles/installation.html) for more
details, including prebuilt Docker images.

## Why anvl

anvl makes numerical R code run fast on CPUs and GPUs, and computes
gradients of your functions automatically. It aspires to be for R what
JAX is for Python.

There are three core ideas:

- **Compilation.** {anvl} converts R functions into an optimized program
  via XLA – the same compiler that powers JAX and TensorFlow. Due to the
  compilation, resulting programs can be faster compared to implementing
  them in [{torch}](https://torch.mlverse.org).
- **Function transformation.** Programmatically derive new functions
  from existing ones. Currently the only available transformation is
  reverse-mode automatic differentiation via `gradient()`, which returns
  the derivative of a function as another R function.
- **Hardware portability.** The same code runs on CPU or GPU.

Moreover, the package is designed to be extensible. As the package is
written in R, new primitives and transformations can be added without
needing a lower-level language.

## Usage

We define an R function operating on `AnvlArray`s – the primary data
type of {anvl}. It can be executed in either *eager* mode (each
operation is performed immediately) or *jit* mode (the whole function is
compiled into a single executable via `jit()`).

``` r
library(anvl)
f <- function(a, b, x) {
  a * x + b
}

a <- nv_scalar(1.0, "f32")
b <- nv_scalar(-2.0, "f32")
x <- nv_scalar(3.0, "f32")

# Eager mode
f(a, b, x)
#> AnvlArray
#>  1
#> [ CPUf32{} ]

# JIT mode
f_jit <- jit(f)
f_jit(a, b, x)
#> AnvlArray
#>  1
#> [ CPUf32{} ]
```

Through automatic differentiation, we can also obtain the gradient of
the above function.

``` r
g_jit <- jit(gradient(f, wrt = c("a", "b")))
g_jit(a, b, x)
#> $a
#> AnvlArray
#>  3
#> [ CPUf32{} ] 
#> 
#> $b
#> AnvlArray
#>  1
#> [ CPUf32{} ]
```

For more complex examples, such as implementing a Gaussian Process, see
the package website.

## Platform Support

| OS      | Architecture | CPU |    CUDA     |
|---------|--------------|:---:|:-----------:|
| Linux   | x86_64       |  ✓  |      ✓      |
| Linux   | arm64        |  ✓  |      ✓      |
| Windows | x86_64       |  ✓  | ◐ WSL2 only |
| Windows | arm64        |  ✗  |      ✗      |
| macOS   | x86_64       |  ✓  |      ✗      |
| macOS   | arm64        |  ✓  |      ✗      |

✓ fully supported  ·  ◐ limited support  ·  ✗ not supported

## Acknowledgments

- This work is supported by [MaRDI](https://www.mardi4nfdi.de).
- The design of this package was inspired by and borrows from:
  - JAX, especially the [autodidax
    tutorial](https://docs.jax.dev/en/latest/autodidax.html).
  - The [microjax](https://github.com/joey00072/microjax) project.
- For JIT compilation, we leverage the [OpenXLA](https://openxla.org/)
  project.
