# anvl

Package website: [release](https://r-xla.github.io/anvl/) \|
[dev](https://r-xla.github.io/anvl/dev/)

Accelerated array computing and code transformations for R, allowing you
to run numerical programs at the speed of light. The package supports
JIT compilation for very fast execution and reverse-mode automatic
differentiation. Programs can run on CPU and NVIDIA GPU.

## Installation

``` r

install.packages("anvl", repos = c("https://r-xla.r-universe.dev", getOption("repos")))
```

Afterwards, install the additional dependencies as follows:

``` r

anvl::install_anvl()
```

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
  reverse-mode automatic differentiation via
  [`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md),
  which returns the derivative of a function as another R function.
- **Hardware portability.** The same code runs on CPU or GPU.

Moreover, the package is designed to be extensible. As the package is
written in R, new primitives and transformations can be added without
needing a lower-level language.

## Usage

We define an R function operating on `AnvlArray`s – the primary data
type of {anvl}. It can be executed in either *eager* mode (each
operation is performed immediately) or *jit* mode (the whole function is
compiled into a single executable via
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)).

``` r

library(anvl)
f <- function(a, b, x) {
  a * x + b
}

a <- nv_scalar(1, "f32")
b <- nv_scalar(2, "f32")
x <- nv_scalar(3, "f32")

# Eager mode
f(a, b, x)
#> AnvlArray
#>  5
#> [ CPUf32{} ]

# JIT mode
f_jit <- jit(f)
f_jit(a, b, x)
#> AnvlArray
#>  5
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

| OS      | Architecture | CPU | CUDA |
|---------|--------------|:---:|:----:|
| Linux   | x86_64       |  ✓  |  ✓   |
| Linux   | arm64        |  ✓  |  ✓   |
| Windows | x86_64       |  ✓  | WSL2 |
| Windows | arm64        |  ✗  |  ✗   |
| macOS   | x86_64       |  ✓  |  ✗   |
| macOS   | arm64        |  ✓  |  ✗   |

✓ fully supported  ·  ✗ not supported

## Acknowledgments

- This work is supported by [MaRDI](https://www.mardi4nfdi.de).
- The design of this package is inspired by
  [JAX](https://github.com/jax-ml/jax).
- For JIT compilation, we build on the [OpenXLA](https://openxla.org/)
  project.
- Most of the compiler binaries are downloaded from
  [zml](https://github.com/zml/pjrt-artifacts/).
