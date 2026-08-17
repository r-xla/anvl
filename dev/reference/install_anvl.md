# Install what a backend needs to run

A backend needs more than the packages anvl declares as dependencies:
the `"pjrt"` backend runs on PJRT plugins that are downloaded rather
than shipped with the package, and the `"quickr"` backend needs
[quickr](https://CRAN.R-project.org/package=quickr), which is only
suggested. This installs whichever of the two the given backend is
missing.

## Usage

``` r
install_anvl(backend = default_backend(), ...)
```

## Arguments

- backend:

  (`character(1)`)  
  Backend to install for. Defaults to
  [`default_backend()`](https://r-xla.github.io/anvl/dev/reference/default_backend.md).
  The `"plain"` backend has nothing to install and is not accepted.

- ...:

  Passed to the underlying installer:
  [`pjrt::install_pjrt()`](https://r-xla.github.io/pjrt/reference/install_pjrt.html)
  for `"pjrt"`,
  [`utils::install.packages()`](https://rdrr.io/r/utils/install.packages.html)
  for `"quickr"`.

## Value

`NULL`, invisibly. Called for its side effect.

## Details

The PJRT plugins are downloaded on demand, but not silently: the first
time a plugin is needed, an interactive session asks for confirmation,
while a non-interactive session does not download at all. Call this to
make the download an explicit step instead, for instance in a
`Dockerfile` layer of its own or at the start of a script that later
runs unattended. The `PJRT_INSTALL` environment variable overrides the
prompt: `"1"` always downloads without asking, `"0"` never downloads.

Which plugins you get – and whether CUDA is available at all – is
decided by the repository anvl was installed from, not by this call. See
the installation vignette:
[`vignette("installation", package = "anvl")`](https://r-xla.github.io/anvl/dev/articles/installation.md).
