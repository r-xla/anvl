# Install What a Backend Needs to Run

A backend needs more than the packages anvl declares as dependencies:
the `"pjrt"` backend runs on PJRT plugins that are downloaded rather
than shipped with the package, and the `"quickr"` backend needs
[quickr](https://CRAN.R-project.org/package=quickr), which is only
suggested. This installs whichever of the two the given backend is
missing.

## Usage

``` r
install_anvl(backend = active_backend(), ...)
```

## Arguments

- backend:

  (`character(1)`)  
  Backend to install for. Defaults to
  [`active_backend()`](https://r-xla.github.io/anvl/dev/reference/active_backend.md).

- ...:

  Passed to the underlying installer:
  [`pjrt::install_pjrt()`](https://r-xla.github.io/pjrt/reference/install_pjrt.html)
  for `"pjrt"`,
  [`utils::install.packages()`](https://rdrr.io/r/utils/install.packages.html)
  for `"quickr"`.

## Value

(`NULL`)  
Invisibly. Called for its side effect.

## Details

The PJRT plugins are downloaded on demand, but not silently: the first
time a plugin is needed, an interactive session asks for confirmation,
while a non-interactive session does not download at all. Call this to
make the download an explicit step instead, for instance in a
`Dockerfile` layer of its own or at the start of a script that later
runs unattended. The `PJRT_INSTALL` environment variable overrides the
prompt: `"1"` always downloads without asking, `"0"` never downloads.

For `"pjrt"`, the CPU plugin is always installed, and the CUDA plugin
too when an NVIDIA GPU is detected on Linux (or `cuda = TRUE` is
passed). The CUDA plugin additionally needs the CUDA libraries, which
come in the `pjrt.cuda` R package from the r-xla r-universe;
`install_anvl()` installs it along with the CUDA plugin. See
[`pjrt::install_pjrt()`](https://r-xla.github.io/pjrt/reference/install_pjrt.html)
for details.
