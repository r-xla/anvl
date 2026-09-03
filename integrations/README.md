# Integration tests

Tests that need more than `R CMD check` can offer, and therefore live outside
`tests/`.

At the moment that is the custom-call path documented in
`vignettes/articles/custom_calls.Rmd`: `hypot.cpp` is a handler compiled
against {pjrt}'s FFI headers with `Rcpp::sourceCpp()`, registered under the
target name `anvl_hypot`, and exercised from anvl in
`testthat/test-custom-call-ffi.R` -- eagerly, under `jit()`, and through a
primitive with a reverse rule. Compiling it needs a C++20 toolchain, {Rcpp}
and an installed {pjrt}, which is why these are not part of the package test
suite and run in a single CI job (Ubuntu x86, CPU) --
`.github/workflows/integration-tests.yaml`.

Run them against an installed anvl:

```sh
R CMD INSTALL .
Rscript integrations/run.R          # everything
Rscript integrations/run.R custom   # only test files matching "custom"
```

Only the CPU handler is covered. A CUDA handler cannot be compiled by
`sourceCpp()` (it needs `nvcc` for the kernel) and there is no GPU CI runner,
so testing that path means building it the way the article's "Going to CUDA"
section describes.
