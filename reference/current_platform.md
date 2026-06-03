# Current Lowering Target Platform

Returns the target platform currently set by an enclosing
[`stablehlo()`](https://r-xla.github.io/anvl/reference/stablehlo.md)
call (e.g. `"cpu"`, `"cuda"`). Platform-aware lowering rules call this
to branch on the target — e.g. SVD switches to a layout-flip variant
when targeting CUDA with `m < n` because cuSOLVER's `gesvd` requires
`m >= n`. Returns `NULL` outside of a lowering call.

`local_platform()` sets the current platform for the duration of the
calling scope, restoring the previous value via
[`withr::defer()`](https://withr.r-lib.org/reference/defer.html) when
the scope exits. Useful in tests and for manually exercising
platform-aware lowering rules outside of a
[`stablehlo()`](https://r-xla.github.io/anvl/reference/stablehlo.md)
call.

## Usage

``` r
current_platform()

local_platform(platform, envir = parent.frame())
```

## Arguments

- platform:

  (`character(1)` \| `NULL`)  
  Target platform name (e.g. `"cpu"`, `"cuda"`), or `NULL` to clear it.

- envir:

  (`environment`)  
  Environment whose exit triggers restoration of the previous platform.

## Value

`current_platform()` returns `NULL` or `character(1)`.
`local_platform()` invisibly returns the previous platform.

## See also

[`stablehlo()`](https://r-xla.github.io/anvl/reference/stablehlo.md)
