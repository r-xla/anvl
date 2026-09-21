# Gotchas

This vignette lists various things to be aware of, specifically in
relation to base R.

## Row-major vs column-major ordering

R stores matrices and arrays in *column-major* order, while {anvl}
(following XLA) uses *row-major* order. For most operations, this is an
internal implementation detail that does not change the semantics.
However, for reshaping operations such as
[`nv_flatten()`](https://r-xla.github.io/anvl/dev/reference/nv_flatten.md)
there is a difference.

Consider the 2x2 matrix below:

``` r

m <- matrix(1:4, nrow = 2)
m
```

    ##      [,1] [,2]
    ## [1,]    1    3
    ## [2,]    2    4

In base R, [`as.vector()`](https://rdrr.io/r/base/vector.html) flattens
it column-by-column, so we get `1, 2, 3, 4`:

``` r

as.vector(m)
```

    ## [1] 1 2 3 4

In {anvl}, reshaping to a length-4 vector traverses the data row-by-row,
so we get `1, 3, 2, 4`:

``` r

nv_flatten(m)
```

    ## AnvlArray
    ##  1
    ##  3
    ##  2
    ##  4
    ## [ CPUi32{4} ]

If you need column-major flattening in {anvl}, transpose first:

``` r

nv_flatten(t(m))
```

    ## AnvlArray
    ##  1
    ##  2
    ##  3
    ##  4
    ## [ CPUi32{4} ]

## No recycling

Base R *recycles* the shorter operand when two vectors of different
lengths are combined elementwise:

``` r

c(1, 2, 3, 4) + c(1, 2)
```

    ## [1] 2 4 4 6

{anvl} only auto-broadcasts *scalars* (operands with shape
[`integer()`](https://rdrr.io/r/base/integer.html)). Adding a scalar to
an array works as you would expect:

``` r

nv_array(1:4) + 10L
```

    ## AnvlArray
    ##  11
    ##  12
    ##  13
    ##  14
    ## [ CPUi32{4} ]

But combining two non-scalar arrays of different shapes errors, even
when one shape is a “tile” of the other:

``` r

nv_array(1:4) + nv_array(1:2)
```

    ## Error:
    ## ! All non-scalar arrays must have the same shape, but got (4), (2). Use
    ##   `nv_broadcast_arrays()` for general broadcasting.

When two non-scalar arrays differ only by size-1 axes (numpy-style
broadcasting, e.g. shape `(2, 3)` and `(1, 3)`), use
[`nv_broadcast_arrays()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_arrays.md)
to align them explicitly first:

``` r

a <- nv_matrix(1:6, nrow = 2)
shape(a)
```

    ## [1] 2 3

``` r

b <- nv_matrix(c(10, 20, 30), nrow = 1)
shape(b)
```

    ## [1] 1 3

``` r

xs <- nv_broadcast_arrays(a, b)
lapply(xs, shape)
```

    ## [[1]]
    ## [1] 2 3
    ## 
    ## [[2]]
    ## [1] 2 3

``` r

xs[[1]] + xs[[2]]
```

    ## AnvlArray
    ##  11 23 35
    ##  12 24 36
    ## [ CPUf32{2,3} ]

Note that even
[`nv_broadcast_arrays()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_arrays.md)
cannot replicate R’s recycling for shapes like `(4)` and `(2)` – the
shapes must be broadcast-compatible in the numpy sense.

## No `NA`s

R has a dedicated missing-value marker (`NA`) for every atomic type.
{anvl} arrays do not – there is no representation of “missing” at the
XLA level, only `NaN` for floating point numbers. When you convert R
values containing `NA` into an `AnvlArray`, the `NA`s are silently
turned into `NaN`s.

``` r

nv_array(NA_real_)
```

    ## AnvlArray
    ##  nan
    ## [ CPUf32{1} ]

``` r

nv_array(c(1, NA, 3))
```

    ## AnvlArray
    ##    1
    ##  nan
    ##    3
    ## [ CPUf32{3} ]

Round-tripping back to R is not guaranteed to produce `NA`, but can also
yield `NaN`:

``` r

as_array(nv_array(c(1, NA, 3)))
```

    ## [1]   1 NaN   3

For other data types, the situation is even worse, especially for
integers, where R uses the smallest possible value to represent
missingness:

``` r

nv_scalar(NA_integer_)
```

    ## Warning: Input `data` contains 1 "NA" value, stored on the device as "-2147483648".
    ## ℹ The value materializes as "NA" again in R, which `as_array()` reports on the
    ##   way back.
    ## ℹ Use `suppressWarnings()` to silence this.

    ## AnvlArray
    ##  -2.1475e+09
    ## [ CPUi32{} ]

However, when you convert it back, you get a missing value again:

``` r

as.integer(nv_scalar(NA_integer_))
```

    ## Warning: Input `data` contains 1 "NA" value, stored on the device as "-2147483648".
    ## ℹ The value materializes as "NA" again in R, which `as_array()` reports on the
    ##   way back.
    ## ℹ Use `suppressWarnings()` to silence this.

    ## [1] NA

When creating logicals, `NA` will be interpreted as `TRUE`:

``` r

nv_scalar(NA)
```

    ## AnvlArray
    ##  1
    ## [ CPUbool{} ]

``` r

as.logical(nv_scalar(NA))
```

    ## [1] TRUE

In order to avoid these pitfals, array creators such as
[`nv_array()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
have a `check` argument to prevent the above problems. It is `FALSE` by
default, because it needs to scan the complete data.

``` r

nv_array(c(1, NA, 3), check = TRUE)
```

    ## Error in `nv_array()`:
    ## ! Input `data` contains 1 "NA" value, which has no representation at the
    ##   XLA level.
    ## ℹ Replace or drop missing values before transferring, or set `check = FALSE` to
    ##   skip this check.

The same flag is available for converters like
[`as_array()`](https://r-xla.github.io/anvl/dev/reference/as_array.md):

``` r

as_array(nv_scalar(NA_integer_), check = TRUE)
```

    ## Warning: Input `data` contains 1 "NA" value, stored on the device as "-2147483648".
    ## ℹ The value materializes as "NA" again in R, which `as_array()` reports on the
    ##   way back.
    ## ℹ Use `suppressWarnings()` to silence this.

    ## Error in `check_level()`:
    ## ! `check` must be "warn", "err" or `FALSE`, not TRUE.
    ## ℹ `TRUE` is not accepted; pick "warn" or "err".

## Subnormal floating-point values

Subnormal values are extremely small nonzero numbers, very close to
zero. They allow floating-point numbers to extend below their normal
range, but with fewer significant digits of precision. For R’s usual
double-precision numbers (IEEE 754 binary64), subnormal magnitudes range
from approximately \\4.94 \times 10^{-324}\\ to just below \\2.23 \times
10^{-308}\\. More precisely, their magnitudes are at least \\2^{-1074}\\
but strictly less than \\2^{-1022}\\, which is the smallest positive
normal value.

With {anvl}’s default PJRT backend, these tiny values can be stored in
an array and read back into R unchanged, but may become zero when used
in calculations. On CPUs, XLA enables a mode that treats subnormal
inputs and results as zero, often called “flushing to zero”. The exact
behavior depends on the platform, backend, and operation.
Single-precision (`f32`) values are affected in the same way, below
approximately \\1.18 \times 10^{-38}\\. Comparisons flush their inputs
too, so for example a negative subnormal can test as greater than or
equal to zero.

Running the following example demonstrates this effect, with “Observed”
comments corresponding to results from the PJRT CPU backend.

``` r

library(anvl)

with_backend("pjrt", {
  x <- nv_scalar(1e-310, dtype = "f64")

  as.vector(x)  # Preserved: approximately 1e-310

  # Subnormal input; mathematically normal result (should give ~1e-210).
  as.vector(x * nv_scalar(1e100, dtype = "f64"))  # Observed: 0

  # Normal inputs; mathematically subnormal result (should give ~1e-310).
  as.vector(
    nv_scalar(1e-200, dtype = "f64") *
      nv_scalar(1e-110, dtype = "f64")
  )  # Observed: 0

  # Comparisons flush their inputs as well.
  as.vector(nv_scalar(-1e-310, dtype = "f64") >= 0)  # Observed: TRUE

  # The same happens in single precision.
  as.vector(nv_scalar(1e-40, dtype = "f32"))  # Preserved: approximately 1e-40
  as.vector(nv_scalar(1e-40, dtype = "f32") * 1e30)  # Observed: 0
})
```

    ## [1] 0

If the first argument to
[`with_backend()`](https://r-xla.github.io/anvl/dev/reference/with_backend.md)
is changed to `"quickr"` – the experimental, optional backend described
in the [internals
vignette](https://r-xla.github.io/anvl/dev/articles/internals.md) – then
you should observe the results of subnormal computations are preserved
correctly.

This is also present, for example, in the [JAX gotchas
guide](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#miscellaneous-divergences-from-numpy)
which also illustrates values preserved in storage but flushed during
operations on some backends.

## No unsigned integers

R’s `integer` type is signed 32-bit (range `-2147483648` to
`2147483647`). {anvl} also exposes unsigned integer dtypes (`ui8`,
`ui16`, `ui32`, `ui64`) backed by XLA, but R has no native counterpart.
For values that fit into R’s signed integer range, the round-trip works
as expected:

``` r

as_array(nv_array(c(0L, 200L, 255L), dtype = "ui8"))
```

    ## [1]   0 200 255

Because `ui32` does not fit into R’s native integer type, it will be
converted to
[`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html)
data type:

``` r

big <- nv_array(2147483647L, dtype = "ui32") + 1L
as_array(big)
```

    ## integer64
    ## [1] 2147483648

However, for `ui64`, we also convert to `integer64`, which does not
cover the whole range, so overflow is possible, but can be detected via
the `check` flag:

``` r

big <- nv_array(0L, dtype = "ui64") - 1L
big
```

    ## AnvlArray
    ##  1.8447e+19
    ## [ CPUui64{1} ]

``` r

as_array(big)
```

    ## integer64
    ## [1] -1

``` r

as_array(big, check = TRUE)
```

    ## Error in `check_level()`:
    ## ! `check` must be "warn", "err" or `FALSE`, not TRUE.
    ## ℹ `TRUE` is not accepted; pick "warn" or "err".

## Differences between eager and jit-mode

We try to keep the semantics of eager and jit-mode as close as possible.
There are a few reasons why this might not be the case, which are listed
here:

## The function does not canonicalize its inputs.

One difference arises when eager functions do not canonicalize their
inputs. Here, canoicalizing refers to the conversion or dynamic R inputs
to `AnvlArray`s. This can be done via `as_anvl_array` for single
arguments and `as_anvl_arrays` for multiple arguments.

E.g., the following function does not behave the same in jit-mode and in
eager mode.

``` r

add_pi <- function(x) {
  x + pi
}
as.numeric(jit(add_pi)(1)) == add_pi(1)
```

    ## [1] FALSE

We can fix this, by canonicalizing the inputs:

``` r

add_pi2 <- function(x) {
  x <- as_anvl_array(x)
  x + pi
}
jit(add_pi2)(1) == add_pi2(1)
```

    ## AnvlArray
    ##  1
    ## [ CPUbool{} ]

Not canonicalizing inputs can also be a problem for device placement. If
we were to call the function below as
`threeway_add(1, 2, nv_scalar(3, "cuda"))`, then the `1 + 2` would first
move the `1` and `2` to the default device (which is CPU by default) and
the second addition would then fail because it would attempt to call
`nv_add` with mixed-device inputs.

``` r

threeway_add <- function(x, y, z) {
  nv_add(nv_add(x, y), z)
}
```

Therefore, you should always canonicalize your inputs in eager
functions.
