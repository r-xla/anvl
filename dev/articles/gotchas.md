# Gotchas

This vignette lists various things to be aware of, specifically in
relation to base R.

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

When two non-scalar arrays differ only by size-1 axes (e.g. shape
`(2, 3)` and `(1, 3)`), use
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

Axes are aligned **from the first**: an array with fewer axes gets
size-1 axes appended, so its axis 1 meets axis 1 of the other. Anvl
arrays are column-major, so the first axis is the one that varies
fastest, and appending leaves every axis the array already had meaning
what it did. (NumPy prepends instead, which is the matching choice for a
row-major array.) A vector therefore meets a matrix as a **column**: a
length-`nrow` vector broadcasts against it, one value per row, and a
length-`ncol` one does not.

``` r

x <- nv_array(c(1, 2, 3))
y <- nv_array(matrix(1, nrow = 3, ncol = 3))
xs <- nv_broadcast_arrays(x, y)
xs[[1]] + xs[[2]]
```

    ## AnvlArray
    ##  2 2 2
    ##  3 3 3
    ##  4 4 4
    ## [ CPUf32{3,3} ]

which is what base R already gives you for the same two operands:

``` r

matrix(1, nrow = 3, ncol = 3) + c(1, 2, 3)
```

    ##      [,1] [,2] [,3]
    ## [1,]    2    2    2
    ## [2,]    3    3    3
    ## [3,]    4    4    4

``` r

nv_broadcast_to(nv_array(c(10, 20, 30)), shape = c(2, 3))
```

    ## Error in `prim_broadcast_in_axes()`:
    ## ! Axis 1 of `x` must be 2 or 1 to broadcast to axis 1 of the result.
    ## ✖ Got shapes (3) and (2x3).

This is *not* base R’s recycling, which is a different mechanism: base R
has no broadcasting between arrays at all
(`matrix(1:6, 2) + matrix(c(10, 20, 30), 1)` is a “non-conformable
arrays” error), and recycles a shorter *vector* over the column-major
flattening instead, warning only when the lengths do not divide. That
agrees with broadcasting whenever the vector’s length is the size of the
first axis – the case above, and the usual way a vector meets a matrix –
and to disagree quietly otherwise – `matrix(1:6, 2) + c(10, 20, 30)`
recycles rather than adding one value per column. So
[`nv_broadcast_arrays()`](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_arrays.md)
cannot replicate recycling for shapes like `(4)` and `(2)`: every axis
must either match or be 1.

## No `NA`s

R has a dedicated missing-value marker (`NA`) for every atomic type.
{anvl} arrays do not – there is no representation of “missing” at the
XLA level, only `NaN` for floating point numbers. At a floating-point
dtype, an `NA` silently turns into a `NaN`:

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
yield `NaN`. What comes back depends on the data type the value was
built at, so it is pinned here:

``` r

as_array(nv_array(c(1, NA, 3), dtype = "f32"))
```

    ## [1]   1 NaN   3

The signed integer dtypes are a special case. R spells `NA_integer_` as
the bit pattern `-2147483648`, which is also what an `i32` array stores,
so the value survives the round trip – but on the device it is an
ordinary, very negative integer that nothing distinguishes from a
missing value. Both directions warn about it:

``` r

nv_scalar(NA_integer_, dtype = "i32")
```

    ## Warning: Input `data` contains at least one "NA", stored on the device as "-2147483648".
    ## ℹ The value materializes as "NA" again in R, which `as_array()` reports on the
    ##   way back.
    ## ℹ Use `suppressWarnings()` to silence this.

    ## AnvlArray
    ##  -2.1475e+09
    ## [ CPUi32{} ]

``` r

as.integer(nv_scalar(NA_integer_, dtype = "i32"))
```

    ## Warning: Input `data` contains at least one "NA", stored on the device as "-2147483648".
    ## ℹ The value materializes as "NA" again in R, which `as_array()` reports on the
    ##   way back.
    ## ℹ Use `suppressWarnings()` to silence this.

    ## Warning: Materialized <i32> buffer contains a value that R cannot distinguish from "NA".
    ## ℹ "i32" reserves the bit pattern "-2147483648" (`INT_MIN`); "i64" reserves
    ##   "-9223372036854775808" (`INT64_MIN`).
    ## ℹ Set `check = "err"` to make this an error, or `check = FALSE` to silence it.

    ## [1] NA

Nothing on the device reserves that bit pattern, so it also arises from
ordinary arithmetic – and such a value comes back to R as an `NA` too:

``` r

as_array(nv_scalar(-2147483647L) - nv_scalar(1L))
```

    ## Warning: Materialized <i32> buffer contains a value that R cannot distinguish from "NA".
    ## ℹ "i32" reserves the bit pattern "-2147483648" (`INT_MIN`); "i64" reserves
    ##   "-9223372036854775808" (`INT64_MIN`).
    ## ℹ Set `check = "err"` to make this an error, or `check = FALSE` to silence it.

    ## [1] NA

At every other dtype there is no bit pattern for an `NA` to land on, so
a missing value is an error, including for `bool`:

``` r

nv_scalar(NA)
```

    ## Error:
    ## ! Missing value (NA/NaN) cannot be converted to "pred".

``` r

nv_array(c(1L, NA, 3L), dtype = "i16")
```

    ## Error:
    ## ! Missing value (NA/NaN) cannot be converted to "i16" (element 2).

Array creators take no argument to change any of this: what happens to
an `NA` is fixed by the dtype you build at. Scan the data yourself with
[`anyNA()`](https://rdrr.io/r/base/NA.html) if you want to hear about a
missing value the dtype accepts.

Converters like
[`as_array()`](https://r-xla.github.io/anvl/dev/reference/as_array.md)
do take a `check` argument, with three levels: `"warn"` (the default)
warns about a value R’s type cannot hold, `"err"` makes it an error, and
`FALSE` skips the scan.

``` r

as_array(nv_scalar(NA_integer_, dtype = "i32"), check = "err")
```

    ## Warning: Input `data` contains at least one "NA", stored on the device as "-2147483648".
    ## ℹ The value materializes as "NA" again in R, which `as_array()` reports on the
    ##   way back.
    ## ℹ Use `suppressWarnings()` to silence this.

    ## Error in `tengen::as_array()`:
    ## ! Materialized <i32> buffer contains a value that R cannot distinguish
    ##   from "NA".
    ## ℹ "i32" reserves the bit pattern "-2147483648" (`INT_MIN`); "i64" reserves
    ##   "-9223372036854775808" (`INT64_MIN`).
    ## ℹ Set `check = FALSE` to skip this check.

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
cover the whole range, so wrapping is possible.
[`as_array()`](https://r-xla.github.io/anvl/dev/reference/as_array.md)
warns about it, and `check = "err"` makes it an error:

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

    ## Warning: Materialized <ui64> buffer contains a value `>= 2^63` that wrapped through R's
    ## signed <integer64>.
    ## ℹ Exactly `2^63` becomes `NA_integer64_`; larger values become negative
    ##   <integer64>.
    ## ℹ Set `check = "err"` to make this an error, or `check = FALSE` to silence it.

    ## integer64
    ## [1] -1

``` r

as_array(big, check = "err")
```

    ## Error in `tengen::as_array()`:
    ## ! Materialized <ui64> buffer contains a value `>= 2^63` that wrapped
    ##   through R's signed <integer64>.
    ## ℹ Exactly `2^63` becomes `NA_integer64_`; larger values become negative
    ##   <integer64>.
    ## ℹ Set `check = FALSE` to skip this check.

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
eager mode. The untyped `1` materializes at the default float, so under
the default `f32` the jitted result is rounded where the eager one is
not (at an `f64` default the two would agree, which is exactly the
point: the answer depends on a default rather than on the code):

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
