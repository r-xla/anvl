# Primitive Generic Reduce

Reduces an array along the specified axes using a user-supplied
associative reducer. `reductor` and `init` must satisfy the constraints
in the "Associativity Requirement" section below.

## Usage

``` r
prim_reduce(x, init, axes, drop = TRUE, reductor)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The array to reduce. Can be any data type. `x` and `init` must have
  the same data type. An R value among them assumes the data type of the
  others when it is in its [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md), and
  its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when none of them has one.

- init:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Scalar initial value, with no axes. Must be the neutral element w.r.t.
  `reductor`, and shares `x`'s data type – see `x`.

- axes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Axes to reduce over. Negative values count from the end, i.e. `-1`
  refers to the last axis.

- drop:

  (`logical(1)`)  
  If `TRUE` (default) the reduced axes are removed; if `FALSE` they are
  kept with size 1.

- reductor:

  (`function`)  
  Binary reducer producing a scalar of the same data type as `x`. Its
  two arguments are passed by position, so they may carry any names.
  Must be associative (see "Associativity Requirement").

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's data type. The shape is the input's with the reduced
axes removed (`drop = TRUE`) or set to 1 (`drop = FALSE`).

## Associativity Requirement

The order in which `reductor` is applied across the reduction window is
implementation-defined. If the reductor is not associative, the result
is ill-defined. Furthermore, `init` must be the neutral element for this
reductor. Because arithmetic in a float data type is non-associative,
the output of the reduction can differ between backends (GPU, CPU), even
if the underlying mathematical function (like `+`) is associative.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`hlo_reduce()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce.html),
specified under [reduce](https://openxla.org/stablehlo/spec#reduce). The
body is `reductor`.

## See also

[`prim_reduce_sum()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce_sum.md),
[`prim_reduce_max()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce_max.md)

## Examples

``` r
# `init` shares `x`'s data type, and the reduced axis disappears
x <- nv_array(c(1, 2, 3, 4))
prim_reduce(x, init = nv_scalar(0), axes = 1L, reductor = prim_add)
#> AnvlArray
#>  10
#> [ CPUf32{} ] 
prim_reduce(x, init = nv_scalar(1), axes = 1L, reductor = prim_mul)
#> AnvlArray
#>  24
#> [ CPUf32{} ] 
```
