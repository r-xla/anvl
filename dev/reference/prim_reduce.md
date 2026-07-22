# Primitive Generic Reduce

Reduces an array along the specified dimensions using a user-supplied
associative reducer.

## Usage

``` r
prim_reduce(x, init, dims, drop = TRUE, reductor)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- init:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Scalar (0-dimensional) initial value. Must have the same data type as
  `x` and be the neutral element w.r.t. `reductor`.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions to reduce over.

- drop:

  (`logical(1)`)  
  If `TRUE` (default) the reduced dimensions are removed; if `FALSE`
  they are kept with size 1.

- reductor:

  (`function(lhs, rhs)`)  
  Binary reducer producing a scalar of the same dtype as `x`. Must be
  associative (see "Associativity Requirement").

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Same data type as `x`. Shape is `x` with `dims` removed (or set to 1 if
`drop = FALSE`).

## Associativity Requirement

The order in which `reductor` is applied across the reduction window is
implementation-defined. If the reductor is not associative, the result
is ill-defined. Furthermore, `init` must be the neutral element for this
reductor. Because floating point math is non-associative, the output of
the reduction can differ between backends (GPU, CPU), even if the
underlying mathematical function (like `+`) is associative.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`hlo_reduce()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce.html)
with `reductor` as the body.

## See also

[`prim_reduce_sum()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce_sum.md),
[`prim_reduce_max()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce_max.md)

## Examples

``` r
x <- nv_array(c(1, 2, 3, 4))
prim_reduce(x, init = nv_scalar(0), dims = 1L, reductor = prim_add)
#> AnvlArray
#>  10
#> [ CPUf32{} ] 
prim_reduce(x, init = nv_scalar(1), dims = 1L, reductor = prim_mul)
#> AnvlArray
#>  24
#> [ CPUf32{} ] 
```
