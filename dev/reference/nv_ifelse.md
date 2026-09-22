# Conditional Element Selection

Selects elements from `true_value` or `false_value` based on `pred`,
analogous to R's [`ifelse()`](https://rdrr.io/r/base/ifelse.html).

## Usage

``` r
nv_ifelse(pred, true_value, false_value)
```

## Arguments

- pred:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Predicate array. Must be a boolean or an R logical, and scalar or the
  same shape as the non-scalar arguments.

- true_value, false_value:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Values to return where `pred` is `TRUE` / `FALSE`. Can be of any data
  type; the two are [promoted to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md).
  Scalars (including `pred`) are
  [broadcast](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_scalars.md)
  to the shape of the non-scalar arguments.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the common data type of `true_value` and `false_value`, and their
broadcast shape.

## See also

[`prim_ifelse()`](https://r-xla.github.io/anvl/dev/reference/prim_ifelse.md)
for the underlying primitive.

## Examples

``` r
pred <- nv_array(c(TRUE, FALSE, TRUE))
nv_ifelse(pred, nv_array(c(1, 2, 3)), nv_array(c(4, 5, 6)))
#> AnvlArray
#>  1
#>  5
#>  3
#> [ CPUf32{3} ] 
# scalar branches are broadcast and promoted to a common data type
nv_ifelse(pred, nv_scalar(1L), nv_scalar(0.5))
#> AnvlArray
#>  1.0000
#>  0.5000
#>  1.0000
#> [ CPUf32{3} ] 
```
