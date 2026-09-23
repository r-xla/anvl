# Conditional Element Selection

Selects elements from `yes` or `no` based on `test`, mirroring R's
[`ifelse()`](https://rdrr.io/r/base/ifelse.html).

[`nv_if()`](https://r-xla.github.io/anvl/dev/reference/nv_if.md) is the
other conditional: it mirrors R's `if` construct and branches between
two *functions*, evaluating only the selected one.

## Usage

``` r
nv_ifelse(test, yes, no)
```

## Arguments

- test:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Predicate array. Must be a boolean or an R logical, and scalar or the
  same shape as the non-scalar arguments.

- yes, no:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Values to return where `test` is `TRUE` / `FALSE`. Can be of any data
  type; the two are [promoted to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md).
  Scalars (including `test`) are
  [broadcast](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_scalars.md)
  to the shape of the non-scalar arguments.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the common data type of `yes` and `no`, and their broadcast shape.

## See also

[`prim_ifelse()`](https://r-xla.github.io/anvl/dev/reference/prim_ifelse.md)
for the underlying primitive,
[`nv_if()`](https://r-xla.github.io/anvl/dev/reference/nv_if.md).

## Examples

``` r
test <- nv_array(c(TRUE, FALSE, TRUE))
nv_ifelse(test, nv_array(c(1, 2, 3)), nv_array(c(4, 5, 6)))
#> AnvlArray
#>  1
#>  5
#>  3
#> [ CPUf32{3} ] 
# scalar branches are broadcast and promoted to a common data type
nv_ifelse(test, nv_scalar(1L), nv_scalar(0.5))
#> AnvlArray
#>  1.0000
#>  0.5000
#>  1.0000
#> [ CPUf32{3} ] 
```
