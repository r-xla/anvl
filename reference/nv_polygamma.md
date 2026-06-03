# Polygamma

Element-wise polygamma function: the `(n+1)`-th derivative of the
log-gamma function. The order `n` is broadcast against `x` (so
`nv_polygamma(1, x)` works for any `x`). For `n = 0` this is the digamma
function; for `n = 1`,
[`trigamma()`](https://rdrr.io/r/base/Special.html) dispatches here.

Inputs are [promoted to a common floating data
type](https://r-xla.github.io/anvl/reference/nv_promote_to_common.md)
and scalar arguments are
[broadcast](https://r-xla.github.io/anvl/reference/nv_broadcast_scalars.md)
to the shape of the non-scalar arguments.

## Usage

``` r
nv_polygamma(n, x)
```

## Arguments

- n, x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Floating-point arrayish values. After promotion and broadcasting, `n`
  and `x` must have the same shape; `n` typically holds non-negative
  integer values.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and the promoted common data type of the inputs.

## See also

[`prim_polygamma()`](https://r-xla.github.io/anvl/reference/prim_polygamma.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(0.5, 1, 2, 5))
nv_polygamma(1, x) # trigamma
#> AnvlArray
#>  4.9348
#>  1.6449
#>  0.6449
#>  0.2213
#> [ CPUf32{4} ] 
```
