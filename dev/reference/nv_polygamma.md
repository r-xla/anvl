# Polygamma

Element-wise polygamma function: the `(n+1)`-th derivative of the
log-gamma function. For `n = 0` this is the digamma function; for
`n = 1`, [`trigamma()`](https://rdrr.io/r/base/Special.html) dispatches
here.

## Usage

``` r
nv_polygamma(n, x)
```

## Arguments

- n, x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Order of the polygamma function and the value to evaluate it at. `n`
  typically holds non-negative whole numbers. Can be any numeric data
  type: the two are [promoted to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md)
  and that is then converted to the default float data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
  where it is not a float already, since a float is all
  [`prim_polygamma()`](https://r-xla.github.io/anvl/dev/reference/prim_polygamma.md)
  takes. An R value assumes the other operand's data type within its
  [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md), and
  settles on the default float when neither has one. Scalars are
  [broadcast](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_scalars.md)
  to the shape of the other, so `nv_polygamma(1, x)` works for any float
  `x`.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the inputs' broadcast shape, and their common data type – or the
default float data type (see
[`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md))
where that was an integer one.

## The [`trigamma()`](https://rdrr.io/r/base/Special.html) generic

`trigamma(x)` is `nv_polygamma(1L, x)`.

## See also

[`prim_polygamma()`](https://r-xla.github.io/anvl/dev/reference/prim_polygamma.md)
for the underlying primitive.

## Examples

``` r
# the R `1` is built at `x`'s float data type and broadcast
x <- nv_array(c(0.5, 1, 2, 5))
nv_polygamma(1, x) # trigamma
#> AnvlArray
#>  4.9348
#>  1.6449
#>  0.6449
#>  0.2213
#> [ CPUf32{4} ] 
```
