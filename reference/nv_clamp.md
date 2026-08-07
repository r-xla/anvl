# Clamp

Element-wise clamp: `max(min_val, min(x, max_val))`. Converts `min_val`
and `max_val` to the data type of `x`.

## Usage

``` r
nv_clamp(min_val, x, max_val)
```

## Arguments

- min_val, max_val:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Minimum and maximum values (scalar or same shape as `x`).

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## Details

The underlying stableHLO function already broadcasts scalars, so no need
to broadcast manually.

## See also

[`prim_clamp()`](https://r-xla.github.io/anvl/reference/prim_clamp.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(-1, 0.5, 2))
nv_clamp(nv_scalar(0), x, nv_scalar(1))
#> AnvlArray
#>  0.0000
#>  0.5000
#>  1.0000
#> [ CPUf32{3} ] 
```
