# Clamp

Element-wise clamp: `min(max(min, x), max)`.

## Usage

``` r
nv_clamp(x, min, max)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- min, max:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Lower and upper bound, each scalar or the same shape as `x`. They are
  brought to `x`'s data type: an R value is built at it when its
  category can reach it (`0L` serves an integer and a float `x` alike,
  `0` only a float one), and a value that already has a data type is
  converted unless that would narrow it – an `f64` bound for an `f32`
  `x` is an error rather than a silent narrowing.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has `x`'s shape and data type.

## See also

[`prim_clamp()`](https://r-xla.github.io/anvl/dev/reference/prim_clamp.md)
for the underlying primitive.

## Examples

``` r
# the bounds are brought to `x`'s data type
x <- nv_array(c(-1, 0.5, 2))
nv_clamp(x, nv_scalar(0), nv_scalar(1))
#> AnvlArray
#>  0.0000
#>  0.5000
#>  1.0000
#> [ CPUf32{3} ] 

# an R integer serves a float `x` too, since a float can hold it
nv_clamp(x, 0L, 1L)
#> AnvlArray
#>  0.0000
#>  0.5000
#>  1.0000
#> [ CPUf32{3} ] 
```
