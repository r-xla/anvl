# Primitive Is Finite

Element-wise check if values are finite (not Inf, -Inf, or NaN).

## Usage

``` r
prim_is_finite(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any float data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the same shape as the input and boolean data type.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_is_finite()`](https://r-xla.github.io/stablehlo/reference/hlo_is_finite.html),
specified under
[is_finite](https://openxla.org/stablehlo/spec#is_finite).

## See also

[`nv_is_finite()`](https://r-xla.github.io/anvl/dev/reference/nv_is_finite.md)

## Examples

``` r
# the result is boolean, whatever float data type the input has
x <- nv_array(c(1, Inf, NaN, -Inf, 0))
prim_is_finite(x)
#> AnvlArray
#>  1
#>  0
#>  0
#>  0
#>  1
#> [ CPUbool{5} ] 

# an R value materializes at its default data type before the test
prim_is_finite(1)
#> AnvlArray
#>  1
#> [ CPUbool{} ] 
```
