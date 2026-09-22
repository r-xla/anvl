# Primitive Reciprocal Square Root

Element-wise reciprocal square root.

## Usage

``` r
prim_rsqrt(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any float data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's shape and data type.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_rsqrt()`](https://r-xla.github.io/stablehlo/reference/hlo_rsqrt.html),
specified under [rsqrt](https://openxla.org/stablehlo/spec#rsqrt).

## See also

[`nv_rsqrt()`](https://r-xla.github.io/anvl/dev/reference/nv_rsqrt.md)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1, 4, 9))
prim_rsqrt(x)
#> AnvlArray
#>  1.0000
#>  0.5000
#>  0.3333
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_rsqrt(4)
#> AnvlArray
#>  0.5000
#> [ CPUf32{} ] 
```
