# Primitive Logarithm

Element-wise natural logarithm.

## Usage

``` r
prim_log(x)
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

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_log()`](https://r-xla.github.io/stablehlo/reference/hlo_log.html),
specified under [log](https://openxla.org/stablehlo/spec#log).

## See also

[`nv_log()`](https://r-xla.github.io/anvl/dev/reference/nv_log.md),
[`log()`](https://rdrr.io/r/base/Log.html)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1, 2.718, 7.389))
prim_log(x)
#> AnvlArray
#>  0.0000
#>  0.9999
#>  2.0000
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_log(2)
#> AnvlArray
#>  0.6931
#> [ CPUf32{} ] 
```
