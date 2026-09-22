# Primitive Log Plus One

Element-wise log(1 + x), more accurate for small x.

## Usage

``` r
prim_log1p(x)
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
[`hlo_log_plus_one()`](https://r-xla.github.io/stablehlo/reference/hlo_log_plus_one.html),
specified under
[log_plus_one](https://openxla.org/stablehlo/spec#log_plus_one).

## See also

[`nv_log1p()`](https://r-xla.github.io/anvl/dev/reference/nv_log1p.md)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(0, 0.001, 1))
prim_log1p(x)
#> AnvlArray
#>  0.0000
#>  0.0010
#>  0.6931
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_log1p(0.001)
#> AnvlArray
#>  0.0010
#> [ CPUf32{} ] 
```
