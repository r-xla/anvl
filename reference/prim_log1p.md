# Primitive Log Plus One

Element-wise log(1 + x), more accurate for small x.

## Usage

``` r
prim_log1p(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrayish value of data type floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_log_plus_one()`](https://r-xla.github.io/stablehlo/reference/hlo_log_plus_one.html).

## See also

[`nv_log1p()`](https://r-xla.github.io/anvl/reference/nv_log1p.md)

## Examples

``` r
x <- nv_array(c(0, 0.001, 1))
prim_log1p(x)
#> AnvlArray
#>  0.0000
#>  0.0010
#>  0.6931
#> [ CPUf32{3} ] 
```
