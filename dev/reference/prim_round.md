# Primitive Round

Rounds the elements of an array to the nearest integer.

## Usage

``` r
prim_round(x, method = "nearest_even")
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of data type floating-point.

- method:

  (`character(1)`)  
  Rounding method. `"nearest_even"` (default) rounds to the nearest even
  integer on a tie, `"afz"` rounds away from zero on a tie.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same dtype and shape as `x`.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_round_nearest_even()`](https://r-xla.github.io/stablehlo/reference/hlo_round_nearest_even.html)
or
[`hlo_round_nearest_afz()`](https://r-xla.github.io/stablehlo/reference/hlo_round_nearest_afz.html)
depending on the `method` parameter.

## See also

[`nv_round()`](https://r-xla.github.io/anvl/dev/reference/nv_round.md)

## Examples

``` r
x <- nv_array(c(1.4, 2.5, 3.6))
prim_round(x)
#> AnvlArray
#>  1
#>  2
#>  4
#> [ CPUf32{3} ] 
```
