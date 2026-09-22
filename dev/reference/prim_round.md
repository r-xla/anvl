# Primitive Round

Rounds the elements of an array to the nearest integer.

## Usage

``` r
prim_round(x, method = "nearest_even")
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any float data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- method:

  (`character(1)`)  
  Rounding method. `"nearest_even"` (default) rounds to the nearest even
  integer on a tie, `"afz"` rounds away from zero on a tie.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has `x`'s data type and shape.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`hlo_round_nearest_even()`](https://r-xla.github.io/stablehlo/reference/hlo_round_nearest_even.html),
specified under
[round_nearest_even](https://openxla.org/stablehlo/spec#round_nearest_even).
With `method = "afz"` it lowers to
[`hlo_round_nearest_afz()`](https://r-xla.github.io/stablehlo/reference/hlo_round_nearest_afz.html)
instead, specified under
[round_nearest_afz](https://openxla.org/stablehlo/spec#round_nearest_afz).

## See also

[`nv_round()`](https://r-xla.github.io/anvl/dev/reference/nv_round.md)

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(1.4, 2.5, 3.6))
prim_round(x)
#> AnvlArray
#>  1
#>  2
#>  4
#> [ CPUf32{3} ] 

# an R value materializes at its default data type
prim_round(2.5)
#> AnvlArray
#>  2
#> [ CPUf32{} ] 
```
