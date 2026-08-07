# Primitive Not

Element-wise logical NOT.

## Usage

``` r
prim_not(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrayish value of data type boolean, integer, or unsigned integer.

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
[`hlo_not()`](https://r-xla.github.io/stablehlo/reference/hlo_not.html).

## See also

[`nv_not()`](https://r-xla.github.io/anvl/reference/nv_not.md)

## Examples

``` r
x <- nv_array(c(TRUE, FALSE, TRUE))
prim_not(x)
#> AnvlArray
#>  0
#>  1
#>  0
#> [ CPUbool{3} ] 
```
