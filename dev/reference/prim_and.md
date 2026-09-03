# Primitive And

Element-wise logical AND.

## Usage

``` r
prim_and(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish values of data type boolean, integer, or unsigned integer.
  Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the inputs.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_and()`](https://r-xla.github.io/stablehlo/reference/hlo_and.html).

## See also

[`nv_and()`](https://r-xla.github.io/anvl/dev/reference/nv_and.md), `&`

## Examples

``` r
x <- nv_array(c(TRUE, FALSE, TRUE))
y <- nv_array(c(TRUE, TRUE, FALSE))
prim_and(x, y)
#> AnvlArray
#>  1
#>  0
#>  0
#> [ CPUbool{3} ] 
```
