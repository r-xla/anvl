# Primitive Xor

Element-wise logical XOR.

## Usage

``` r
prim_xor(lhs, rhs)
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
[`hlo_xor()`](https://r-xla.github.io/stablehlo/reference/hlo_xor.html).

## See also

[`nv_xor()`](https://r-xla.github.io/anvl/dev/reference/nv_xor.md)

## Examples

``` r
x <- nv_array(c(TRUE, FALSE, TRUE))
y <- nv_array(c(TRUE, TRUE, FALSE))
prim_xor(x, y)
#> AnvlArray
#>  0
#>  1
#>  1
#> [ CPUbool{3} ] 
```
