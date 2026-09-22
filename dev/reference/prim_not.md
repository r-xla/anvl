# Primitive Bitwise Not

Element-wise bitwise NOT – a logical NOT on a boolean input, and a
bit-by-bit one on an integer.

## Usage

``` r
prim_not(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any integerish data type. An R value materializes at
  its [default data
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
[`hlo_not()`](https://r-xla.github.io/stablehlo/reference/hlo_not.html),
specified under [not](https://openxla.org/stablehlo/spec#not).

## See also

[`nv_not()`](https://r-xla.github.io/anvl/dev/reference/nv_not.md)

## Examples

``` r
# on a boolean this is a logical negation
x <- nv_array(c(TRUE, FALSE, TRUE))
prim_not(x)
#> AnvlArray
#>  0
#>  1
#>  0
#> [ CPUbool{3} ] 

# on an integer it complements every bit, so `12L` becomes `-13`
prim_not(nv_array(12L))
#> AnvlArray
#>  -13
#> [ CPUi32{1} ] 
```
