# Bitwise Not

Element-wise bitwise NOT – a logical negation on a boolean input, and a
bit-by-bit complement on an integer, so `nv_not(12L)` is `-13`. You can
also use the `!` operator, which expects `bool` inputs.

## Usage

``` r
nv_not(x)
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

## The `!` operator

`!` is *logical*, like in base R. Unlike base R it only accepts booleans
and does not auto-convert non-booleans by comparing them with 0.

## See also

[`prim_not()`](https://r-xla.github.io/anvl/dev/reference/prim_not.md)
for the underlying primitive.

## Examples

``` r
# on a boolean this is a logical negation
x <- nv_array(c(TRUE, FALSE, TRUE))
!x
#> AnvlArray
#>  0
#>  1
#>  0
#> [ CPUbool{3} ] 

# on an integer it complements every bit, so `12L` becomes `-13`
nv_not(nv_array(12L))
#> AnvlArray
#>  -13
#> [ CPUi32{1} ] 
```
