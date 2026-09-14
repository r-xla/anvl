# Bitwise Not

Element-wise bitwise NOT of an integer array, which for a boolean array
is the logical NOT.

## Usage

``` r
nv_not(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## The `!` operator

`!` is *logical*, like in base R. Unlike base R it only accepts booleans
and does not auto-convert non-booleans by comparing them with 0.

## See also

[`prim_not()`](https://r-xla.github.io/anvl/dev/reference/prim_not.md)
for the underlying primitive.

## Examples

``` r
nv_not(nv_array(c(TRUE, FALSE, TRUE)))
#> AnvlArray
#>  0
#>  1
#>  0
#> [ CPUbool{3} ] 
nv_not(nv_array(12L)) # bitwise: -13
#> AnvlArray
#>  -13
#> [ CPUi32{1} ] 
!nv_array(c(TRUE, FALSE)) # logical
#> AnvlArray
#>  0
#>  1
#> [ CPUbool{2} ] 
```
