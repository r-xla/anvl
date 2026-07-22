# Logical Not

Element-wise logical NOT. You can also use the `!` operator.

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

## See also

[`prim_not()`](https://r-xla.github.io/anvl/dev/reference/prim_not.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(TRUE, FALSE, TRUE))
!x
#> AnvlArray
#>  0
#>  1
#>  0
#> [ CPUbool{3} ] 
```
