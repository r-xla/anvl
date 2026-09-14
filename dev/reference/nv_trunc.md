# Truncate

Element-wise truncation (round toward zero). You can also use
[`trunc()`](https://rdrr.io/r/base/Round.html).

## Usage

``` r
nv_trunc(x)

# S3 method for class 'AnvlArray'
trunc(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array. An integer array is returned unchanged.

- ...:

  Further arguments of `nv_trunc()`.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nv_floor()`](https://r-xla.github.io/anvl/dev/reference/nv_floor.md),
[`nv_ceiling()`](https://r-xla.github.io/anvl/dev/reference/nv_ceiling.md),
[`nv_round()`](https://r-xla.github.io/anvl/dev/reference/nv_round.md).

## Examples

``` r
x <- nv_array(c(1.2, 2.7, -1.5))
trunc(x)
#> AnvlArray
#>   1
#>   2
#>  -1
#> [ CPUf32{3} ] 
trunc(nv_array(1:3)) # an integer array is already whole
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUi32{3} ] 
```
