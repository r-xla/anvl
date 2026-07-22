# Round

Element-wise rounding. You can also use the
[`round()`](https://rdrr.io/r/base/Round.html) generic.

## Usage

``` r
nv_round(x, method = "nearest_even")
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Input array.

- method:

  (`character(1)`)  
  Rounding method. Either `"nearest_even"` (default) or `"afz"` (away
  from zero).

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_round()`](https://r-xla.github.io/anvl/dev/reference/prim_round.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1.4, 2.5, 3.6))
round(x)
#> AnvlArray
#>  1
#>  2
#>  4
#> [ CPUf32{3} ] 
```
