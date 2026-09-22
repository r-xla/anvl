# Sign

Element-wise sign function: `-1`, `0` or `1` at the input's data type.
An unsigned input holds no negative value, so its sign is `0` or `1`,
like base R's [`sign()`](https://rdrr.io/r/base/sign.html) on a
non-negative number. You can also use
[`sign()`](https://rdrr.io/r/base/sign.html).

## Usage

``` r
nv_sign(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any numeric data type. An R value materializes at
  its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's shape and data type.

## See also

[`prim_sign()`](https://r-xla.github.io/anvl/dev/reference/prim_sign.md)
for the underlying primitive, which takes a signed input only.

## Examples

``` r
# the input's data type carries through
x <- nv_array(c(-3, 0, 5))
sign(x)
#> AnvlArray
#>  -1
#>   0
#>   1
#> [ CPUf32{3} ] 

# an unsigned input is 0 where it is 0 and 1 everywhere else
nv_sign(nv_array(c(0L, 3L), dtype = "ui32"))
#> AnvlArray
#>  0
#>  1
#> [ CPUui32{2} ] 

# an R value materializes at its default data type
nv_sign(-3)
#> AnvlArray
#>  -1
#> [ CPUf32{} ] 
```
