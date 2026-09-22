# Is Finite

Element-wise check if values are finite (not `Inf`, `-Inf`, or `NaN`).
Only a float holds a non-finite value, so the answer for any other data
type is all `TRUE` and is built as a constant rather than computed, the
way base R's [`is.finite()`](https://rdrr.io/r/base/is.finite.html)
answers `TRUE` for an integer.

## Usage

``` r
nv_is_finite(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the input's shape and boolean data type.

## See also

[`prim_is_finite()`](https://r-xla.github.io/anvl/dev/reference/prim_is_finite.md)
for the underlying primitive, which takes a float only.

## Examples

``` r
# the result is boolean, whatever float data type the input has
x <- nv_array(c(1, Inf, NaN, -Inf, 0))
nv_is_finite(x)
#> AnvlArray
#>  1
#>  0
#>  0
#>  0
#>  1
#> [ CPUbool{5} ] 

# all TRUE for an integer input, which has no non-finite value
nv_is_finite(nv_array(1:3))
#> AnvlArray
#>  1
#>  1
#>  1
#> [ CPUbool{3} ] 

# an R value materializes at its default data type before the test
nv_is_finite(1)
#> AnvlArray
#>  1
#> [ CPUbool{} ] 
```
