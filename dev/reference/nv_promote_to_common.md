# Promote Arrays to a Common Data Type

Promote arrays to a common data type, see
[`promotion_common()`](https://r-xla.github.io/anvl/dev/reference/promotion_rule.md)
for more details.

## Usage

``` r
nv_promote_to_common(...)
```

## Arguments

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Values to promote.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
The inputs, each at their common data type and with its own shape.

## Examples

``` r
# An integer is promoted to float
x <- nv_scalar(1, dtype = "f32")
y <- nv_scalar(1L, dtype = "i32")
nv_promote_to_common(x, y)
#> [[1]]
#> AnvlArray
#>  1
#> [ CPUf32{} ] 
#> 
#> [[2]]
#> AnvlArray
#>  1
#> [ CPUf32{} ] 
#> 
# an R value yields to the data type it meets within its category
with_default_dtypes(c(float = "f64"), nv_promote_to_common(1, x))
#> [[1]]
#> AnvlArray
#>  1
#> [ CPUf32{} ] 
#> 
#> [[2]]
#> AnvlArray
#>  1
#> [ CPUf32{} ] 
#> 
# and settles on its default otherwise
with_default_dtypes(c(float = "f64"), nv_promote_to_common(1, y))
#> [[1]]
#> AnvlArray
#>  1
#> [ CPUf64{} ] 
#> 
#> [[2]]
#> AnvlArray
#>  1
#> [ CPUf64{} ] 
#> 
```
