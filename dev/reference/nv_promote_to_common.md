# Promote Arrays to a Common Data Type

Promote arrays to a common data type, see
[`common_dtype`](https://r-xla.github.io/anvl/dev/reference/common_dtype.md)
for more details.

## Usage

``` r
nv_promote_to_common(...)
```

## Arguments

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrays to promote.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))

## Examples

``` r
x <- nv_array(1L)
y <- nv_array(1.5)
# integer is promoted to float
nv_promote_to_common(x, y)
#> [[1]]
#> AnvlArray
#>  1
#> [ CPUf32{1} ] 
#> 
#> [[2]]
#> AnvlArray
#>  1.5000
#> [ CPUf32{1} ] 
#> 
```
