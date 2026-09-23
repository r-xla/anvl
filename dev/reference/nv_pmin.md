# Parallel Minimum

Element-wise minimum of two arrays, like
[`base::pmin()`](https://rdrr.io/r/base/Extremes.html).

## Usage

``` r
nv_pmin(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two inputs with a [common data
  type](https://r-xla.github.io/anvl/dev/reference/common_dtype.md). Can
  be any data type. Scalars are broadcast, and R values assume the other
  operand's data type within their [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md),
  otherwise falling back to their [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and being converted to the common data type.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the inputs' broadcast shape and common data type.

## See also

[`prim_pmin()`](https://r-xla.github.io/anvl/dev/reference/prim_pmin.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 5, 3))
y <- nv_array(c(4, 2, 6))
nv_pmin(x, y)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 

# different data types are promoted to their common one
nv_pmin(nv_scalar(1, "f32"), nv_scalar(5, "f64"))
#> AnvlArray
#>  1
#> [ CPUf64{} ] 

# a scalar is broadcast and an R integer is converted to a float
nv_pmin(x, 2L)
#> AnvlArray
#>  1
#>  2
#>  2
#> [ CPUf32{3} ] 
```
