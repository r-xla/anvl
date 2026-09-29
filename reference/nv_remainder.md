# Remainder (Truncating)

Element-wise remainder. This differs from base R's `%%`, use
[`nv_mod()`](https://r-xla.github.io/anvl/reference/nv_mod.md)/`%%`
instead.

## Usage

``` r
nv_remainder(x, y)
```

## Arguments

- x, y:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Two inputs with a [common data
  type](https://r-xla.github.io/anvl/reference/common_dtype.md). Can be
  any numeric data type. Scalars are broadcast. An R value takes the
  other operand's data type when that is in its own or a higher
  [category](https://r-xla.github.io/anvl/reference/dtype_categories.md).
  Otherwise it settles on its [default data
  type](https://r-xla.github.io/anvl/reference/default_dtypes.md), and
  the operands meet at their common data type.

## Value

([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
Has the inputs' broadcast shape and common data type.

## See also

[`nv_mod()`](https://r-xla.github.io/anvl/reference/nv_mod.md) for the
flooring remainder,
[`prim_remainder()`](https://r-xla.github.io/anvl/reference/prim_remainder.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(7, 8, 9))
y <- nv_array(c(3, 3, 4))
nv_remainder(x, y)
#> AnvlArray
#>  1
#>  2
#>  1
#> [ CPUf32{3} ] 

# different data types are promoted to their common one
nv_remainder(nv_scalar(7, "f32"), nv_scalar(3, "f64"))
#> AnvlArray
#>  1
#> [ CPUf64{} ] 

# a scalar is broadcast and an R integer is converted to a float
nv_remainder(x, 3L)
#> AnvlArray
#>  1
#>  2
#>  0
#> [ CPUf32{3} ] 
```
