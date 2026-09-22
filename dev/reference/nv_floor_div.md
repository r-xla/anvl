# Flooring Division

Element-wise flooring division. You can also call this via the `%/%`
operator. The result is the largest whole number that does not exceed
`lhs / rhs`.

## Usage

``` r
nv_floor_div(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two inputs with a [common data
  type](https://r-xla.github.io/anvl/dev/reference/common_dtype.md). Can
  be any numeric data type. Scalars are broadcast, and R values assume
  the other operand's data type within their [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md),
  otherwise falling back to their [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  and being converted to the common data type.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the inputs' broadcast shape and common data type.

## See also

[`nv_mod()`](https://r-xla.github.io/anvl/dev/reference/nv_mod.md) for
the matching remainder,
[`nv_div()`](https://r-xla.github.io/anvl/dev/reference/nv_div.md) for
the division itself.

## Examples

``` r
x <- nv_array(c(7L, -7L))
y <- nv_array(c(2L, 2L))
nv_floor_div(x, y)
#> AnvlArray
#>   3
#>  -4
#> [ CPUi32{2} ] 
x %/% y
#> AnvlArray
#>   3
#>  -4
#> [ CPUi32{2} ] 
```
