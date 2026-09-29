# Primitive Division

Divides two arrays element-wise.

## Usage

``` r
prim_div(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Two inputs of the same data type and shape. Can be any numeric data
  type. R values take the other operand's data type when it is in their
  [data type
  category](https://r-xla.github.io/anvl/reference/dtype_categories.md),
  and their [default data
  type](https://r-xla.github.io/anvl/reference/default_dtypes.md) when
  neither operand has one. An R value outside the other operand's
  category is an error, as are two R values of different storage types.

## Value

([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
Has the inputs' shape and data type.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_divide()`](https://r-xla.github.io/stablehlo/reference/hlo_divide.html),
specified under [divide](https://openxla.org/stablehlo/spec#divide).

## See also

[`nv_div()`](https://r-xla.github.io/anvl/reference/nv_div.md), `/`

## Examples

``` r
# two R values: both take an R double's default data type
prim_div(10, 4)
#> AnvlArray
#>  2.5000
#> [ CPUf32{} ] 

# the R value is built at the array's data type instead
prim_div(10, nv_scalar(4, "f64"))
#> AnvlArray
#>  2.5000
#> [ CPUf64{} ] 
```
