# Primitive Subtraction

Subtracts two arrays element-wise.

## Usage

``` r
prim_sub(lhs, rhs)
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
[`hlo_subtract()`](https://r-xla.github.io/stablehlo/reference/hlo_subtract.html),
specified under [subtract](https://openxla.org/stablehlo/spec#subtract).

## See also

[`nv_sub()`](https://r-xla.github.io/anvl/reference/nv_sub.md), `-`

## Examples

``` r
# two R values: both take an R double's default data type
prim_sub(5, 3)
#> AnvlArray
#>  2
#> [ CPUf32{} ] 

# the R value is built at the array's data type instead
prim_sub(5, nv_scalar(3, "f64"))
#> AnvlArray
#>  2
#> [ CPUf64{} ] 
```
