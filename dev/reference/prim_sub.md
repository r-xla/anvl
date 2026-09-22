# Primitive Subtraction

Subtracts two arrays element-wise.

## Usage

``` r
prim_sub(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two inputs of the same data type and shape. Can be any numeric data
  type. R values assume the other operand's data type when it is in
  their [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md), and
  their [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when neither operand has one.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
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

[`nv_sub()`](https://r-xla.github.io/anvl/dev/reference/nv_sub.md), `-`

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
