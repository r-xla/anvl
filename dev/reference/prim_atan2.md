# Primitive Arctangent 2

Element-wise atan2 operation.

## Usage

``` r
prim_atan2(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two inputs of the same data type and shape. Can be any float data
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

- `reverse`

## StableHLO

Lowers to
[`hlo_atan2()`](https://r-xla.github.io/stablehlo/reference/hlo_atan2.html),
specified under [atan2](https://openxla.org/stablehlo/spec#atan2).

## See also

[`nv_atan2()`](https://r-xla.github.io/anvl/dev/reference/nv_atan2.md)

## Examples

``` r
# two R values: both take an R double's default data type
prim_atan2(1, 1)
#> AnvlArray
#>  0.7854
#> [ CPUf32{} ] 

# the R value is built at the array's data type instead
prim_atan2(1, nv_scalar(1, "f64"))
#> AnvlArray
#>  0.7854
#> [ CPUf64{} ] 
```
