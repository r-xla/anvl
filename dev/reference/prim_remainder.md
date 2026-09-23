# Primitive Remainder

Element-wise remainder. The result has the sign of the dividend, which
is what StableHLO's `remainder` does. Base R's `%%` takes the sign of
the divisor instead and is available via
[`nv_mod()`](https://r-xla.github.io/anvl/dev/reference/nv_mod.md).

## Usage

``` r
prim_remainder(x, y)
```

## Arguments

- x, y:

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

- `reverse`

## StableHLO

Lowers to
[`hlo_remainder()`](https://r-xla.github.io/stablehlo/reference/hlo_remainder.html),
specified under
[remainder](https://openxla.org/stablehlo/spec#remainder).

## See also

[`nv_remainder()`](https://r-xla.github.io/anvl/dev/reference/nv_remainder.md)

## Examples

``` r
# two R values: both take an R double's default data type
prim_remainder(1, -3)
#> AnvlArray
#>  1
#> [ CPUf32{} ] 

# the R value is built at the array's data type instead
prim_remainder(1, nv_scalar(-3, "f64"))
#> AnvlArray
#>  1
#> [ CPUf64{} ] 

# the sign follows the dividend, where base R's %% follows the divisor
1 %% -3
#> [1] -2
```
