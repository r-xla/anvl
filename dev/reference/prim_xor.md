# Primitive Bitwise Xor

Element-wise bitwise XOR – a logical XOR on a boolean input, and a
bit-by-bit one on an integer.

## Usage

``` r
prim_xor(x, y)
```

## Arguments

- x, y:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two inputs of the same data type and shape. Can be any integerish data
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
[`hlo_xor()`](https://r-xla.github.io/stablehlo/reference/hlo_xor.html),
specified under [xor](https://openxla.org/stablehlo/spec#xor).

## See also

[`nv_xor()`](https://r-xla.github.io/anvl/dev/reference/nv_xor.md)

## Examples

``` r
# two R values: both take an R integer's default data type
prim_xor(12L, 10L)
#> AnvlArray
#>  6
#> [ CPUi32{} ] 

# the R value is built at the array's data type instead
prim_xor(12L, nv_scalar(10L, "i64"))
#> AnvlArray
#>  6
#> [ CPUi64{} ] 
```
