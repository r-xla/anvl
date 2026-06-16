# Primitive Print

Prints an array value to the console during execution and returns the
input unchanged. This is useful for debugging JIT-compiled code.

## Usage

``` r
prim_print(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Returns `operand` as-is.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`hlo_custom_call()`](https://r-xla.github.io/stablehlo/reference/hlo_custom_call.html).

## See also

[`nv_print()`](https://r-xla.github.io/anvl/dev/reference/nv_print.md)

## Examples

``` r
x <- nv_array(c(1, 2, 3))
prim_print(x)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ f32{3} ]
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
