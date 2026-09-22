# Primitive Print

Prints an array value to the console during execution and returns the
input unchanged. This is useful for debugging JIT-compiled code. Bare R
inputs print at their category's default data type.

## Usage

``` r
prim_print(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Returns the input unchanged.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`hlo_custom_call()`](https://r-xla.github.io/stablehlo/reference/hlo_custom_call.html).

## See also

[`nv_print()`](https://r-xla.github.io/anvl/dev/reference/nv_print.md)

## Examples

``` r
# the value is printed and handed back unchanged
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
# bare R inputs are printed at their category's default data type
prim_print(1L)
#> RData
#>  1
#> [ integer{} printed at i32 ]
#> AnvlArray
#>  1
#> [ CPUi32{} ] 
```
