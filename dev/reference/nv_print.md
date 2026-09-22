# Print Array

Prints an array value to the console during JIT execution and returns
the input unchanged. Useful for debugging. For
[`RData`](https://r-xla.github.io/anvl/dev/reference/RData.md) inputs,
that do not have an actual data type, the [default data
type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md) is
used for printing.

## Usage

``` r
nv_print(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Returns the input unchanged, data type and shape included.

## See also

[`prim_print()`](https://r-xla.github.io/anvl/dev/reference/prim_print.md)
for the underlying primitive.

## Examples

``` r
# the value is printed and handed back unchanged
x <- nv_array(c(1, 2, 3))
nv_print(x)
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
# RData is printed at the default dtype
nv_print(1)
#> RData
#>  1
#> [ double{} printed at f32 ]
#> AnvlArray
#>  1
#> [ CPUf32{} ] 
```
