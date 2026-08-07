# Print Array

Prints an array value to the console during JIT execution and returns
the input unchanged. Useful for debugging.

## Usage

``` r
nv_print(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Input array.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Returns `x` unchanged.

## See also

[`prim_print()`](https://r-xla.github.io/anvl/reference/prim_print.md)
for the underlying primitive.

## Examples

``` r
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
```
