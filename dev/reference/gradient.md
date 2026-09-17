# Gradient

Returns a new function that computes the gradient of `f` via
reverse-mode automatic differentiation. `f` must return a single float
scalar. The returned function has the same signature as `f` and returns
the gradients in the same structure as the inputs (or the subset
selected by `wrt`).

## Usage

``` r
gradient(f, wrt = NULL)
```

## Arguments

- f:

  (`function`)  
  Function to differentiate. Arguments can be arrayish
  ([`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md))
  or static (non-array) values. Must return a single scalar float array.

- wrt:

  (`character` \| `integer` \| `NULL`)  
  Names or positions of the arguments to compute the gradient with
  respect to. Only arrayish (float array) arguments can be included;
  static arguments must not appear in `wrt`. If `NULL` (the default),
  the gradient is computed with respect to all arguments (which must all
  be arrayish in that case).

## Value

`function`

## See also

[`value_and_gradient()`](https://r-xla.github.io/anvl/dev/reference/value_and_gradient.md)
to get both the output and gradients,
[`transform_gradient()`](https://r-xla.github.io/anvl/dev/reference/transform_gradient.md)
for the low-level graph transformation.

## Examples

``` r
f <- function(x, y) sum(x * y)
g <- jit(gradient(f))
g(nv_array(c(1, 2), dtype = "f32"), nv_array(c(3, 4), dtype = "f32"))
#> $x
#> AnvlArray
#>  3
#>  4
#> [ CPUf32{2} ] 
#> 
#> $y
#> AnvlArray
#>  1
#>  2
#> [ CPUf32{2} ] 
#> 

# differentiate with respect to a single argument
g_x <- jit(gradient(f, wrt = "x"))
g_x(nv_array(c(1, 2), dtype = "f32"), nv_array(c(3, 4), dtype = "f32"))
#> $x
#> AnvlArray
#>  3
#>  4
#> [ CPUf32{2} ] 
#> 

# static (non-array) arguments are passed through but cannot be in wrt
f2 <- function(x, power) sum(x^power)
g2 <- jit(gradient(f2, wrt = "x"), static = "power")
g2(nv_array(c(1, 2, 3), dtype = "f32"), power = 2L)
#> $x
#> AnvlArray
#>  2
#>  4
#>  6
#> [ CPUf32{3} ] 
#> 
```
