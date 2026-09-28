# Gradients

Return a new function that computes the gradient of `f` via reverse-mode
automatic differentiation. `f` must return a single float scalar. The
returned function has the same signature as `f`.

- `gradient()` returns only the gradients, structured like the inputs
  (or the subset selected by `wrt`).

- `value_and_gradient()` returns both the output of `f` and its
  gradients, computed in a single forward and reverse pass.

## Usage

``` r
gradient(f, wrt = NULL)

value_and_gradient(f, wrt = NULL)
```

## Arguments

- f:

  (`function`)  
  Function to differentiate. Must return a single scalar float array.

- wrt:

  (`character` \| `integer` \| `NULL`)  
  Names or positions of the arguments to compute the gradient with
  respect to. Only float arrays can be included; static arguments must
  not appear in `wrt`. At call time, an argument in `wrt` must be an
  array with a data type, not an R value such as `3`, since the data
  type of its gradient would otherwise be undetermined. If `NULL` (the
  default), the gradient is computed with respect to all arguments
  (which must all be arrayish in that case).

## Value

(`function`)  
Has the same formals as `f` and must be called inside
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md). For
`gradient()`, it returns a named `list` of gradients, one per argument
of `f` (or per `wrt` entry), each structured like that argument. For
`value_and_gradient()`, it returns `list(value = , grad = )`: the return
value of `f`, and that same `list` of gradients.

## See also

the [Automatic
Differentiation](https://r-xla.github.io/anvl/articles/autodiff.html)
article,
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

# an argument in `wrt` must be passed as an array, not as an R value
g3 <- jit(gradient(function(x) x^2L))
g3(nv_scalar(3))
#> $x
#> AnvlArray
#>  6
#> [ CPUf32{} ] 
#> 
try(g3(3))
#> Error in check_wrt_arrayish(args_flat, is_wrt_flat) : 
#>   Cannot compute gradient with respect to a value that has no data type.
#> ✖ It is an R double, which takes its data type from the way the function body
#>   uses it (see `?RData`).
#> ℹ Give it one first, e.g. `nv_array(x, dtype = "f32")` or an explicit
#>   `nv_array(x, dtype = "f64")`, so the gradient's data type is the caller's
#>   choice.

# the value of `f` together with its gradient
loss_fn <- function(x) sum(x^2L)
vg <- jit(value_and_gradient(loss_fn))
result <- vg(nv_array(c(3, 4), dtype = "f32"))
result$value
#> AnvlArray
#>  25
#> [ CPUf32{} ] 
result$grad
#> $x
#> AnvlArray
#>  6
#>  8
#> [ CPUf32{2} ] 
#> 
```
