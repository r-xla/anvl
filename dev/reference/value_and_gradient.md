# Value and Gradient

Returns a new function that computes both the output of `f` and its
gradient in a single forward+reverse pass. The result is a named list
with elements `value` (the original return value of `f`) and `grad` (the
gradients, structured like the inputs or the `wrt` subset).

## Usage

``` r
value_and_gradient(f, wrt = NULL)
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

(`function`)  
Has the same formals as `f` and returns `list(value = ..., grad = ...)`.

## See also

[`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md)

## Examples

``` r
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
