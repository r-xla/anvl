# JIT-compile and evaluate an expression

Convenience wrapper that JIT-compiles and immediately evaluates a single
expression. Equivalent to wrapping `expr` in an anonymous function,
calling [`jit()`](https://r-xla.github.io/anvl/reference/jit.md) on it,
and invoking the result. Useful if you want to evaluate an expression
once.

## Usage

``` r
jit_eval(expr, ...)
```

## Arguments

- expr:

  (NSE)  
  Expression to compile and evaluate.

- ...:

  Backend-specific options forwarded to
  [`jit()`](https://r-xla.github.io/anvl/reference/jit.md) (e.g.
  `device` for the `"pjrt"` backend, `unwrap` for the `"quickr"`
  backend).

## Value

(`any`)  
Result of the compiled and evaluated expression.

## Examples

``` r
x <- nv_array(c(1, 2, 3), dtype = "f32")
jit_eval(x + x)
#> AnvlArray
#>  2
#>  4
#>  6
#> [ CPUf32{3} ] 
```
