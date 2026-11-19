# Pullbacks

Evaluate `.f` at the arguments `...` and return its pullback: the
function that takes a cotangent of the result and returns the
corresponding cotangent back to the arguments, by reverse-mode automatic
differentiation. That is the cotangent times the Jacobian of `.f`, the
vector-Jacobian product. Where
[`gradient()`](https://r-xla.github.io/anvl/reference/gradient.md)
differentiates a function that returns a single float scalar, a pullback
differentiates one with any result: it gives the gradient of the sum of
the result's elements, each weighted by its cotangent.

- `pullback()` returns only the pullback.

- `value_and_pullback()` returns both the result of `.f` and its
  pullback.

`.f` runs once, when `pullback()` is called. The pullback can be called
more than once, e.g. with every unit vector to build a Jacobian row by
row.

## Usage

``` r
pullback(.f, ...)

value_and_pullback(.f, ...)
```

## Arguments

- .f:

  (`function`)  
  Function to differentiate. It can return any (nested) list of arrays.

- ...:

  (any)  
  The arguments to evaluate `.f` at, matched to its formals as in a call
  of `.f`.

## Value

For `pullback()`, the pullback (`function`); for `value_and_pullback()`,
`list(value = , pullback = )`: the return value of `.f`, and its
pullback.

The pullback takes one argument, `cotangent`, structured like the value
of `.f`, with an array of the same shape and data type for each float
array of it; an R value takes the data type of its array. `NULL` in
place of an array – or of a whole part of the value – stands for zeros.
An array of the value that is not a float has no cotangent, and gets
`NULL`.

The pullback returns a named `list` with the cotangent of each argument,
structured like that argument, as
[`gradient()`](https://r-xla.github.io/anvl/reference/gradient.md) does.
An argument `.f` takes through its `...` keeps the name it was passed
under: name it in the call to read its cotangent by name. One that is
not a float array – an integer or boolean array, an R value such as `3`,
a random state – is held fixed, and gets `NULL` in place of a cotangent,
as does each such part of a list.

Like [`gradient()`](https://r-xla.github.io/anvl/reference/gradient.md),
`pullback()` must be called inside
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md). The pullback
must be called in the same function, or in a function traced inside it,
such as a branch of
[`nv_if()`](https://r-xla.github.io/anvl/reference/nv_if.md).

## See also

the [Automatic
Differentiation](https://r-xla.github.io/anvl/articles/autodiff.html)
article,
[`gradient()`](https://r-xla.github.io/anvl/reference/gradient.md)

## Examples

``` r
f <- function(x, y) x * y
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(4, 5, 6))
# cotangent * y and cotangent * x
g <- jit(function(x, y, ct) pullback(f, x, y)(ct))
g(x, y, nv_array(c(1, 0, 1)))
#> $x
#> AnvlArray
#>  4
#>  0
#>  6
#> [ CPUf32{3} ] 
#> 
#> $y
#> AnvlArray
#>  1
#>  0
#>  3
#> [ CPUf32{3} ] 
#> 

# the value of `f` together with its pullback
h <- jit(function(x, y, ct) {
  out <- value_and_pullback(f, x, y)
  list(value = out$value, x = out$pullback(ct)$x)
})
h(x, y, nv_array(c(1, 0, 1)))
#> $value
#> AnvlArray
#>   4
#>  10
#>  18
#> [ CPUf32{3} ] 
#> 
#> $x
#> AnvlArray
#>  4
#>  0
#>  6
#> [ CPUf32{3} ] 
#> 

# a result with several parts: `NULL` is the cotangent of one that does not
# matter
stats <- function(x) list(total = sum(x), squares = x^2L)
dtotal <- jit(function(x) pullback(stats, x)(list(total = 1, squares = NULL)))
dtotal(x)
#> $x
#> AnvlArray
#>  1
#>  1
#>  1
#> [ CPUf32{3} ] 
#> 
```
