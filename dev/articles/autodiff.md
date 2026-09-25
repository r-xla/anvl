# Automatic Differentiation

In this article, you will learn how to compute gradients of functions
with {anvl}. If you haven’t yet, read the [Get
Started](https://r-xla.github.io/anvl/dev/articles/anvl.md) article
first, which uses gradients to fit a linear model.

## Computing Gradients

[`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md)
takes a function and returns a new function that computes its gradient.
Consider \\f(x) = \sum_i x_i^2\\, whose gradient is \\2x\\:

``` r

library(anvl)

f <- function(x) sum(x^2)
f_grad <- jit(gradient(f))

f_grad(nv_array(c(1, 2, 3)))
#> $x
#> AnvlArray
#>  2
#>  4
#>  6
#> [ CPUf32{3} ]
```

The gradient function has the same arguments as `f`, and returns a named
list with one gradient per argument. Each gradient has the same shape
and data type as its argument.

Note that the result of
[`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md)
must be jitted. This is because {anvl} computes gradients by
transforming the traced program of `f`, which only exists within
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md).

## Differentiable Functions

The function that is differentiated must return a single float scalar,
such as a loss. If you are interested in the derivative of a function
that returns an array, reduce it to a scalar first. For an elementwise
function such as [`sin()`](https://rdrr.io/r/base/Trig.html), the
gradient of `sum(sin(x))` is exactly the elementwise derivative
`cos(x)`:

``` r

jit(gradient(function(x) sin(x)))(nv_array(c(0, pi)))
#> Error in `validate_gradient_output()`:
#> ! gradient can only be computed for functions that return a scalar

jit(gradient(function(x) sum(sin(x))))(nv_array(c(0, pi)))
#> $x
#> AnvlArray
#>   1
#>  -1
#> [ CPUf32{2} ]
```

Gradients can only be computed with respect to float arrays, because the
derivative with respect to an integer or boolean is not defined.

``` r

jit(gradient(f))(nv_array(1:3))
#> Error in `check_wrt_arrayish()`:
#> ! Can only compute gradient with respect to float arrays.
#> ✖ Got "i32".
```

Bare R values cannot be differentiated with respect to either, even if
they are doubles. They only take their data type from how the function
uses them, so the data type of the gradient would not be determined by
the caller. Convert them to an `AnvlArray` first:

``` r

jit(gradient(f))(arr(1, 2, 3))
#> Error in `check_wrt_arrayish()`:
#> ! Cannot compute gradient with respect to a value that has no data type.
#> ✖ It is an R double, which takes its data type from the way the function body
#>   uses it (see `?RData`).
#> ℹ Give it one first, e.g. `nv_array(x, dtype = "f32")` or an explicit
#>   `nv_array(x, dtype = "f64")`, so the gradient's data type is the caller's
#>   choice.
jit(gradient(f))(nv_array(c(1, 2, 3)))
#> $x
#> AnvlArray
#>  2
#>  4
#>  6
#> [ CPUf32{3} ]
```

## Selecting Arguments with `wrt`

By default, the gradient is computed with respect to all arguments. The
`wrt` argument selects a subset of them, by name or position. This is
necessary when some arguments are not float arrays, e.g. data that is
not being optimized over, or static arguments:

``` r

power_sum <- function(x, p) sum(x^p)
power_sum_grad <- jit(gradient(power_sum, wrt = "x"), static = "p")

power_sum_grad(nv_array(c(1, 2)), p = 3L)
#> $x
#> AnvlArray
#>   3
#>  12
#> [ CPUf32{2} ]
```

## Nested Inputs

Arguments can also be (nested) lists of arrays, as is common for the
parameters of a model. The gradient then has the same structure as the
argument:

``` r

loss <- function(params, x, y) {
  y_hat <- x * params$w + params$b
  mean((y_hat - y)^2)
}
loss_grad <- jit(gradient(loss, wrt = "params"))

params <- list(w = nv_scalar(1), b = nv_scalar(0))
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(2, 4, 6))

loss_grad(params, x, y)
#> $params
#> $params$w
#> AnvlArray
#>  -9.3333
#> [ CPUf32{} ] 
#> 
#> $params$b
#> AnvlArray
#>  -4
#> [ CPUf32{} ]
```

This makes it easy to update all parameters at once with
[`pmap_tree()`](https://r-xla.github.io/pjrt/reference/pmap_tree.html),
which applies a function to the corresponding `AnvlArray`s of several
lists with the same structure.

``` r

grads <- loss_grad(params, x, y)$params
pmap_tree(list(params, grads), \(p, g) p - 0.1 * g)
#> $w
#> AnvlArray
#>  1.9333
#> [ CPUf32{} ] 
#> 
#> $b
#> AnvlArray
#>  0.4000
#> [ CPUf32{} ]
```

## Value and Gradient

When fitting a model, one usually needs both the value of the loss and
its gradient. Instead of calling the function and its gradient
separately,
[`value_and_gradient()`](https://r-xla.github.io/anvl/dev/reference/value_and_gradient.md)
computes both in a single pass:

``` r

loss_vg <- jit(value_and_gradient(loss, wrt = "params"))
loss_vg(params, x, y)
#> $value
#> AnvlArray
#>  4.6667
#> [ CPUf32{} ] 
#> 
#> $grad
#> $grad$params
#> $grad$params$w
#> AnvlArray
#>  -9.3333
#> [ CPUf32{} ] 
#> 
#> $grad$params$b
#> AnvlArray
#>  -4
#> [ CPUf32{} ]
```

## Composing with `jit()`

[`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md)
works on any function that can be traced, including jitted ones, and the
gradient function can itself be called within other jitted functions.
Because nested jitted functions are compiled together, it does not
matter whether you jit `f`, its gradient, or the function that calls the
gradient. Typically, you jit the outermost function, e.g. a complete
step of the model fitting:

``` r

fit_step <- jit(function(params, x, y, lr) {
  out <- value_and_gradient(loss, wrt = "params")(params, x, y)
  list(
    loss = out$value,
    params = pmap_tree(list(params, out$grad$params), \(p, g) p - lr * g)
  )
})

for (i in 1:3) {
  step <- fit_step(params, x, y, lr = 0.05)
  params <- step$params
  print(as_array(step$loss))
}
#> [1] 4.666667
#> [1] 0.9407408
#> [1] 0.201381
```

## Non-Differentiable Operations

Some operations have no meaningful derivative, e.g. comparisons. Where
they only select values, gradients flow through the selected values. For
example, in `nv_ifelse(x > 0, x, 0)`, the gradient is `1` where `x` is
positive and `0` elsewhere:

``` r

relu_sum <- function(x) sum(nv_ifelse(x > 0, x, 0))
jit(gradient(relu_sum))(nv_array(c(-1, 2)))
#> $x
#> AnvlArray
#>  0
#>  1
#> [ CPUf32{2} ]
```

Other functions, such as [`max()`](https://rdrr.io/r/base/Extremes.html)
or [`abs()`](https://rdrr.io/r/base/MathFun.html), are differentiable
everywhere except at a few points. At these points, {anvl} follows the
conventions of other frameworks such as PyTorch.

## Limitations

- {anvl} only implements *reverse-mode* differentiation, which is
  efficient for functions with many inputs and a single scalar output,
  such as losses. Forward mode differentiation is currently not
  supported.
- Higher-order derivatives, such as Hessians, are currently not really
  supported.
- Not every primitive has a derivative yet, so some functions cannot be
  differentiated. The *Reverse* column of the [Primitives
  Reference](https://r-xla.github.io/anvl/dev/articles/primitives.md)
  shows which primitives implement it. If a derivative is missing and
  you need it, you can add a reverse rule to the primitive yourself, as
  shown in the [add the reverse
  rule](https://r-xla.github.io/anvl/dev/articles/extending_primitive.html#step-3-add-the-reverse-rule)
  step of the Adding a Primitive article.

## How It Works

See the [transforming graphs into other
graphs](https://r-xla.github.io/anvl/dev/articles/internals.html#transforming-graphs-into-other-graphs)
and
[`gradient()`](https://r-xla.github.io/anvl/dev/articles/internals.html#gradient)
sections of the internals article for how {anvl} computes gradients, and
the [Adding a
Primitive](https://r-xla.github.io/anvl/dev/articles/extending_primitive.md)
article for how to write a reverse rule.
