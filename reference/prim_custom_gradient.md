# Primitive Custom Gradient

Calls `fn` on `args`, but differentiates the result with `rule` instead
of through the operations `fn` performs. Use it where those operations
give no gradient – e.g. a
[`prim_while()`](https://r-xla.github.io/anvl/reference/prim_while.md)
loop, which
[`gradient()`](https://r-xla.github.io/anvl/reference/gradient.md)
cannot differentiate – or an inaccurate one, while the derivative has a
formula of its own.

## Usage

``` r
prim_custom_gradient(args, fn, rule)
```

## Arguments

- args:

  (named [`list()`](https://rdrr.io/r/base/list.html))  
  The arguments of `fn`:
  [`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)
  values, or trees of them in the sense of pjrt's
  [`RTree`](https://r-xla.github.io/pjrt/reference/build_tree.html), by
  name. `rule` gives the gradient with respect to these. No entry may be
  named `out`, `grad` or `required`, the names under which `rule`
  receives the result, its gradient and which gradients are required. An
  R value materializes at its default data type, as in
  [`prim_while()`](https://r-xla.github.io/anvl/reference/prim_while.md).

- fn:

  (`function`)  
  Computes the result from the entries of `args`, which it receives as
  arguments. Returns an array, or a tree of them.

- rule:

  (`function` \| `anvl_rule_reverse`)  
  Computes the gradient. A function receives the entries of `args` as
  arguments, plus `out`, the result of `fn`, `grad`, the gradient with
  respect to `out`, of the same structure, data types and shapes, and
  `required`, a named [`logical()`](https://rdrr.io/r/base/logical.html)
  with one element per entry of `args`: `TRUE` where the gradient with
  respect to that entry is required, i.e. where any of its arrays
  requires one. Returns a named list with the gradient with respect to
  each required entry – for an array `x`, the sum over the elements of
  `out` of their `grad` times their derivative with respect to `x` – of
  `x`'s structure, data types and shapes. An entry that is not required,
  e.g. an integer, which never is, can be left out or be `NULL`.

  Alternatively, a
  [`rule_reverse()`](https://r-xla.github.io/anvl/reference/rule_reverse.md)
  with a `forward` pass, which replaces `fn` where the result is
  differentiated, so that the backward pass can read what the forward
  pass computed. Its passes are those of a primitive's rule, except that
  `inputs` are the entries of `args` by name rather than one array per
  operand, `params` is an empty list, and `outputs`, `grads`, `required`
  and the returned gradients are those of a function above.

## Value

([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md) \|
`list`)  
The result of `fn`.

## Gradients

`rule` is traced when the gradient is, once the function being
differentiated has been traced. It can close over constants – R values
and arrays – but not over a traced value: pass that through `args`
instead.

`fn` may close over traced values, but
[`gradient()`](https://r-xla.github.io/anvl/reference/gradient.md)
cannot differentiate with respect to those, as `rule` only gives the
gradient with respect to `args`. Pass such a value through `args`
instead.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers the operations of `fn` in place. `rule` is not lowered: a
gradient is traced, as anvl operations, before the program is lowered.

## See also

[`gradient()`](https://r-xla.github.io/anvl/reference/gradient.md)

## Examples

``` r
# x^n by repeated multiplication: a while loop, which gradient() cannot
# differentiate on its own
pow_loop <- function(x, n) {
  prim_while(
    init = list(i = 1L, y = x),
    cond = function(i, y) i < n,
    body = function(i, y) list(i = i + 1L, y = y * x)
  )$y
}
pow <- function(x, n) {
  prim_custom_gradient(
    list(x = x, n = n),
    fn = pow_loop,
    rule = function(x, n, out, grad, required) list(x = grad * n * x^(n - 1L))
  )
}
x <- nv_scalar(2)
pow(x, 3L)
#> AnvlArray
#>  8
#> [ CPUf32{} ] 
# the derivative 3 * x^2
jit(gradient(pow, wrt = "x"))(x, 3L)
#> $x
#> AnvlArray
#>  12
#> [ CPUf32{} ] 
#> 
# and, as it is computed from x, the second derivative 6 * x
dpow <- function(x, n) gradient(pow, wrt = "x")(x, n)$x
jit(gradient(dpow, wrt = "x"))(x, 3L)
#> $x
#> AnvlArray
#>  12
#> [ CPUf32{} ] 
#> 

# With a forward pass, x^(n - 1) is computed once, for the result and for
# its gradient
pow_fwd <- function(x, n) {
  prim_custom_gradient(
    list(x = x, n = n),
    fn = pow_loop,
    rule = rule_reverse(forward = function(inputs, params, required) {
      p <- inputs$x^(inputs$n - 1L)
      list(
        outputs = p * inputs$x,
        backward = function(inputs, outputs, grads, params, required) {
          list(x = grads * inputs$n * p)
        }
      )
    })
  )
}
jit(gradient(pow_fwd, wrt = "x"))(x, 3L)
#> $x
#> AnvlArray
#>  12
#> [ CPUf32{} ] 
#> 
```
