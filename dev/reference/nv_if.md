# Conditional Branching

Conditional execution of two branches, mirroring R's `if` construct: it
branches between two *functions* and evaluates only the selected one.
Its arguments are named after that construct, where
[`nv_ifelse()`](https://r-xla.github.io/anvl/dev/reference/nv_ifelse.md)
– which selects element-wise between two *arrays* – is named after
[`ifelse()`](https://rdrr.io/r/base/ifelse.html).

## Usage

``` r
nv_if(pred, true, false)
```

## Arguments

- pred:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Predicate. Must be a scalar of the boolean data type, or an R logical.

- true:

  (`function()`)  
  Zero-argument function for the true branch.

- false:

  (`function()`)  
  Zero-argument function for the false branch. Must return the same
  structure, data types and shapes as the true branch; nothing is
  promoted.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md) \|
`list`)  
Result of the executed branch: an array, or a tree of them in the sense
of pjrt's
[`RTree`](https://r-xla.github.io/pjrt/reference/build_tree.html) – a
`list`, nested arbitrarily – with the structure, data types and shapes
both branches share.

## Gradients

[`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md)
differentiates through the branch the predicate selects, including the
values the branches close over; a value only the other branch uses gets
a zero. See
[`prim_if()`](https://r-xla.github.io/anvl/dev/reference/prim_if.md).

## See also

[`prim_if()`](https://r-xla.github.io/anvl/dev/reference/prim_if.md) for
the underlying primitive,
[`nv_ifelse()`](https://r-xla.github.io/anvl/dev/reference/nv_ifelse.md)
for element-wise selection.

## Examples

``` r
# both branches must return the same structure, data types and shapes
nv_if(nv_scalar(TRUE), \() nv_scalar(1), \() nv_scalar(2))
#> AnvlArray
#>  1
#> [ CPUf32{} ] 

# the gradient flows through the branch that is taken
f <- function(p, x) nv_if(p, \() sum(x * x), \() sum(x))
jit(gradient(f, wrt = "x"))(nv_scalar(TRUE), nv_array(c(1, 2, 3)))
#> $x
#> AnvlArray
#>  2
#>  4
#>  6
#> [ CPUf32{3} ] 
#> 
```
