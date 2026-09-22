# Primitive If

Conditional execution of one of two branches based on a scalar boolean
predicate. Unlike
[`prim_ifelse()`](https://r-xla.github.io/anvl/dev/reference/prim_ifelse.md)
which operates element-wise, this evaluates only the selected branch.

## Usage

``` r
prim_if(pred, true, false)
```

## Arguments

- pred:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Predicate deciding which branch to execute. Must be a scalar of the
  boolean data type, or an R logical.

- true, false:

  (`function()`)  
  Zero-argument functions for the true and false branches. Both must
  return outputs of the same structure, data types and shapes. As with
  [`prim_ifelse()`](https://r-xla.github.io/anvl/dev/reference/prim_ifelse.md),
  whose two values must already agree, nothing is promoted: branches
  that disagree are an error.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md) \|
`list`)  
Result of the executed branch: an array, or a tree of them in the sense
of pjrt's
[`RTree`](https://r-xla.github.io/pjrt/reference/build_tree.html) – a
`list`, nested arbitrarily – with the structure, data types and shapes
both branches share.

## Implemented Rules

- `stablehlo`

- `quickr`

## StableHLO

Lowers to
[`hlo_if()`](https://r-xla.github.io/stablehlo/reference/hlo_if.html),
specified under [if](https://openxla.org/stablehlo/spec#if).

## See also

[`nv_if()`](https://r-xla.github.io/anvl/dev/reference/nv_if.md),
[`prim_ifelse()`](https://r-xla.github.io/anvl/dev/reference/prim_ifelse.md)

## Examples

``` r
# both branches must return the same structure, data types and shapes
prim_if(nv_scalar(TRUE), \() nv_scalar(1), \() nv_scalar(2))
#> AnvlArray
#>  1
#> [ CPUf32{} ] 
```
