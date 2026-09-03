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
  Scalar boolean predicate that determines which branch to execute.

- true, false:

  (`function()`)  
  Zero-argument functions for the true and false branches. Both must
  return outputs with the same structure, dtypes, and shapes.

## Value

Result of the executed branch.  

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_if()`](https://r-xla.github.io/stablehlo/reference/hlo_if.html).

## See also

[`nv_if()`](https://r-xla.github.io/anvl/dev/reference/nv_if.md),
[`prim_ifelse()`](https://r-xla.github.io/anvl/dev/reference/prim_ifelse.md)

## Examples

``` r
prim_if(nv_scalar(TRUE), \() nv_scalar(1), \() nv_scalar(2))
#> AnvlArray
#>  1
#> [ CPUf32{} ] 
```
