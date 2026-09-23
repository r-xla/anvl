# Primitive Scan

Runs `body` a fixed number of times, threading a carry through the steps
and stacking each step's outputs along a new leading axis. Step `t`
receives the carry and, for every array in `xs`, its slice at position
`t` along axis 1 with that axis dropped.

## Usage

``` r
prim_scan(init, xs, body, steps, reverse = FALSE)
```

## Arguments

- init:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Initial carry: a (possibly nested) list of arrays. Every leaf keeps
  its shape and data type across steps.

- xs:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Per-step inputs: a (possibly nested) list of arrays sliced along axis
  1, all of size `steps` along it. An empty list runs a counted loop.

- body:

  (`function`)  
  Step function `function(carry, x)` returning `list(carry = , out = )`,
  where `carry` has the structure of `init` and `out` is a (possibly
  nested) list of arrays or `NULL`. `x` is `NULL` when `xs` is empty.

- steps:

  (`integer(1)`)  
  Static trip count; the size of axis 1 of every array in `xs`. `0` runs
  no step and returns `init` with zero-length stacked outputs.

- reverse:

  (`logical(1)`)  
  If `TRUE`, steps run from `steps` down to `1`; each step still reads
  `xs` at its own position and writes its output there.

## Value

`list(carry = , out = )`: the final carry and the stacked outputs, each
leaf of `out` gaining a leading axis of size `steps`.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`hlo_while()`](https://r-xla.github.io/stablehlo/reference/hlo_while.html)
over a counter, the carry, the output buffers and `xs`, with
[`hlo_dynamic_slice()`](https://r-xla.github.io/stablehlo/reference/hlo_dynamic_slice.html)
reading each step's inputs and
[`hlo_dynamic_update_slice()`](https://r-xla.github.io/stablehlo/reference/hlo_dynamic_update_slice.html)
writing its outputs.

## See also

[`nv_scan()`](https://r-xla.github.io/anvl/dev/reference/nv_scan.md),
[`prim_while()`](https://r-xla.github.io/anvl/dev/reference/prim_while.md)

## Examples

``` r
prim_scan(
  init = list(s = nv_scalar(0)),
  xs = list(x = nv_array(c(1, 2, 3))),
  body = function(carry, x) {
    s <- carry$s + x$x
    list(carry = list(s = s), out = s)
  },
  steps = 3L
)
#> $carry
#> $carry$s
#> AnvlArray
#>  6
#> [ CPUf32{} ] 
#> 
#> 
#> $out
#> AnvlArray
#>  1
#>  3
#>  6
#> [ CPUf32{3} ] 
#> 
```
