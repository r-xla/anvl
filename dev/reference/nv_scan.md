# Scan (Loop With Per-Step Outputs)

Runs a fixed-length loop that threads a carry through `body` while
stacking each step's output into preallocated buffers.

At step `t`, `body` receives the current carry and the step's slice of
`xs` (taken along axis 1, with that unit axis dropped; a 1-D leaf yields
a scalar), and must return
`list(carry = <same structure as init>, out = <arrays to stack>)`. The
stacked `out` buffers gain a new leading axis of size `steps`.

The whole loop, written out in R:

    carry <- init
    out <- <empty, `steps` rows>
    order <- if (reverse) rev(seq_len(steps)) else seq_len(steps)
    for (t in order) {
      step <- body(carry, xs[t, ...])  # `x` is NULL when `xs` is empty
      carry <- step$carry
      out[t, ...] <- step$out          # position t, not the loop's position
    }
    list(carry = carry, out = out)

## Usage

``` r
nv_scan(init, xs = NULL, body, steps = NULL, reverse = FALSE)
```

## Arguments

- init:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
  \| [`list()`](https://rdrr.io/r/base/list.html))  
  Initial carry: a single array or a (possibly nested) named list. Every
  slot must keep a fixed shape and dtype across steps.

- xs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
  \| [`list()`](https://rdrr.io/r/base/list.html) \| `NULL`)  
  Per-step inputs, sliced along axis 1. All leaves must agree on the
  size of axis 1. `NULL` or a list with no leaves runs a counted loop
  over `steps` steps instead.

- body:

  (`function`)  
  Step function `function(carry, x)` returning `list(carry = , out = )`.
  `out` may be a single array, a (nested) list of arrays, or `NULL`
  (loop for the carry only). Its structure must be identical at every
  step. `x` is `NULL` when `xs` is empty.

- steps:

  (`integer(1)` \| `NULL`)  
  Static trip count. Required when `xs` is empty; otherwise inferred
  from (and checked against) axis 1 of `xs`. A trip count of `0` runs no
  step.

- reverse:

  (`logical(1)`)  
  If `TRUE`, steps run `t = steps, ..., 1`; each step still reads `xs`
  at position `t` and writes its output at position `t`, so a reverse
  scan consumes and produces arrays in the original order.

## Value

`list(carry = , out = )`: the final carry (same structure as `init`) and
the stacked outputs (structure of `body`'s `out`, each leaf gaining a
leading axis of size `steps`).

## See also

[`prim_scan()`](https://r-xla.github.io/anvl/dev/reference/prim_scan.md),
[`nv_while()`](https://r-xla.github.io/anvl/dev/reference/nv_while.md),
[`nv_cumsum()`](https://r-xla.github.io/anvl/dev/reference/nv_cumsum.md)
for fixed associative scans.

## Examples

``` r
# cumulative sum along axis 1
x <- nv_array(c(1, 2, 3, 4))
nv_scan(
  init = nv_scalar(0),
  xs = x,
  body = function(carry, x) list(carry = carry + x, out = carry + x)
)$out
#> AnvlArray
#>   1
#>   3
#>   6
#>  10
#> [ CPUf32{4} ] 
```
