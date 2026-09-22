# Primitive While Loop

Repeatedly executes `body` while `cond` returns `TRUE`, like R's `while`
loop. The loop state is initialized with `init` and passed through each
iteration; it is the only thing carried from one iteration to the next.

## Usage

``` r
prim_while(init, cond, body)
```

## Arguments

- init:

  (`named list()`)  
  Named list of initial state values, a tree in the sense of pjrt's
  [`RTree`](https://r-xla.github.io/pjrt/reference/build_tree.html).
  Each leaf becomes a parameter of the loop's sub-graphs. R values are
  materialized at their [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- cond:

  (`function`)  
  Condition function that receives the current state as arguments and
  outputs whether to continue the loop.

- body:

  (`function`)  
  Body function that receives the current state as arguments and returns
  a named list with the same structure, data types and shapes as `init`.
  Nothing is promoted: a loop-carried state is meant to be
  heterogeneous, so each member keeps its own data type across
  iterations.

## Value

(named `list`)  
A tree of the loop-carried arrays – see
[`RTree`](https://r-xla.github.io/pjrt/reference/build_tree.html) – with
the same structure, data types and shapes as `init`, in its final state
after the loop terminates.

## Implemented Rules

- `stablehlo`

- `quickr`

## StableHLO

Lowers to
[`hlo_while()`](https://r-xla.github.io/stablehlo/reference/hlo_while.html),
specified under [while](https://openxla.org/stablehlo/spec#while).

## See also

[`nv_while()`](https://r-xla.github.io/anvl/dev/reference/nv_while.md)

## Examples

``` r
# the loop state is a named list, and each member keeps its data type
prim_while(
  init = list(i = nv_scalar(0L), total = nv_scalar(0L)),
  cond = function(i, total) i <= 5L,
  body = function(i, total) list(
    i = i + 1L,
    total = total + i
  )
)
#> $i
#> AnvlArray
#>  6
#> [ CPUi32{} ] 
#> 
#> $total
#> AnvlArray
#>  15
#> [ CPUi32{} ] 
#> 
```
