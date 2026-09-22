# While Loop

Executes a functional while loop.

## Usage

``` r
nv_while(init, cond, body)
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
  Condition function returning a scalar boolean. Receives the state
  values as arguments.

- body:

  (`function`)  
  Body function returning the updated state as a named list with the
  same structure, data types and shapes as `init`. Nothing is promoted:
  a loop-carried state is meant to be heterogeneous, so each member
  keeps its own data type across iterations.

## Value

(named `list`)  
A tree of the loop-carried arrays – see
[`RTree`](https://r-xla.github.io/pjrt/reference/build_tree.html) – in
its final state after the loop terminates, with `init`'s structure, data
types and shapes.

## See also

[`prim_while()`](https://r-xla.github.io/anvl/dev/reference/prim_while.md)
for the underlying primitive.

## Examples

``` r
# the loop state is a named list, and each member keeps its data type
nv_while(
  init = list(i = nv_scalar(0L), total = nv_scalar(0L)),
  cond = function(i, total) i < 5L,
  body = function(i, total) list(
    i = i + 1L,
    total = total + i
  )
)
#> $i
#> AnvlArray
#>  5
#> [ CPUi32{} ] 
#> 
#> $total
#> AnvlArray
#>  10
#> [ CPUi32{} ] 
#> 
```
