# Trace an R function into a Graph

Executes `f` with abstract array arguments and records every primitive
operation into an
[`AnvlGraph`](https://r-xla.github.io/anvl/reference/AnvlGraph.md).

The resulting graph can be lowered to StableHLO (via
[`stablehlo()`](https://r-xla.github.io/anvl/reference/stablehlo.md)) or
transformed (e.g. via
[`transform_gradient()`](https://r-xla.github.io/anvl/reference/transform_gradient.md)).

## Usage

``` r
trace_fn(
  f,
  args = NULL,
  desc = NULL,
  mode = NULL,
  args_flat = NULL,
  in_tree = NULL,
  optimize = FALSE
)
```

## Arguments

- f:

  (`function`)  
  The function to trace. Must not be a `JitFunction` (i.e. already
  jitted).

- args:

  (`list` of
  ([`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md) \|
  [`AbstractArray`](https://r-xla.github.io/anvl/reference/AbstractArray.md)))  
  The (unflattened) arguments to the function. Mutually exclusive with
  the `args_flat`/`in_tree` pair.

- desc:

  (`NULL` \| `GraphDescriptor`)  
  Optional descriptor. When `NULL` (default), a new descriptor is
  created.

- mode:

  (`character(1)`)  
  How to handle the inputs. Options are:

  - `"toplevel"`: Used for jit(). Default.

  - `"subgraph"`: Use for tracing subgraphs in higher-order primitives
    like
    [`prim_while()`](https://r-xla.github.io/anvl/reference/prim_while.md).

  - `"inline"`: Use for transformations like jit, where the graph is
    later inlined into the parent graph.

- args_flat:

  (`list`)  
  Flattened arguments. Must be accompanied by `in_tree`.

- in_tree:

  (`Node`)  
  Tree structure describing how `args_flat` maps back to `f`'s
  arguments.

- optimize:

  (`logical(1)` \|
  [`character()`](https://rdrr.io/r/base/character.html))  
  Which graph optimization passes to run on the traced graph before
  returning it. `TRUE` runs all passes, `FALSE` (default) runs none, and
  a character vector selects a subset by name. The available passes are:

  - `"inline_scalars"`: replace scalar-shaped constants with inline
    literals.

  - `"remove_unused_constants"`: drop constants not referenced by the
    graph.

  [`jit()`](https://r-xla.github.io/anvl/reference/jit.md) always traces
  with all passes enabled.

## Value

An [`AnvlGraph`](https://r-xla.github.io/anvl/reference/AnvlGraph.md)
containing the traced operations.

## See also

[`stablehlo()`](https://r-xla.github.io/anvl/reference/stablehlo.md) to
lower the graph,
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md) for end-to-end
compilation.

## Examples

``` r
graph <- trace_fn(function(x, y) x + y,
  args = list(x = nv_array(1, dtype = "f32"), y = nv_array(2, dtype = "f32")))
graph
#> <AnvlGraph>
#>   Inputs:
#>     %x1: f32[1]
#>     %x2: f32[1]
#>   Body:
#>     %1: f32[1] = add(%x1, %x2)
#>   Outputs:
#>     %1: f32[1] 
```
