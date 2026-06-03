# Lower a graph to StableHLO

Converts a traced
[`AnvlGraph`](https://r-xla.github.io/anvl/reference/AnvlGraph.md) into
the StableHLO intermediate representation (IR). Each graph operation is
translated to its corresponding StableHLO op. The result can be
serialized to MLIR text via
[`stablehlo::repr()`](https://r-xla.github.io/stablehlo/reference/repr.html)
and subsequently compiled to an XLA executable with
[`pjrt::pjrt_compile()`](https://r-xla.github.io/pjrt/reference/pjrt_compile.html).

The rules for translating to stablehlo are stored in
`$rules[["stablehlo"]]` of the primitives.

This is a low-level function; most users should use
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md) or
[`xla()`](https://r-xla.github.io/anvl/reference/xla.md) instead.

## Usage

``` r
stablehlo(
  graph,
  constants_as_inputs = TRUE,
  env = NULL,
  donate = character(),
  platform = NULL
)
```

## Arguments

- graph:

  ([`AnvlGraph`](https://r-xla.github.io/anvl/reference/AnvlGraph.md))  
  The graph to lower (e.g. produced by
  [`trace_fn()`](https://r-xla.github.io/anvl/reference/trace_fn.md)).

- constants_as_inputs:

  (`logical(1)`)  
  If `TRUE` (default), constants are registered as inputs to the
  StableHLO function so they can be passed in at execution time. If
  `FALSE`, they are not added as inputs. Set to `FALSE` for closures.
  Note that `GraphLiteral`s are always inlined into the StableHLO
  function.

- env:

  (`HloEnv` \| `NULL`)  
  Optional environment for reusing variable mappings across nested
  function lowerings (e.g. for higher-order primitives like `nv_while`).

- donate:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of the arguments whose buffers should be donated. Donated
  buffers can be aliased with outputs of the same type, enabling
  in-place operations.

- platform:

  (`NULL` \| `character(1)`)  
  Target platform name (e.g. `"cpu"`, `"cuda"`). Stored on a
  process-wide global during the call so that platform-aware lowering
  rules (queried via
  [`current_platform()`](https://r-xla.github.io/anvl/reference/current_platform.md))
  can branch on it. `NULL` (the default) leaves the current value
  untouched — recursive calls from higher-order primitives inherit the
  platform of the enclosing call.

## Value

A `list` of length 2:

- the
  [`stablehlo::Func`](https://r-xla.github.io/stablehlo/reference/Func.html)

- The list of
  [`GraphValue`](https://r-xla.github.io/anvl/reference/GraphValue.md)s
  holding
  [`ConcreteArray`](https://r-xla.github.io/anvl/reference/ConcreteArray.md)s.

## See also

[`trace_fn()`](https://r-xla.github.io/anvl/reference/trace_fn.md),
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md),
[`xla()`](https://r-xla.github.io/anvl/reference/xla.md),
[`current_platform()`](https://r-xla.github.io/anvl/reference/current_platform.md)

## Examples

``` r
x <- nv_array(c(1, 2))
graph <- trace_fn(function(y) y + x, list(y = nv_aval("f32", shape = c())))
graph
#> <AnvlGraph>
#>   Inputs:
#>     %x1: f32[]
#>   Constants:
#>     %c1: f32[2]
#>   Body:
#>     %1: f32[2] = broadcast_in_dim [shape = 2, broadcast_dimensions = <any>] (%x1)
#>     %2: f32[2] = add(%1, %c1)
#>   Outputs:
#>     %2: f32[2] 
stablehlo(graph)
#> [[1]]
#> func.func @main (%0: tensor<2xf32>, %1: tensor<f32>) -> tensor<2xf32> {
#> %2 = "stablehlo.broadcast_in_dim" (%1) {
#> broadcast_dimensions = array<i64>
#> }: (tensor<f32>) -> (tensor<2xf32>)
#> %3 = stablehlo.add %2, %0 : tensor<2xf32>
#> return %3 : tensor<2xf32>
#> }
#> 
#> [[2]]
#> [[2]][[1]]
#> GraphValue(ConcreteArray(f32, (2))) 
#> 
#> 
```
