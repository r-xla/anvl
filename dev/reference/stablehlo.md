# Lower a Graph to StableHLO

Converts a traced
[`AnvlGraph`](https://r-xla.github.io/anvl/dev/reference/AnvlGraph.md)
into the StableHLO intermediate representation (IR), building a
[`stablehlo::Func`](https://r-xla.github.io/stablehlo/reference/Func.html)
with the [stablehlo](https://r-xla.github.io/stablehlo/) package. Each
statement of the graph is translated to its corresponding StableHLO op.
The result can be serialized to MLIR text via
[`stablehlo::repr()`](https://r-xla.github.io/stablehlo/reference/repr.html)
and subsequently compiled to an XLA executable with
[`pjrt::pjrt_compile()`](https://r-xla.github.io/pjrt/reference/pjrt_compile.html).

The rule for translating a primitive to stablehlo is
`prim_<name>[["stablehlo"]]`.

The arguments of the resulting function are, in this order: the graph's
constants (when `constants_as_inputs = TRUE`), the graph's inputs, and
the phantom donated inputs (when `donate_unaliased_outputs = TRUE`), one
per output that is not already aliased to a donated input, in the order
of the outputs.

This is a low-level function; most users should use
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) instead.

## Usage

``` r
stablehlo(
  graph,
  id = "main",
  constants_as_inputs = TRUE,
  env = NULL,
  donate = character(),
  donate_unaliased_outputs = FALSE,
  platform = NULL
)
```

## Arguments

- graph:

  ([`AnvlGraph`](https://r-xla.github.io/anvl/dev/reference/AnvlGraph.md))  
  The graph to lower (e.g. produced by
  [`trace_fn()`](https://r-xla.github.io/anvl/dev/reference/trace_fn.md)).

- id:

  (`character(1)`)  
  The id of the resulting StableHLO function. Use `"main"` (the default)
  for a top-level lowering (returning from the `main` function finalizes
  the module) and `""` for a closure/region lowering (e.g. a while body
  or a scatter update computation) that builds an anonymous nested
  function inside an enclosing build.

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
  function lowerings (e.g. for higher-order primitives like
  `prim_while`).

- donate:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of the arguments whose buffers should be donated. Donated
  buffers can be aliased with outputs of the same type, enabling
  in-place operations.

- donate_unaliased_outputs:

  (`logical(1)`)  
  If `TRUE` and the current target platform is `"cpu"`, append a phantom
  donated input for every output that isn't already aliased to a
  user-`donate`d input. They come after all other arguments. This is
  needed internally so R keeps track of the CPU buffers memory in order
  to know when to garbage collect.

- platform:

  (`NULL` \| `character(1)`)  
  Target platform name (e.g. `"cpu"`, `"cuda"`). Stored on a
  process-wide global during the call so that platform-aware lowering
  rules (queried via
  [`current_platform()`](https://r-xla.github.io/anvl/dev/reference/current_platform.md))
  can branch on it. `NULL` (the default) leaves the current value
  untouched — recursive calls from higher-order primitives inherit the
  platform of the enclosing call.

## Value

(`list`)  
Of length 3:

- the
  [`stablehlo::Func`](https://r-xla.github.io/stablehlo/reference/Func.html)

- The graph's constants: the
  [`GraphValue`](https://r-xla.github.io/anvl/dev/reference/GraphValue.md)s
  holding
  [`ConcreteArray`](https://r-xla.github.io/anvl/dev/reference/ConcreteArray.md)s,
  whose data must be passed as the leading inputs at execution time when
  `constants_as_inputs = TRUE`.

- A list of phantom-output specs, one per phantom donated input appended
  when `donate_unaliased_outputs = TRUE`. Each entry is a
  `list(dtype, shape)` describing the buffer the executor must allocate.
  Empty when no phantoms were added.

## See also

[`trace_fn()`](https://r-xla.github.io/anvl/dev/reference/trace_fn.md),
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md),
[`current_platform()`](https://r-xla.github.io/anvl/dev/reference/current_platform.md)

## Examples

``` r
# the closed-over array `x` becomes the constant input %0, before the
# graph's own input `y` (%1)
x <- nv_array(c(1, 2))
graph <- trace_fn(function(y) y + x, list(y = nv_aval("f32", shape = 2)))
out <- stablehlo(graph)
out[[1L]]
#> func.func @main (%0: tensor<2xf32>, %1: tensor<2xf32>) -> tensor<2xf32> {
#> %2 = stablehlo.add %1, %0 : tensor<2xf32>
#> return %2 : tensor<2xf32>
#> }
out[[2L]]
#> [[1]]
#> GraphValue(ConcreteArray(f32, (2))) 
#> 

# a donated input is aliased with an output of the same type
graph <- trace_fn(
  function(a, b) list(a + b, a * b),
  list(a = nv_aval("f32", 3), b = nv_aval("f32", 3))
)
stablehlo(graph, donate = "a")[[1L]]
#> func.func @main (%0: tensor<3xf32> {tf.aliasing_output = 0 : i32}, %1: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
#> %2 = stablehlo.add %0, %1 : tensor<3xf32>
#> %3 = stablehlo.multiply %0, %1 : tensor<3xf32>
#> return %2, %3 : tensor<3xf32>, tensor<3xf32>
#> }

# on CPU, a phantom input is appended for every output not aliased yet, and
# the third element describes the buffers to allocate for them
out <- stablehlo(graph, donate_unaliased_outputs = TRUE, platform = "cpu")
out[[1L]]
#> func.func @main (%0: tensor<3xf32>, %1: tensor<3xf32>, %2: tensor<3xf32> {tf.aliasing_output = 0 : i32}, %3: tensor<3xf32> {tf.aliasing_output = 1 : i32}) -> (tensor<3xf32>, tensor<3xf32>) {
#> %4 = stablehlo.add %0, %1 : tensor<3xf32>
#> %5 = stablehlo.multiply %0, %1 : tensor<3xf32>
#> return %4, %5 : tensor<3xf32>, tensor<3xf32>
#> }
out[[3L]]
#> [[1]]
#> [[1]]$dtype
#> <f32>
#> 
#> [[1]]$shape
#> [1] 3
#> 
#> 
#> [[2]]
#> [[2]]$dtype
#> <f32>
#> 
#> [[2]]$shape
#> [1] 3
#> 
#> 
```
