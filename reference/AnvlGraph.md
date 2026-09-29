# Graph of Statements

Computational graph consisting exclusively of statements that apply
primitives. This is a mutable class.

An `AnvlGraph` is usually created by tracing a function with
[`trace_fn()`](https://r-xla.github.io/anvl/reference/trace_fn.md),
which records each primitive call as a
[`GraphStatement`](https://r-xla.github.io/anvl/reference/GraphStatement.md)
into a
[`GraphDescriptor`](https://r-xla.github.io/anvl/reference/GraphDescriptor.md)
and converts it into an `AnvlGraph` at the end. The graph is then
lowered, e.g. with
[`stablehlo()`](https://r-xla.github.io/anvl/reference/stablehlo.md),
and compiled.

## Usage

``` r
AnvlGraph(
  statements = list(),
  in_tree = NULL,
  out_tree = NULL,
  inputs = list(),
  outputs = list(),
  constants = list(),
  is_static_flat = NULL,
  static_args_flat = NULL,
  rdata_types = NULL
)
```

## Arguments

- statements:

  (`list(GraphStatement)`)  
  The statements that make up the graph.

- in_tree:

  (`NULL` \|
  [`RTree`](https://r-xla.github.io/pjrt/reference/build_tree.html))  
  The tree of inputs. May contain leaves for both array inputs and
  static (non-array) arguments. Only the array leaves correspond to
  entries in `inputs`; use `is_static_flat` to distinguish them.

- out_tree:

  (`NULL` \|
  [`RTree`](https://r-xla.github.io/pjrt/reference/build_tree.html))  
  The tree of outputs.

- inputs:

  (`list(GraphValue)`)  
  The inputs to the graph (array arguments only).

- outputs:

  (`list(GraphValue)`)  
  The outputs of the graph.

- constants:

  (`list(GraphValue)`)  
  The constants of the graph.

- is_static_flat:

  (`NULL | logical()`)  
  Boolean mask indicating which flat positions in `in_tree` are static
  (non-array) args. `NULL` when all args are array inputs.

- static_args_flat:

  (`NULL | list()`)  
  Flattened traced values for the static arguments indicated by
  `is_static_flat`.

- rdata_types:

  (`NULL | character()`)  
  One entry per input: the R storage type of an input the caller
  supplies as bare R data (`"double"`, `"integer"`, `"logical"`), and
  `NA` for one that arrives as an array and already has a data type.
  `NULL` when no input comes from R data, which is the common case.
  Together with the inputs' own avals this says everything about how a
  call's arguments are uploaded: the aval gives the data type and shape,
  this gives the R type it is uploaded from.

## Value

(`AnvlGraph`)

## Examples

``` r
# the inputs are %x1 and %x2; each line is one statement, and `sum`
# shows its parameters in brackets
graph <- trace_fn(function(x, y) {
  nv_sum(x * y)
}, list(x = nv_aval("f32", c(2, 3)), y = nv_aval("f32", c(2, 3))))
graph
#> <AnvlGraph> (%x1: f32[2,3], %x2: f32[2,3]) {
#>   %1: f32[2,3] = mul(%x1, %x2)
#>   %2: f32[] = sum [axes = c(1, 2), drop = TRUE] (%1)
#>   return %2
#> }
graph$inputs
#> [[1]]
#> GraphValue(AbstractArray(dtype=f32, shape=2x3)) 
#> 
#> [[2]]
#> GraphValue(AbstractArray(dtype=f32, shape=2x3)) 
#> 
graph$outputs
#> [[1]]
#> GraphValue(AbstractArray(dtype=f32, shape=)) 
#> 

# an array the function closes over becomes the constant %c1, and the R
# value `2` the literal `2:f32`
w <- nv_array(c(1, 2), dtype = "f32")
graph <- trace_fn(function(x) x + w * 2, list(x = nv_aval("f32", 2)))
graph
#> <AnvlGraph> [%c1: f32[2]] (%x1: f32[2]) {
#>   %1: f32[2] = broadcast_in_axes [
#>     shape = 2, broadcast_axes = integer(0)
#>   ] (2:f32)
#>   %2: f32[2] = mul(%c1, %1)
#>   %3: f32[2] = add(%x1, %2)
#>   return %3
#> }
graph$constants
#> [[1]]
#> GraphValue(ConcreteArray(f32, (2))) 
#> 

# several outputs are returned together; `out_tree` records their structure
graph <- trace_fn(function(x) list(a = x, b = nv_exp(x)), list(x = nv_aval("f32", c())))
graph
#> <AnvlGraph> (%x1: f32[]) {
#>   %1: f32[] = exp(%x1)
#>   return (%x1, %1)
#> }
graph$out_tree
#> list<named>(a = *, b = *)
```
