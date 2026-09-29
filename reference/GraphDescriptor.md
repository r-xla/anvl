# Graph Descriptor

The in-progress representation of an
[`AnvlGraph`](https://r-xla.github.io/anvl/reference/AnvlGraph.md)
during tracing. This is a mutable class.

While [`trace_fn()`](https://r-xla.github.io/anvl/reference/trace_fn.md)
runs a function, every primitive call is recorded as a
[`GraphStatement`](https://r-xla.github.io/anvl/reference/GraphStatement.md)
into the current descriptor (see
[`graph_desc_add()`](https://r-xla.github.io/anvl/reference/graph_desc_add.md)
and
[`current_descriptor()`](https://r-xla.github.io/anvl/reference/current_descriptor.md)).
The descriptor also does the book-keeping the trace needs: which
[`GraphValue`](https://r-xla.github.io/anvl/reference/GraphValue.md) an
`AnvlArray` or
[`GraphBox`](https://r-xla.github.io/anvl/reference/GraphBox.md) stands
for, the constants and devices encountered, and the default data types
the trace is pinned to. Once tracing finishes, it is converted to an
[`AnvlGraph`](https://r-xla.github.io/anvl/reference/AnvlGraph.md),
which only keeps what is needed to lower and run the program.

## Usage

``` r
GraphDescriptor(
  statements = list(),
  array_to_gval = NULL,
  gval_to_box = NULL,
  constants = list(),
  in_tree = NULL,
  out_tree = NULL,
  inputs = list(),
  outputs = list(),
  is_static_flat = NULL,
  static_args_flat = NULL,
  devices = character(),
  default_dtypes = NULL,
  backend
)
```

## Arguments

- statements:

  (`list(GraphStatement)`)  
  The statements that make up the graph.

- array_to_gval:

  (`hashtab`)  
  Mapping: `AnvlArray` -\> `GraphValue`

- gval_to_box:

  (`hashtab`)  
  Mapping: `GraphValue` -\> `GraphBox`

- constants:

  (`list(GraphValue)`)  
  The constants of the graph.

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

- is_static_flat:

  (`NULL | logical()`)  
  Boolean mask indicating which flat positions in `in_tree` are static
  (non-array) args. `NULL` when all args are array inputs.

- static_args_flat:

  (`NULL | list()`)  
  Flattened traced values for the static arguments indicated by
  `is_static_flat`.

- devices:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Devices encountered during tracing: the device of every concrete array
  registered in the graph, plus the ones declared by
  [`graph_desc_add()`](https://r-xla.github.io/anvl/reference/graph_desc_add.md).

- default_dtypes:

  (`NULL` \| `list(float, int)`)  
  The data types every R value in this trace materializes at when
  nothing else decides one (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/reference/default_dtypes.md)).

- backend:

  (`character(1)`)  
  The backend this trace is compiled for. Required: it decides which
  entry of the `anvl.default_dtypes` option applies to the trace, so
  switching the active backend inside a traced body changes nothing.
  [`local_descriptor()`](https://r-xla.github.io/anvl/reference/local_descriptor.md)
  fills it in from
  [`active_backend()`](https://r-xla.github.io/anvl/reference/active_backend.md),
  so only a direct call has to name it.

## Value

(`GraphDescriptor`)
