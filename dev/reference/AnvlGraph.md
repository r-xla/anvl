# Graph of Primitive Calls

Computational graph consisting exclusively of primitive calls. This is a
mutable class.

## Usage

``` r
AnvlGraph(
  calls = list(),
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

- calls:

  (`list(PrimitiveCall)`)  
  The primitive calls that make up the graph.

- in_tree:

  (`NULL | Node`)  
  The tree of inputs. May contain leaves for both array inputs and
  static (non-array) arguments. Only the array leaves correspond to
  entries in `inputs`; use `is_static_flat` to distinguish them.

- out_tree:

  (`NULL | Node`)  
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
  Together with the inputs\\ own avals this says everything about how a
  call\\s arguments are uploaded: the aval gives the data type and
  shape, this gives the R type it is uploaded from.

## Value

(`AnvlGraph`)
