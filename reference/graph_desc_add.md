# Add a Statement to a Graph Descriptor

Record a call of a primitive as a
[`GraphStatement`](https://r-xla.github.io/anvl/reference/GraphStatement.md)
in a graph descriptor. Inside a primitive body created with
[`new_primitive()`](https://r-xla.github.io/anvl/reference/new_primitive.md),
pass the lexically-bound `self` as the primitive argument.

## Usage

``` r
graph_desc_add(
  primitive,
  args,
  params = list(),
  infer_fn,
  desc = NULL,
  device = NULL
)
```

## Arguments

- primitive:

  ([`AnvlPrimitiveDef`](https://r-xla.github.io/anvl/reference/AnvlPrimitiveDef.md)
  \|
  [`AnvlPrimitive`](https://r-xla.github.io/anvl/reference/new_primitive.md))  
  The primitive the statement applies. An `AnvlPrimitive` is accepted
  and unwrapped to its underlying `AnvlPrimitiveDef`.

- args:

  (`list` of
  [`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  The arguments to the primitive:
  [`GraphBox`](https://r-xla.github.io/anvl/reference/GraphBox.md)es,
  [`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md)s
  (registered as constants of the graph) or R values (materialized at
  their default data type).

- params:

  (`list`)  
  The parameters to the primitive.

- infer_fn:

  (`function`)  
  The inference function to use. Must output a list of
  [`AbstractArray`](https://r-xla.github.io/anvl/reference/AbstractArray.md)s.

- desc:

  ([`GraphDescriptor`](https://r-xla.github.io/anvl/reference/GraphDescriptor.md)
  \| `NULL`)  
  The graph descriptor to add the statement to. Uses the [current
  descriptor](https://r-xla.github.io/anvl/reference/current_descriptor.md)
  if `NULL`.

- device:

  (`NULL` \| `character(1)` \| device object)  
  The device the call places its result on, for a primitive that
  constructs an array out of nothing (e.g.
  [`prim_fill()`](https://r-xla.github.io/anvl/reference/prim_fill.md),
  [`prim_iota()`](https://r-xla.github.io/anvl/reference/prim_iota.md))
  and so has no operand to carry one. It is declared to `desc`, where it
  counts like the device of an array input to the same trace: it decides
  what that program is compiled for, and disagreeing with another device
  in it is an error. Every other primitive takes its device from its
  operands and leaves this `NULL`.

## Value

(`list` of
[`GraphBox`](https://r-xla.github.io/anvl/reference/GraphBox.md))
