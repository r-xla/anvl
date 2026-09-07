# Add a Primitive Call to a Graph Descriptor

Add a primitive call to a graph descriptor. Inside a primitive body
created with
[`new_primitive()`](https://r-xla.github.io/anvl/dev/reference/new_primitive.md),
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

  ([`AnvlPrimitive`](https://r-xla.github.io/anvl/dev/reference/AnvlPrimitive.md)
  \| `JitPrimitive`)  
  The primitive the call is for. A `JitPrimitive` is accepted and
  unwrapped to its underlying `AnvlPrimitive` metadata.

- args:

  (`list` of
  [`GraphNode`](https://r-xla.github.io/anvl/dev/reference/GraphNode.md))  
  The arguments to the primitive.

- params:

  (`list`)  
  The parameters to the primitive.

- infer_fn:

  (`function`)  
  The inference function to use. Must output a list of
  [`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md)s.

- desc:

  ([`GraphDescriptor`](https://r-xla.github.io/anvl/dev/reference/GraphDescriptor.md)
  \| `NULL`)  
  The graph descriptor to add the primitive call to. Uses the [current
  descriptor](https://r-xla.github.io/anvl/dev/reference/dot-current_descriptor.md)
  if `NULL`.

- device:

  (`NULL` \| `character(1)` \| device object)  
  The device the call places its result on, for a primitive that
  constructs an array out of nothing (e.g.
  [`prim_fill()`](https://r-xla.github.io/anvl/dev/reference/prim_fill.md),
  [`prim_iota()`](https://r-xla.github.io/anvl/dev/reference/prim_iota.md))
  and so has no operand to carry one. It is declared to `desc`, where it
  counts like the device of an array input to the same trace: it decides
  what that program is compiled for, and disagreeing with another device
  in it is an error. Every other primitive takes its device from its
  operands and leaves this `NULL`.

## Value

(`list` of
[`GraphBox`](https://r-xla.github.io/anvl/dev/reference/GraphBox.md))
