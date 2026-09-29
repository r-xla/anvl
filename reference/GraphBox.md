# Graph Box

Wraps a
[`GraphNode`](https://r-xla.github.io/anvl/reference/GraphNode.md)
during graph construction (tracing). When a function is traced via
[`trace_fn()`](https://r-xla.github.io/anvl/reference/trace_fn.md), each
intermediate array value is represented as a `GraphBox`. It also
contains an associated
[`GraphDescriptor`](https://r-xla.github.io/anvl/reference/GraphDescriptor.md)
in which the node "lives".

## Usage

``` r
GraphBox(gnode, desc)
```

## Arguments

- gnode:

  ([`GraphNode`](https://r-xla.github.io/anvl/reference/GraphNode.md))  
  The graph node – either a
  [`GraphValue`](https://r-xla.github.io/anvl/reference/GraphValue.md)
  or a
  [`GraphLiteral`](https://r-xla.github.io/anvl/reference/GraphLiteral.md).

- desc:

  ([`GraphDescriptor`](https://r-xla.github.io/anvl/reference/GraphDescriptor.md))  
  The descriptor of the graph being built.

## Value

(`GraphBox`)

## See also

[`trace_fn()`](https://r-xla.github.io/anvl/reference/trace_fn.md),
[`jit()`](https://r-xla.github.io/anvl/reference/jit.md)
