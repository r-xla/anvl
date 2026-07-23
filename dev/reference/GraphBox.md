# Graph Box

An [`AnvlBox`](https://r-xla.github.io/anvl/dev/reference/AnvlBox.md)
subclass that wraps a
[`GraphNode`](https://r-xla.github.io/anvl/dev/reference/GraphNode.md)
during graph construction (tracing). When a function is traced via
[`trace_fn()`](https://r-xla.github.io/anvl/dev/reference/trace_fn.md),
each intermediate array value is represented as a `GraphBox`. It also
contains an associated
[`GraphDescriptor`](https://r-xla.github.io/anvl/dev/reference/GraphDescriptor.md)
in which the node "lives".

## Usage

``` r
GraphBox(gnode, desc)
```

## Arguments

- gnode:

  ([`GraphNode`](https://r-xla.github.io/anvl/dev/reference/GraphNode.md))  
  The graph node – either a
  [`GraphValue`](https://r-xla.github.io/anvl/dev/reference/GraphValue.md)
  or a
  [`GraphLiteral`](https://r-xla.github.io/anvl/dev/reference/GraphLiteral.md).

- desc:

  ([`GraphDescriptor`](https://r-xla.github.io/anvl/dev/reference/GraphDescriptor.md))  
  The descriptor of the graph being built.

## Value

(`GraphBox`)

## Extractors

- [`dtype()`](https://r-xla.github.io/tengen/reference/dtype.html)

- [`shape()`](https://r-xla.github.io/tengen/reference/shape.html)

- [`naxes()`](https://r-xla.github.io/tengen/reference/naxes.html)

- [`ambiguous()`](https://r-xla.github.io/anvl/dev/reference/ambiguous.md)

## See also

[AnvlBox](https://r-xla.github.io/anvl/dev/reference/AnvlBox.md),
[`trace_fn()`](https://r-xla.github.io/anvl/dev/reference/trace_fn.md),
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)
