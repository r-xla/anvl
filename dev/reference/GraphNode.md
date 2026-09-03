# Graph Node

Virtual base class for nodes in an
[`AnvlGraph`](https://r-xla.github.io/anvl/dev/reference/AnvlGraph.md).
Is a
[`GraphValue`](https://r-xla.github.io/anvl/dev/reference/GraphValue.md),
a
[`GraphLiteral`](https://r-xla.github.io/anvl/dev/reference/GraphLiteral.md),
or – only while tracing, and never as part of a primitive call – an
input whose aval is an
[`RData`](https://r-xla.github.io/anvl/dev/reference/RData.md). Cannot
be instantiated directly - use
[`GraphValue()`](https://r-xla.github.io/anvl/dev/reference/GraphValue.md)
or
[`GraphLiteral()`](https://r-xla.github.io/anvl/dev/reference/GraphLiteral.md)
instead.
