# Create a graph

Creates a new
[`GraphDescriptor`](https://r-xla.github.io/anvl/dev/reference/GraphDescriptor.md)
which is afterwards accessible via
[`.current_descriptor()`](https://r-xla.github.io/anvl/dev/reference/dot-current_descriptor.md).
The graph is automatically removed when exiting the current scope. After
the graph is either cleaned up automatically (by exiting the scope) or
finalized, the previously built graph is restored, i.e., accessible via
[`.current_descriptor()`](https://r-xla.github.io/anvl/dev/reference/dot-current_descriptor.md).

## Usage

``` r
local_descriptor(..., envir = parent.frame())
```

## Arguments

- ...:

  (`any`)  
  Additional arguments to pass to the
  [`GraphDescriptor`](https://r-xla.github.io/anvl/dev/reference/GraphDescriptor.md)
  constructor.

- envir:

  (`environment`)  
  Environment where exit handler will be registered for cleaning up the
  [`GraphDescriptor`](https://r-xla.github.io/anvl/dev/reference/GraphDescriptor.md)
  if it was not returned yet.

## Value

([`GraphDescriptor`](https://r-xla.github.io/anvl/dev/reference/GraphDescriptor.md))
