# Get the current graph

Get the current graph being built (via
[`local_descriptor`](https://r-xla.github.io/anvl/dev/reference/local_descriptor.md)).

## Usage

``` r
.current_descriptor(silent = FALSE)
```

## Arguments

- silent:

  (`logical(1)`)  
  Whether to return `NULL` if no graph is currently being built (as
  opposed to aborting).

## Value

([`GraphDescriptor`](https://r-xla.github.io/anvl/dev/reference/GraphDescriptor.md))
