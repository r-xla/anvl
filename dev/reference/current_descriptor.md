# Get the Current Graph

Get the current graph being built.

## Usage

``` r
current_descriptor(silent = FALSE)
```

## Arguments

- silent:

  (`logical(1)`)  
  Whether to return `NULL` if no graph is currently being built (as
  opposed to aborting).

## Value

([`GraphDescriptor`](https://r-xla.github.io/anvl/dev/reference/GraphDescriptor.md)
\| `NULL`)  
`NULL` only when `silent = TRUE` and no graph is being built.
