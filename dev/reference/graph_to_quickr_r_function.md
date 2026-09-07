# Convert an AnvlGraph to a plain R function

Lowers a supported subset of `AnvlGraph` objects to a plain R function
(no compilation) suitable for
[`quickr::quick()`](https://rdrr.io/pkg/quickr/man/quick.html). The
returned function expects plain R scalars/vectors/arrays and returns
plain R values/arrays.

## Usage

``` r
graph_to_quickr_r_function(graph)
```

## Arguments

- graph:

  ([`AnvlGraph`](https://r-xla.github.io/anvl/dev/reference/AnvlGraph.md))  
  Graph to convert.

## Value

(`function`)

## Details

Most users will prefer
[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) under
`with_backend("quickr", ...)`. This function is the lower-level graph
API.

## See also

[`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) under
`with_backend("quickr", ...)` for tracing and compiling a regular R
function in one step.
