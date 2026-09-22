# Transform a graph to its gradient

Low-level graph transformation that transforms a graph into its
gradient. The function `f` represented by `graph` must return a single
float scalar. The resulting graph computes the gradients of that scalar
with respect to the inputs specified by `wrt`.

## Usage

``` r
transform_gradient(graph, wrt)
```

## Arguments

- graph:

  ([`AnvlGraph`](https://r-xla.github.io/anvl/dev/reference/AnvlGraph.md))  
  The graph to transform. Must produce a single scalar float output.

- wrt:

  (`character`)  
  Names of the graph inputs to differentiate with respect to.

## Value

([`AnvlGraph`](https://r-xla.github.io/anvl/dev/reference/AnvlGraph.md))  
Its outputs are the requested gradients.

## Details

To support alternative forward passes for more efficient backward
passes, we replay and possibly rewrite the graph into a new descriptor.
Afterwards, we traverse it backwards and call the gradient rules where
necessary.

See
[`rule_reverse()`](https://r-xla.github.io/anvl/dev/reference/rule_reverse.md)
for more information.

This is the building block used by
[`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md)
and
[`value_and_gradient()`](https://r-xla.github.io/anvl/dev/reference/value_and_gradient.md);
prefer those higher-level wrappers unless you need to operate on graphs
directly.

## See also

[`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md),
[`value_and_gradient()`](https://r-xla.github.io/anvl/dev/reference/value_and_gradient.md),
[`rule_reverse()`](https://r-xla.github.io/anvl/dev/reference/rule_reverse.md)

## Examples

``` r
graph <- trace_fn(prim_mul, list(nv_aval("f32", c()), nv_aval("f32", c())))
graph
#> <AnvlGraph> (%x1: f32[], %x2: f32[]) {
#>   %1: f32[] = mul(%x1, %x2)
#>   return %1
#> }
transform_gradient(graph, "lhs")
#> <AnvlGraph> [%c1: f32[]] (%x1: f32[], %x2: f32[]) {
#>   %1: f32[] = mul(%x1, %x2)
#>   %2: f32[] = mul(%c1, %x2)
#>   return %2
#> }
```
