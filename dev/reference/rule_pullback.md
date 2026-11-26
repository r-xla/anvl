# Pullback Rule

Construct a pullback rule for a primitive, the rule that reverse-mode
automatic differentiation uses. Provide exactly one of `backward` and
`forward`.

Pass `backward` when the primitive's forward statement can run
unmodified, which covers most use cases. It has the signature
`function(inputs, outputs, grads, params, required)` and returns a
`list` with one entry per input: that input's gradient, or `NULL` where
`required` says it is not needed.

Pass `forward` when a slightly different forward pass enables a more
efficient backward pass. It has the signature
`function(inputs, params, required)`, where `required` says which inputs
need a gradient, and returns `list(outputs = , backward = )`: the
forward results and a closure with the signature of `backward` above,
which can use intermediate values of the forward pass via lexical
scoping.

A rule with a `forward` pass can also be the `rule` of
[`prim_custom_gradient()`](https://r-xla.github.io/anvl/dev/reference/prim_custom_gradient.md).

## Usage

``` r
rule_pullback(backward = NULL, forward = NULL)
```

## Arguments

- backward:

  (`NULL` \| `function`)  
  Backward hook for the default case.

- forward:

  (`NULL` \| `function`)  
  Alternative forward hook that returns both the outputs and a backward
  closure.

## Value

(`anvl_rule_pullback`)

## See also

[`graph_gradient()`](https://r-xla.github.io/anvl/dev/reference/graph_gradient.md)

## Examples

``` r
# the rule of prim_negate()
rule_pullback(function(inputs, outputs, grads, params, required) {
  list(if (required[[1L]]) prim_negate(grads[[1L]]))
})
#> $forward
#> NULL
#> 
#> $backward
#> function (inputs, outputs, grads, params, required) 
#> {
#>     list(if (required[[1L]]) prim_negate(grads[[1L]]))
#> }
#> <environment: 0x5f10c0cf6fc8>
#> 
#> attr(,"class")
#> [1] "anvl_rule_pullback"
```
