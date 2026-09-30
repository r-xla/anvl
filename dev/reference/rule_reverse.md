# Reverse Rule

Construct a reverse-mode autodiff rule for a primitive. Provide exactly
one of `backward` and `forward`.

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

## Usage

``` r
rule_reverse(backward = NULL, forward = NULL)
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

(`anvl_rule_reverse`)

## See also

[`transform_gradient()`](https://r-xla.github.io/anvl/dev/reference/transform_gradient.md)

## Examples

``` r
# the rule of prim_negate()
rule_reverse(function(inputs, outputs, grads, params, required) {
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
#> <environment: 0x55934703e858>
#> 
#> attr(,"class")
#> [1] "anvl_rule_reverse"
```
