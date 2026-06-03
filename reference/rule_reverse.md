# Reverse Rule

Construct a reverse-mode autodiff rule for a primitive. The `backward`
argument should be provided if the `forward` call for the primitive
should run un-modified. This covers most use-cases. The `backward`
argument should have this signature:
`function(inputs, outputs, grads, params, required) -> list(input_grads)`.

In some scenarios, it can be beneficial to perform a slightly different
forward pass to enable a more efficient backward pass. In this case,
pass the `forward` argument. It should return a list containing the
results from the forward pass, as well as closure that has the same
signature as the one above. It can make use of intermediate values
computed in the forward pass via lexical scoping.

## Usage

``` r
rule_reverse(backward = NULL, forward = NULL)
```

## Arguments

- backward:

  (`function`)  
  Backward hook for default case.

- forward:

  (`function`)  
  Alternative-forward hook that returns both primals and backward
  closure.

## Value

An `anvl_rule_reverse` object.

## See also

[`transform_gradient()`](https://r-xla.github.io/anvl/reference/transform_gradient.md)
