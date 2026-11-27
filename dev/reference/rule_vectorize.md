# Vectorize Rule

Construct the rule
[`vectorize()`](https://r-xla.github.io/anvl/dev/reference/vectorize.md)
uses for a primitive applied to values that are mapped over. Inside the
rule, such values carry the batch axis as their last axis.

Many primitives (such as elementwise ops) just work as is. For others,
some parameters need to be adjusted to account for the added batch axis.
This includes shape parameters of axis mappings. The rule is then
responsible for rewriting these parameters so that they work correctly
with the added batch axis. For the most common cases, see
[`vec_shape()`](https://r-xla.github.io/anvl/dev/reference/vec_shape.md)
where it is enough to just mark the parameter with the appropriate kind
via the `params` argument. For more complex rules it is necessary to
provide the `fn` argument.

## Usage

``` r
rule_vectorize(fn = NULL, params = list(), unbatched = integer())
```

## Arguments

- fn:

  (`NULL` \| `function`)  
  A rule of its own, for a primitive for which declaring `params` and
  `unbatched` is not enough. If given, the other arguments must be left
  at their defaults. It is called as `fn(inputs, batched, params, size)`
  for every call of the primitive with at least one operand that is
  mapped over, where

  - `inputs` is the list of operands, in the order of the primitive's
    arguments: one that is mapped over with the batch axis as its last
    axis, the others as they are,

  - `batched` is a [`logical()`](https://rdrr.io/r/base/logical.html)
    saying which of `inputs` are mapped over,

  - `params` is the named list of the parameters of the call, and

  - `size` is the size of the batch axis.

  It returns the outputs of the primitive applied to every slice, as a
  list in the order of the primitive's outputs, or a single output as it
  is. Every output has the batch axis as its last axis, also one that is
  the same for every slice.

- params:

  (named [`list()`](https://rdrr.io/r/base/list.html))  
  How each parameter with an entry per axis of the operands, or a shape,
  changes, one of the kinds in
  [`vec_shape()`](https://r-xla.github.io/anvl/dev/reference/vec_shape.md).

- unbatched:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Positions of the operands that cannot be mapped over, e.g. the padding
  value of
  [`prim_pad()`](https://r-xla.github.io/anvl/dev/reference/prim_pad.md).
  They are passed on unchanged.

## Value

(`anvl_rule_vectorize`)  
A rule to assign to `prim_<name>[["vectorize"]]`.

## See also

[`vectorize()`](https://r-xla.github.io/anvl/dev/reference/vectorize.md),
[`vec_shape()`](https://r-xla.github.io/anvl/dev/reference/vec_shape.md)

## Examples

``` r
# the rule of prim_transpose()
rule_vectorize(params = list(perm = vec_axis_map()))
#> $fn
#> NULL
#> 
#> $params
#> $params$perm
#> function (x, size) 
#> c(x, length(x) + 1L)
#> <bytecode: 0x615ab6c4be90>
#> <environment: 0x615ab6c4c168>
#> attr(,"class")
#> [1] "anvl_param_kind"
#> 
#> 
#> $unbatched
#> integer(0)
#> 
#> attr(,"class")
#> [1] "anvl_rule_vectorize"
```
