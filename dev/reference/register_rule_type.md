# Register a Rule Type

**\[experimental\]**

Makes `name` a type of rule that primitives can carry: afterwards,
`prim_<id>[[name]] <- rule` attaches a rule of that type to a primitive
and `prim_<id>[[name]]` returns it, or `NULL` for a primitive without
one. A backend added with
[`register_backend()`](https://r-xla.github.io/anvl/dev/reference/register_backend.md)
registers the type of the rules it lowers or evaluates the primitives
with, as the `"pjrt"` backend lowers them with their `"stablehlo"` rules
and the `"quickr"` backend with their `"quickr"` rules. Registering a
type again has no further effect.

## Usage

``` r
register_rule_type(name)
```

## Arguments

- name:

  (`character(1)`)  
  The rule type.

## Value

(`NULL`)  
Invisibly. Called for its side effect.

## See also

[`AnvlPrimitiveDef`](https://r-xla.github.io/anvl/dev/reference/AnvlPrimitiveDef.md),
[`register_backend()`](https://r-xla.github.io/anvl/dev/reference/register_backend.md)
