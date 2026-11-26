# Primitive Definition

The definition of a primitive: its name, sub-graph parameters and
interpretation rules. It is not callable; the function a primitive is
called through is an
[`AnvlPrimitive`](https://r-xla.github.io/anvl/dev/reference/new_primitive.md),
which carries its `AnvlPrimitiveDef` as `attr(<fn>, "definition")`. The
graph records the `AnvlPrimitiveDef` in each
[`GraphStatement`](https://r-xla.github.io/anvl/dev/reference/GraphStatement.md).
Note that `[[` and `[[<-` access the interpretation rules. To access
other fields, use `$` and `$<-`.
[`new_primitive()`](https://r-xla.github.io/anvl/dev/reference/new_primitive.md)
stores the `AnvlPrimitive` in the field `fn`, so a rule can call the
primitive it belongs to.

A primitive is considered higher-order if it has subgraphs.

## Usage

``` r
AnvlPrimitiveDef(name, subgraphs = character(), tags = character())
```

## Arguments

- name:

  (`character(1)`)  
  The name of the primitive, without the `prim_` prefix.

- subgraphs:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of parameters that are subgraphs.

- tags:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Tags for internal use, see
  [`new_primitive()`](https://r-xla.github.io/anvl/dev/reference/new_primitive.md).

## Value

(`AnvlPrimitiveDef`)
