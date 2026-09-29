# Primitive Definition

The definition of a primitive: its name, sub-graph parameters and
interpretation rules. It is not callable; the function a primitive is
called through is an
[`AnvlPrimitive`](https://r-xla.github.io/anvl/reference/new_primitive.md),
which carries its `AnvlPrimitiveDef` as `attr(<fn>, "definition")`. The
graph records the `AnvlPrimitiveDef` in each
[`GraphStatement`](https://r-xla.github.io/anvl/reference/GraphStatement.md).
Note that `[[` and `[[<-` access the interpretation rules. To access
other fields, use `$` and `$<-`.

A primitive is considered higher-order if it has subgraphs.

## Usage

``` r
AnvlPrimitiveDef(name, subgraphs = character())
```

## Arguments

- name:

  (`character(1)`)  
  The name of the primitive, without the `prim_` prefix.

- subgraphs:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of parameters that are subgraphs.

## Value

(`AnvlPrimitiveDef`)
