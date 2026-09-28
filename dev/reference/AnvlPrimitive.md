# AnvlPrimitive

Metadata object of a primitive: its name, sub-graph parameters and
interpretation rules. Note that `[[` and `[[<-` access the
interpretation rules. To access other fields, use `$` and `$<-`.

A primitive is considered higher-order if it has subgraphs.

## Usage

``` r
AnvlPrimitive(name, subgraphs = character())
```

## Arguments

- name:

  (`character(1)`)  
  The name of the primitive, without the `prim_` prefix.

- subgraphs:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of parameters that are subgraphs.

## Value

(`AnvlPrimitive`)
