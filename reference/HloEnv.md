# HloEnv

Environment for storing graph value to func value mappings. This is a
mutable class. Every graph is lowered against an environment of its own:
a sub-graph reads nothing of the graph around it except through its
inputs.

## Usage

``` r
HloEnv(gval_to_fval = NULL)
```

## Arguments

- gval_to_fval:

  (`hashtab`)  
  Mapping from graph values to func values.

## Value

(`HloEnv`)
