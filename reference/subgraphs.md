# Get Subgraphs from Higher-Order Primitive

Extracts the subgraphs from the parameters of a statement that applies a
higher-order primitive, such as the branches of
[`prim_if()`](https://r-xla.github.io/anvl/reference/prim_if.md) or the
body of
[`prim_while()`](https://r-xla.github.io/anvl/reference/prim_while.md).

This is not recursive: only the subgraphs held directly by `statement`
are returned, not the ones nested in the statements of those subgraphs.
Call `subgraphs()` on their statements to descend further.

## Usage

``` r
subgraphs(statement)
```

## Arguments

- statement:

  (`GraphStatement`)  
  The statement.

## Value

(named `list(AnvlGraph)`)  
The subgraphs, named after the parameters that hold them. Empty for a
primitive that is not higher-order.
