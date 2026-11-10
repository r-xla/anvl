# Graph Value

Value in an
[`AnvlGraph`](https://r-xla.github.io/anvl/reference/AnvlGraph.md). This
is a mutable class.

## Usage

``` r
GraphValue(aval, label = NULL)
```

## Arguments

- aval:

  ([`AbstractArray`](https://r-xla.github.io/anvl/reference/AbstractArray.md))  
  The abstract value of the variable.

- label:

  (`NULL` \| `character(1)`)  
  What the value is, for a printed graph to name it after, e.g. `"key"`
  for the key of the random state, which is printed as `%key1`.

## Value

(`GraphValue`)
