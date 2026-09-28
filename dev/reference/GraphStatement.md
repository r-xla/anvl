# Graph Statement

One statement of an
[`AnvlGraph`](https://r-xla.github.io/anvl/dev/reference/AnvlGraph.md):
a primitive applied to inputs, with its results assigned to outputs.

## Usage

``` r
GraphStatement(primitive, inputs, params, outputs)
```

## Arguments

- primitive:

  (`AnvlPrimitiveDef`)  
  The function.

- inputs:

  (`list(GraphValue)`)  
  The (array) inputs to the primitive.

- params:

  (`list(<any>)`)  
  The (static) parameters of the function call.

- outputs:

  (`list(GraphValue)`)  
  The (array) outputs of the primitive.

## Value

(`GraphStatement`)
