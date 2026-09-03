# Abstract Array Class

Representation of an abstract array type. During tracing, it is wrapped
in a
[`GraphNode`](https://r-xla.github.io/anvl/dev/reference/GraphNode.md)
held by a
[`GraphBox`](https://r-xla.github.io/anvl/dev/reference/GraphBox.md). In
the lowered
[`AnvlGraph`](https://r-xla.github.io/anvl/dev/reference/AnvlGraph.md)
it is also part of
[`GraphNode`](https://r-xla.github.io/anvl/dev/reference/GraphNode.md)s
representing the values in the program.

The base class represents an *unknown* value, but child classes exist
for:

- closed-over constants:
  [`ConcreteArray`](https://r-xla.github.io/anvl/dev/reference/ConcreteArray.md)

- scalar arrays arising from R literals:
  [`LiteralArray`](https://r-xla.github.io/anvl/dev/reference/LiteralArray.md)

- sequence patterns:
  [`IotaArray`](https://r-xla.github.io/anvl/dev/reference/IotaArray.md)

- R values
  [`RData`](https://r-xla.github.io/anvl/dev/reference/RData.md). They
  are special because they do not have a data type.

To convert a
[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
value to an abstract array, use
[`to_abstract()`](https://r-xla.github.io/anvl/dev/reference/to_abstract.md).

## Usage

``` r
nv_aval(dtype, shape)

AbstractArray(dtype, shape)
```

## Arguments

- dtype:

  ([`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html)
  \| `character(1)`)  
  The data type of the array. To create an
  [`RData`](https://r-xla.github.io/anvl/dev/reference/RData.md) object,
  specify `"double"`, `"integer"`, or `"logical"`.

- shape:

  ([`stablehlo::Shape`](https://r-xla.github.io/stablehlo/reference/Shape.html)
  \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The shape of the array. Can be provided as an integer vector.

## Extractors

The following extractors are available on `AbstractArray` objects:

- [`dtype()`](https://r-xla.github.io/tengen/reference/dtype.html): Get
  the data type of the array.

- [`shape()`](https://r-xla.github.io/tengen/reference/shape.html): Get
  the shape (axis sizes) of the array.

- [`naxes()`](https://r-xla.github.io/tengen/reference/naxes.html): Get
  the number of axes.

## See also

[LiteralArray](https://r-xla.github.io/anvl/dev/reference/LiteralArray.md),
[ConcreteArray](https://r-xla.github.io/anvl/dev/reference/ConcreteArray.md),
[IotaArray](https://r-xla.github.io/anvl/dev/reference/IotaArray.md),
[RData](https://r-xla.github.io/anvl/dev/reference/RData.md),
[GraphValue](https://r-xla.github.io/anvl/dev/reference/GraphValue.md),
[`to_abstract()`](https://r-xla.github.io/anvl/dev/reference/to_abstract.md),
[GraphBox](https://r-xla.github.io/anvl/dev/reference/GraphBox.md)

## Examples

``` r
# -- Creating abstract arrays --
a <- AbstractArray("f32", c(2L, 3L))
a
#> AbstractArray(dtype=f32, shape=2x3) 
dtype(a)
#> <f32>
shape(a)
#> [1] 2 3

# Shorthand
nv_aval("f32", c(2L, 3L))
#> AbstractArray(dtype=f32, shape=2x3) 

# An R value, which has no dtype until it is used
nv_aval("double", c(2L, 3L))
#> RData(double, (2,3)) 

# How AbstractArrays appear in an AnvlGraph
graph <- trace_fn(function(x) x + 1, list(x = nv_aval("i32", 4L)))
graph
#> <AnvlGraph>
#>   Inputs:
#>     %x1: i32[4]
#>   Body:
#>     %1: f32[4] = convert [dtype = f32] (%x1)
#>     %2: f32[4] = broadcast_in_axes [shape = 4, broadcast_axes = <any>] (1:f32)
#>     %3: f32[4] = add(%1, %2)
#>   Outputs:
#>     %3: f32[4] 
graph$inputs[[1]]$aval
#> AbstractArray(dtype=i32, shape=4) 
```
