# Literal Array Class

An
[`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md)
where all elements have the same constant value. This arises from a
literal in traced code (`x + 1`, say). A
[`nv_fill()`](https://r-xla.github.io/anvl/dev/reference/nv_fill.md) is
a recorded operation rather than a constant, so its output is an
ordinary
[`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md).

## Usage

``` r
LiteralArray(data, shape, dtype = default_dtype(data))
```

## Arguments

- data:

  (`double(1)` \| `integer(1)` \| `logical(1)` \|
  [`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md))  
  The scalar value, or a one-element
  [`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  – for which `dtype` has to be named, since the default takes it from
  an R value's storage type.

- shape:

  ([`stablehlo::Shape`](https://r-xla.github.io/stablehlo/reference/Shape.html)
  \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The shape of the array.

- dtype:

  ([`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  The data type. For the default, see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).

## Lowering

`LiteralArray`s become constants inlined into the StableHLO program.
I.e., they lower to
[`hlo_tensor()`](https://r-xla.github.io/stablehlo/reference/hlo_constant.html).

## Examples

``` r
x <- LiteralArray(1L, shape = integer())
x
#> LiteralArray(1, i32, ()) 
shape(x)
#> integer(0)
naxes(x)
#> [1] 0
dtype(x)
#> <i32>
# how it appears during tracing: an R literal that meets nothing
graph <- trace_fn(function() 1, list())
graph
#> <AnvlGraph> () {
#>   return 1:f32
#> }
graph$outputs[[1]]$aval
#> LiteralArray(1, f32, ()) 
# a `nv_fill()`, by contrast, is a recorded operation
graph <- trace_fn(function() nv_fill(2L, shape = c(2, 2)), list())
graph
#> <AnvlGraph> () {
#>   %1: i32[2,2] = fill [value = 2, dtype = i32, shape = c(2, 2)] ()
#>   return %1
#> }
graph$outputs[[1]]$aval
#> AbstractArray(dtype=i32, shape=2x2) 
```
