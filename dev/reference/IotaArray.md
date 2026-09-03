# Iota Array Class

An
[`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md)
representing an integer sequence. Usually created by
[`nv_iota()`](https://r-xla.github.io/anvl/dev/reference/nv_iota.md) /
[`nv_seq()`](https://r-xla.github.io/anvl/dev/reference/nv_seq.md),
which both call
[`prim_iota()`](https://r-xla.github.io/anvl/dev/reference/prim_iota.md)
internally. Inherits from
[`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md).

## Usage

``` r
IotaArray(shape, dtype, axis, start = 1L)
```

## Arguments

- shape:

  ([`stablehlo::Shape`](https://r-xla.github.io/stablehlo/reference/Shape.html)
  \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The shape of the array.

- dtype:

  ([`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  The data type.

- axis:

  (`integer(1)`)  
  The axis along which values increase.

- start:

  (`integer(1)`)  
  The starting value.

## Lowering

When lowering to stableHLO, these become `iota` operations that generate
the integer sequence so they do not need to actually hold the data in
the executable, similar to `ALTREP`s in R. It lowers to
[`hlo_iota()`](https://r-xla.github.io/stablehlo/reference/hlo_iota.html),
optionally shifting the starting value via
[`hlo_add()`](https://r-xla.github.io/stablehlo/reference/hlo_add.html).

## Examples

``` r
x <- IotaArray(shape = 4L, dtype = "i32", axis = 1L)
x
#> IotaArray(shape=(4), dtype=i32, axis=1, start=1) 
shape(x)
#> [1] 4
naxes(x)
#> [1] 1
dtype(x)
#> <i32>
# How it appears during tracing:
graph <- trace_fn(function() nv_iota(axis = 1L, dtype = "i32", shape = 4L), list())
graph
#> <AnvlGraph>
#>   Inputs: (none)
#>   Body:
#>     %1: i32[4] = iota [axis = 1, dtype = i32, shape = 4, start = 1] ()
#>   Outputs:
#>     %1: i32[4] 
graph$outputs[[1]]$aval
#> IotaArray(shape=(4), dtype=i32, axis=1, start=1) 
```
