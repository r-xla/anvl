# R Data Class

The
[`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md)
of a non-static R value that was passed to a jit-compiled function. It
is special, because it does not have a data type, i.e., calling
[`dtype()`](https://r-xla.github.io/anvl/dev/reference/dtype.md) on it
results in an error.

## Usage

``` r
RData(shape, r_type)
```

## Arguments

- shape:

  ([`stablehlo::Shape`](https://r-xla.github.io/stablehlo/reference/Shape.html)
  \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The shape of the value: `()` for a length-1 vector, its
  [`dim()`](https://rdrr.io/r/base/dim.html) for an R array.

- r_type:

  (`character(1)`)  
  The R storage type: `"double"`, `"integer"` or `"logical"`.

## Value

(`RData`)

## Details

It only exists *during* tracing, never in a finalized graph.

## Extractors

Just like
[`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md),
with the exception that
[`dtype()`](https://r-xla.github.io/anvl/dev/reference/dtype.md) errs.

## See also

[AbstractArray](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md),
[AnvlGraph](https://r-xla.github.io/anvl/dev/reference/AnvlGraph.md)

## Examples

``` r
RData(c(2, 3), "double")
#> RData(double, (2,3)) 
# is equivalent to
nv_aval("double", c(2, 3))
#> RData(double, (2,3)) 
# Below, the `RData` input is materialized in 32 and 64-bit precisions, so the input
# dtype becomes f64.
# By NOT converting RData to their default data type we prevent loss of precision
# (double -> f32 -> f64 roundrips)
graph <- trace_fn(function(x) {
    print(x)
    list(x + nv_scalar(1, "f64"), x + nv_scalar(1, "f32"))
  }, list(x = nv_aval("double", c()))
)
#> GraphBox(GraphValue(RData(double, ()))) 
print(graph)
#> <AnvlGraph>
#>   Inputs:
#>     %x1: f64[] <- double
#>   Constants:
#>     %c1: f64[]
#>     %c2: f32[]
#>   Body:
#>     %1: f32[] = convert [dtype = f32] (%x1)
#>     %2: f64[] = add(%x1, %c1)
#>     %3: f32[] = add(%1, %c2)
#>   Outputs:
#>     %2: f64[]
#>     %3: f32[] 
# The actual inputs to the compiled program
graph$inputs
#> [[1]]
#> GraphValue(AbstractArray(dtype=f64, shape=)) 
#> 
# The data types of the R values; AnvlArrays get NA here
graph$rdata_types
#> [1] "double"
```
