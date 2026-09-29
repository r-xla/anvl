# Peek at a Data Type

The data type `x` would take if it materialized. Relevant for R objects
and their [`RData`](https://r-xla.github.io/anvl/reference/RData.md)
trace-time analogon: for those it is the default of the active backend
(see
[`default_dtypes()`](https://r-xla.github.io/anvl/reference/default_dtypes.md)),
which the value has not materialized at yet.

## Usage

``` r
peek_dtype(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md) \|
  [`AbstractArray`](https://r-xla.github.io/anvl/reference/AbstractArray.md))  
  The value to ask about.

## Value

([`xlamisc::DataType`](https://r-xla.github.io/xlamisc/reference/DataType.html))  
The data type `x` has, or the [default data
type](https://r-xla.github.io/anvl/reference/default_dtypes.md) it would
materialize at if it is still a bare R value.

## See also

[`as_anvl_arrays()`](https://r-xla.github.io/anvl/reference/as_anvl_array.md),
[RData](https://r-xla.github.io/anvl/reference/RData.md),
[shape()](https://r-xla.github.io/xlamisc/reference/shape.html)

## Examples

``` r
peek_dtype(1.5)
#> <f32>
peek_dtype(1L)
#> <i32>
peek_dtype(nv_array(1:3, dtype = "i8"))
#> <i8>
```
