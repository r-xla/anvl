# Get the Data Type of an Array

Returns the data type of an array (e.g. `f32`, `i64`).

## Usage

``` r
dtype(x, ...)
```

## Arguments

- x:

  ([`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  \|
  [`GraphBox`](https://r-xla.github.io/anvl/dev/reference/GraphBox.md)
  \|
  [`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md))  
  An array. See Details for the values that have no data type.

- ...:

  Additional arguments passed to methods (unused).

## Value

([`DataType`](https://r-xla.github.io/xlamisc/reference/DataType.html))

## Details

An R value has no data type of its own: it only takes one when it meets
a typed array, or when it materializes at the default. So `dtype()` is
an error for a plain R value, and also for an
[`RData`](https://r-xla.github.io/anvl/dev/reference/RData.md), the
[`AbstractArray`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md)
of an R value passed to a jit-compiled function, and so for the
[`GraphBox`](https://r-xla.github.io/anvl/dev/reference/GraphBox.md)
that carries one during tracing. Use
[`peek_dtype()`](https://r-xla.github.io/anvl/dev/reference/peek_dtype.md)
for the data type such a value would take.

This is implemented via the generic
[`xlamisc::dtype()`](https://r-xla.github.io/xlamisc/reference/dtype.html).

## See also

[`xlamisc::dtype()`](https://r-xla.github.io/xlamisc/reference/dtype.html),
[`peek_dtype()`](https://r-xla.github.io/anvl/dev/reference/peek_dtype.md),
[RData](https://r-xla.github.io/anvl/dev/reference/RData.md)

## Examples

``` r
x <- nv_array(1:4, dtype = "f32")
dtype(x)
#> <f32>
# an R value passed to a jit-compiled function has no data type
try(jit(dtype)(1))
#> Error : An R value has no data type of its own until it is used.
#> ℹ `dtype()` is undefined here for the same reason `dtype(1.5)` is: the value
#>   only takes a data type when it meets a typed array, or when it materializes
#>   at the default.
#> ℹ Give it one explicitly with `nv_convert()`.
# an array passed to it does
jit(function(x) {
  print(dtype(x))
  x
})(nv_scalar(1))
#> <f32>
#> AnvlArray
#>  1
#> [ CPUf32{} ] 
```
