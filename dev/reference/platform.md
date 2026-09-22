# Get the platform of an array or buffer

Returns the platform name (e.g. `"cpu"`, `"cuda"`) identifying the
compute backend.

## Usage

``` r
# S3 method for class 'AnvlArray'
platform(x, ...)

platform(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  An array-like object.

- ...:

  Additional arguments passed to methods (unused).

## Value

(`character(1)`)

## Details

Implemented via the generic
[`pjrt::platform()`](https://r-xla.github.io/pjrt/reference/platform.html).

## See also

[`pjrt::platform()`](https://r-xla.github.io/pjrt/reference/platform.html)

## Examples

``` r
x <- nv_array(1:4, dtype = "f32")
platform(x)
#> [1] "cpu"
```
