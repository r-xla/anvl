# Get the Platform Name of an Array or Buffer

Returns the name of the hardware platform (e.g. `"cpu"`, `"cuda"`) the
data lives on. This is not the backend; see
[`backend()`](https://r-xla.github.io/anvl/dev/reference/backend.md) for
that.

## Usage

``` r
# S3 method for class 'AnvlArray'
platform(x, ...)

platform(x, ...)
```

## Arguments

- x:

  ([`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  \|
  [`PJRTBuffer`](https://r-xla.github.io/pjrt/reference/pjrt_buffer.html))  
  An array or buffer.

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
