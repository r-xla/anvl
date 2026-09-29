# Get Backend Name of an Array

Returns the name of the backend an array or device belongs to.

## Usage

``` r
backend(x, ...)
```

## Arguments

- x:

  ([`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md) \|
  device object)  
  An array or a device (see
  [`nv_device()`](https://r-xla.github.io/anvl/reference/nv_device.md)).

- ...:

  Unused.

## Value

(`character(1)`)  
The backend name.

## Examples

``` r
backend(nv_array(1:3))
#> [1] "pjrt"
backend(nv_device("cpu"))
#> [1] "pjrt"
```
