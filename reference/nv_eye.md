# Identity Matrix

Creates an `n x n` identity matrix.

`nv_eye_like()` is a variant where `dtype` and `device` default to those
of `like`.

## Usage

``` r
nv_eye(n, dtype = "f32", device = NULL)

nv_eye_like(like, n, dtype = NULL, device = NULL)
```

## Arguments

- n:

  (`integer(1)`)  
  Size of the identity matrix.

- dtype:

  (`character(1)` \|
  [`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type.

- device:

  ( `character(1)` \| `PJRTDevice` \|
  [`quickr_device`](https://r-xla.github.io/anvl/reference/quickr_device.md)
  \| `NULL`)  
  Device for data to live on.

- like:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Existing array whose attributes are used as defaults (only for
  `nv_eye_like()`).

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
An `n x n` identity matrix.

## See also

[`nv_diag()`](https://r-xla.github.io/anvl/reference/nv_diag.md) for
general diagonal matrices.

## Examples

``` r
nv_eye(3L)
#> AnvlArray
#>  1 0 0
#>  0 1 0
#>  0 0 1
#> [ CPUf32{3,3} ] 
x <- nv_fill(0, shape = c(3, 3), dtype = "f64")
nv_eye_like(x, 3L)
#> AnvlArray
#>  1 0 0
#>  0 1 0
#>  0 0 1
#> [ CPUf64{3,3} ] 
```
