# Sequence

Creates a 1-D array with the consecutive integer values from `start` to
`end` (inclusive), like R's `seq(start, end)`.

`nv_seq_like()` is a variant where `dtype` and `device` default to those
of `like`.

## Usage

``` r
nv_seq(start, end, dtype = NULL, device = NULL)

nv_seq_like(like, start, end, dtype = NULL, device = NULL)
```

## Arguments

- start, end:

  (`integer(1)`)  
  Start and end values, which must satisfy `start <= end`.

- dtype:

  (`NULL` \| `character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type. `NULL` (default) uses the backend's default integer data
  type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)).
  For `nv_seq_like()`, `NULL` uses `dtype(like)`.

- device:

  (`NULL` \| `character(1)` \|
  [device](https://r-xla.github.io/anvl/dev/reference/nv_device.md))  
  The device the data lives on, given either as:

  - a *device string* naming the platform (e.g. `"cpu"`, `"cuda"`,
    `"cuda:<n>"`), which is resolved against the backend in use, or

  - a *device object* as returned by
    [`nv_device()`](https://r-xla.github.io/anvl/dev/reference/nv_device.md):
    a
    [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html)
    for the `"pjrt"` backend or a
    [`quickr_device`](https://r-xla.github.io/anvl/dev/reference/quickr_device.md)
    for the `"quickr"` backend. Because a device object is
    backend-specific, it also determines the backend.

  The default (`NULL`) uses
  [`default_device()`](https://r-xla.github.io/anvl/dev/reference/default_device.md):
  the CPU, or the platform named by the `PJRT_PLATFORM` environment
  variable on the `"pjrt"` backend.

- like:

  ([`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md))  
  Existing array whose attributes are used as defaults (only for
  `nv_seq_like()`).

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
1-D array of length `end - start + 1`.

## See also

[`nv_linspace()`](https://r-xla.github.io/anvl/dev/reference/nv_linspace.md)
for a given number of evenly spaced values,
[`prim_iota()`](https://r-xla.github.io/anvl/dev/reference/prim_iota.md)
for the underlying primitive.

## Examples

``` r
nv_seq(3, 7)
#> AnvlArray
#>  3
#>  4
#>  5
#>  6
#>  7
#> [ CPUi32{5} ] 
x <- nv_array(c(1, 2, 3), dtype = "f64")
nv_seq_like(x, 1, 5)
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#>  5
#> [ CPUf64{5} ] 
```
