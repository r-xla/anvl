# Sequence

Creates a 1-D array with values from `start` to `end` (inclusive).

Without `steps`, behaves like R's `seq(start, end)` producing integer
values. With `steps`, produces `steps` evenly spaced values (like
`seq(start, end, length.out = steps)`).

`nv_seq_like()` is a variant where `dtype` and `device` default to those
of `like`.

## Usage

``` r
nv_seq(start, end, steps = NULL, dtype = NULL, device = NULL)

nv_seq_like(like, start, end, steps = NULL, dtype = NULL, device = NULL)
```

## Arguments

- start, end:

  (`numeric(1)`)  
  Start and end values. When `steps` is `NULL`, must satisfy
  `start <= end`.

- steps:

  (`integer(1)` or `NULL`)  
  Number of evenly spaced values to generate. Must be at least 1. When
  `NULL` (default), generates consecutive integer values from `start` to
  `end`.

- dtype:

  (`character(1)`)  
  Data type. Default `"i32"` when `steps` is `NULL`, `"f32"` when
  `steps` is given. For `nv_seq_like()`, `NULL` uses `dtype(like)`.

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
