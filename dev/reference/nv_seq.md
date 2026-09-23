# Sequence

Creates a 1-D array with the values from `from` to `to` in steps of
`by`, like R's [`seq()`](https://rdrr.io/r/base/seq.html). The sequence
counts down when `to` lies below `from`, and stops before `to` when `to`
is not reachable in whole steps: `nv_seq(0, 9, by = 2)` ends at `8`.

`nv_seq_like()` is a variant where `dtype` and `device` default to those
of `like`.

## Usage

``` r
nv_seq(from, to, by = NULL, dtype = NULL, device = NULL)

nv_seq_like(like, from, to, by = NULL, dtype = NULL, device = NULL)
```

## Arguments

- from, to:

  (`integer(1)`)  
  First value and upper (or, when counting down, lower) limit of the
  sequence.

- by:

  (`NULL` \| `integer(1)`)  
  Step size, which must be a non-zero whole number pointing from `from`
  towards `to`. `NULL` (default) uses `-1` if `from > to` and `1`
  otherwise.

- dtype:

  (`NULL` \| `character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type of the result. Can be any numeric data type. `NULL`
  (default) uses the default integer data type (see
  [`default_dtypes()`](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)),
  since the values are whole. For `nv_seq_like()`, `NULL` uses
  `dtype(like)`.

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
  [`default_device()`](https://r-xla.github.io/anvl/dev/reference/default_device.md).

- like:

  ([`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md))  
  Existing array whose attributes are used as defaults (only for
  `nv_seq_like()`).

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has `dtype` and shape `(to - from) %/% by + 1`.

## See also

[`nv_linspace()`](https://r-xla.github.io/anvl/dev/reference/nv_linspace.md)
for a given number of evenly spaced values,
[`nv_iota()`](https://r-xla.github.io/anvl/dev/reference/nv_iota.md) for
values increasing along an axis of any shape,
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

# a range that counts down needs no `by`
nv_seq(7, 3)
#> AnvlArray
#>  7
#>  6
#>  5
#>  4
#>  3
#> [ CPUi32{5} ] 

# `to` is only reached where a whole number of steps lands on it
nv_seq(0, 9, by = 2)
#> AnvlArray
#>  0
#>  2
#>  4
#>  6
#>  8
#> [ CPUi32{5} ] 

# a float data type gives the same values as floats
nv_seq(3, 7, dtype = "f32")
#> AnvlArray
#>  3
#>  4
#>  5
#>  6
#>  7
#> [ CPUf32{5} ] 

# nv_seq_like() takes the data type and device from an existing array
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
