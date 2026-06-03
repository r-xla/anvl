# Iota

Creates an array with values increasing along the specified dimension,
starting from `start`.

`nv_iota_like()` is a variant where `dtype`, `shape`, `ambiguous`, and
`device` default to those of `like`.

## Usage

``` r
nv_iota(dim, dtype, shape, start = 1L, ambiguous = FALSE, device = NULL)

nv_iota_like(
  like,
  dim,
  shape = NULL,
  start = 1L,
  dtype = NULL,
  ambiguous = NULL,
  device = NULL
)
```

## Arguments

- dim:

  (`integer(1)`)  
  Dimension along which values increase.

- dtype:

  (`character(1)` \|
  [`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- start:

  (`integer(1)`)  
  Starting value (default 1).

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the
  [`vignette("type-promotion")`](https://r-xla.github.io/anvl/articles/type-promotion.md)
  for more details.

- device:

  ( `character(1)` \| `PJRTDevice` \|
  [`quickr_device`](https://r-xla.github.io/anvl/reference/quickr_device.md)
  \| `NULL`)  
  Device for data to live on.

- like:

  ([`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md))  
  Existing array whose attributes are used as defaults (only for
  `nv_iota_like()`).

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the given `dtype` and `shape`.

## See also

[`nv_seq()`](https://r-xla.github.io/anvl/reference/nv_seq.md) for a
simpler 1-D sequence,
[`prim_iota()`](https://r-xla.github.io/anvl/reference/prim_iota.md) for
the underlying primitive.

## Examples

``` r
nv_iota(dim = 1L, dtype = "i32", shape = 5L)
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#>  5
#> [ CPUi32{5} ] 
x <- nv_fill(0L, shape = c(2, 3))
nv_iota_like(x, dim = 1L)
#> AnvlArray
#>  1 1 1
#>  2 2 2
#> [ CPUi32{2,3} ] 
```
