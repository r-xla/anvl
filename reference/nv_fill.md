# Fill Constant

Creates an array filled with a scalar value. More memory-efficient than
`nv_array(value, shape = shape)` for large arrays.

`nv_fill_like()` is a variant where `dtype`, `shape`, `ambiguous`, and
`device` default to those of `like`.

## Usage

``` r
nv_fill(value, shape, dtype = NULL, ambiguous = FALSE, device = NULL)

nv_fill_like(
  like,
  value,
  shape = NULL,
  dtype = NULL,
  ambiguous = NULL,
  device = NULL
)
```

## Arguments

- value:

  (`numeric(1)`)  
  Scalar value to fill the array with.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape of the output array.

- dtype:

  (`character(1)` \| `NULL`)  
  Data type.

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
  `nv_fill_like()`).

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the given `shape` and `dtype`.

## See also

[`prim_fill()`](https://r-xla.github.io/anvl/reference/prim_fill.md) for
the underlying primitive.

## Examples

``` r
nv_fill(0, shape = c(2, 3))
#> AnvlArray
#>  0 0 0
#>  0 0 0
#> [ CPUf32{2,3} ] 
x <- nv_matrix(1:6, nrow = 2)
nv_fill_like(x, 0)
#> AnvlArray
#>  0 0 0
#>  0 0 0
#> [ CPUi32{2,3} ] 
```
