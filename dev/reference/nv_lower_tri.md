# Lower Triangular Mask

Returns a boolean matrix that is `TRUE` on and below the given diagonal,
mirroring base R's
[`lower.tri()`](https://rdrr.io/r/base/lower.tri.html). Use
[`nv_tril()`](https://r-xla.github.io/anvl/dev/reference/nv_tril.md) to
zero out the other triangle of an existing array instead.

## Usage

``` r
nv_lower_tri(shape, diagonal = -1L, device = NULL)

nv_lower_tri_like(like, diagonal = -1L, shape = NULL, device = NULL)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- diagonal:

  (`integer(1)`)  
  Diagonal offset, with the same meaning as in
  [`nv_tril()`](https://r-xla.github.io/anvl/dev/reference/nv_tril.md).
  The default `-1` excludes the main diagonal, matching
  [`lower.tri()`](https://rdrr.io/r/base/lower.tri.html); use `0` to
  include it, matching `lower.tri(diag = TRUE)`.

- device:

  ( `character(1)` \| `PJRTDevice` \|
  [`quickr_device`](https://r-xla.github.io/anvl/dev/reference/quickr_device.md)
  \| `NULL`)  
  Device for data to live on.

- like:

  ([`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md))  
  Existing array whose attributes are used as defaults (only for
  `nv_lower_tri_like()`).

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the given `shape` and dtype `bool`.

## See also

[`nv_upper_tri()`](https://r-xla.github.io/anvl/dev/reference/nv_upper_tri.md),
[`nv_tril()`](https://r-xla.github.io/anvl/dev/reference/nv_tril.md),
[`prim_iota()`](https://r-xla.github.io/anvl/dev/reference/prim_iota.md)
for the underlying primitive.

## Examples

``` r
nv_lower_tri(c(3, 3))
#> AnvlArray
#>  0 0 0
#>  1 0 0
#>  1 1 0
#> [ CPUbool{3,3} ] 
nv_lower_tri(c(3, 3), diagonal = 0L)
#> AnvlArray
#>  1 0 0
#>  1 1 0
#>  1 1 1
#> [ CPUbool{3,3} ] 
x <- nv_fill(0, shape = c(3, 3))
nv_lower_tri_like(x)
#> AnvlArray
#>  0 0 0
#>  1 0 0
#>  1 1 0
#> [ CPUbool{3,3} ] 
```
