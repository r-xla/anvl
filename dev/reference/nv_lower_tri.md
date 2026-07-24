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
