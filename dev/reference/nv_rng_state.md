# Generate RNG State

Creates an initial RNG state from a seed. This state is required by all
random sampling functions and is updated after each call.

## Usage

``` r
nv_rng_state(seed, device = default_device())
```

## Arguments

- seed:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Scalar `i32` seed value.

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

## Value

[`nv_array`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md) of
dtype `ui64` and shape `(2)`.

## See also

Other rng:
[`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md),
[`nv_rdunif()`](https://r-xla.github.io/anvl/dev/reference/nv_rdunif.md),
[`nv_rnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_rnorm.md),
[`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_runif.md)

## Examples

``` r
state <- nv_rng_state(42L)
state
#> AnvlArray
#>  42
#>   0
#> [ CPUui64{2} ] 
```
