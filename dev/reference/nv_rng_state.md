# Generate RNG State

Creates an initial RNG state from a seed. This state is required by all
random sampling functions and is updated after each call.

## Usage

``` r
nv_rng_state(seed, device = NULL)
```

## Arguments

- seed:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Scalar seed. Must be a signed or unsigned integer; it is brought to
  `i32`, so a wider one is narrowed to its low 32 bits. An R integer is
  built at `i32` directly, whatever the default integer data type is.

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

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
A `ui64` array of length 2, whatever `seed`'s data type was.

## See also

Other rng:
[`nv_normal`](https://r-xla.github.io/anvl/dev/reference/nv_normal.md),
[`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md),
[`nv_sample()`](https://r-xla.github.io/anvl/dev/reference/nv_sample.md),
[`nv_sample_int()`](https://r-xla.github.io/anvl/dev/reference/nv_sample_int.md),
[`nv_uniform`](https://r-xla.github.io/anvl/dev/reference/nv_uniform.md)

## Examples

``` r
state <- nv_rng_state(42L)
state
#> AnvlArray
#>  42
#>   0
#> [ CPUui64{2} ] 
```
