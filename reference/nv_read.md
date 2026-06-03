# Read arrays from a file

Loads arrays from a file in the
[safetensors](https://huggingface.co/docs/safetensors/index) format.

## Usage

``` r
nv_read(path, device = NULL, backend = default_backend())
```

## Arguments

- path:

  (`character(1)`)  
  Path to the safetensors file.

- device:

  (`NULL` \| `character(1)` \|
  [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html))  
  The device on which to place the loaded arrays (`"cpu"`, `"cuda"`,
  ...). Default is to use the CPU.

- backend:

  (`character(1)`)  
  Backend for the loaded arrays. Defaults to
  [`default_backend()`](https://r-xla.github.io/anvl/reference/default_backend.md).

## Value

Named `list` of
[`AnvlArray`](https://r-xla.github.io/anvl/reference/AnvlArray.md)
objects.

## Details

This is a convenience wrapper around
[`nv_unserialize()`](https://r-xla.github.io/anvl/reference/nv_unserialize.md)
that opens and closes a file connection.

## See also

[`nv_save()`](https://r-xla.github.io/anvl/reference/nv_save.md),
[`nv_serialize()`](https://r-xla.github.io/anvl/reference/nv_serialize.md),
[`nv_unserialize()`](https://r-xla.github.io/anvl/reference/nv_unserialize.md)

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
x
#> AnvlArray
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2,3} ] 
path <- tempfile(fileext = ".safetensors")
nv_save(list(x = x), path)
nv_read(path)
#> $x
#> AnvlArray
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2,3} ] 
#> 
```
