# Save and Read Arrays in a File

`nv_save()` saves a named list of arrays to a file in the
[safetensors](https://huggingface.co/docs/safetensors/index) format, and
`nv_read()` loads them back. The data type and shape of each array are
restored.

## Usage

``` r
nv_save(arrays, path)

nv_read(path, device = NULL)
```

## Arguments

- arrays:

  (named `list` of
  [`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md))  
  Named list of arrays. Names must be unique.

- path:

  (`character(1)`)  
  File path to write to or read from.

- device:

  (`NULL` \| `character(1)` \|
  [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html))  
  The device on which to place the loaded arrays (`"cpu"`, `"cuda"`,
  ...) when the active backend is `"pjrt"`, defaulting to
  [`default_device()`](https://r-xla.github.io/anvl/dev/reference/default_device.md).

## Value

`nv_save()`: (`NULL`)  
Invisibly.

`nv_read()`: (named `list` of
[`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md))

## Details

These are convenience wrappers around
[`nv_serialize()`](https://r-xla.github.io/anvl/dev/reference/nv_serialize.md)
and
[`nv_unserialize()`](https://r-xla.github.io/anvl/dev/reference/nv_serialize.md)
that open and close a file connection.

## See also

[`nv_serialize()`](https://r-xla.github.io/anvl/dev/reference/nv_serialize.md),
[`nv_unserialize()`](https://r-xla.github.io/anvl/dev/reference/nv_serialize.md)

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
