# Deserialize arrays from raw bytes

Deserializes arrays from the
[safetensors](https://huggingface.co/docs/safetensors/index) format.

## Usage

``` r
nv_unserialize(con, device = NULL, backend = default_backend())
```

## Arguments

- con:

  (connection \| [`raw`](https://rdrr.io/r/base/raw.html))  
  A connection or raw vector to read from.

- device:

  (`NULL` \| `character(1)` \|
  [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html))  
  The device on which to place the loaded arrays (`"cpu"`, `"cuda"`,
  ...). Default is to use the CPU.

- backend:

  (`character(1)`)  
  Backend for the loaded arrays. Defaults to
  [`default_backend()`](https://r-xla.github.io/anvl/dev/reference/default_backend.md).

## Value

Named `list` of
[`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
objects.

## Details

The data type and shape of each array are restored from the serialized
data.

## See also

[`nv_serialize()`](https://r-xla.github.io/anvl/dev/reference/nv_serialize.md),
[`nv_save()`](https://r-xla.github.io/anvl/dev/reference/nv_save.md),
[`nv_read()`](https://r-xla.github.io/anvl/dev/reference/nv_read.md)

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2)
x
#> AnvlArray
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2,3} ] 
raw_data <- nv_serialize(list(x = x))
raw_data
#>  [1] 39 00 00 00 00 00 00 00 7b 22 78 22 3a 7b 22 73 68 61 70 65 22 3a 5b 32 2c
#> [26] 33 5d 2c 22 64 74 79 70 65 22 3a 22 49 33 32 22 2c 22 64 61 74 61 5f 6f 66
#> [51] 66 73 65 74 73 22 3a 5b 30 2c 32 34 5d 7d 7d 01 00 00 00 03 00 00 00 05 00
#> [76] 00 00 02 00 00 00 04 00 00 00 06 00 00 00
nv_unserialize(raw_data)
#> $x
#> AnvlArray
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2,3} ] 
#> 
```
