# Serialize arrays to raw bytes

Serializes a named list of arrays into the
[safetensors](https://huggingface.co/docs/safetensors/index) format.

## Usage

``` r
nv_serialize(arrays, con = NULL)
```

## Arguments

- arrays:

  (named `list` of
  [`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md))  
  Named list of arrays to serialize. Names must be unique.

- con:

  (`NULL` \| connection)  
  An optional connection to write to. If `NULL` (default), a raw vector
  is returned.

## Value

A [`raw`](https://rdrr.io/r/base/raw.html) vector if `con` is `NULL`,
otherwise `NULL` (invisibly).

## See also

[`nv_unserialize()`](https://r-xla.github.io/anvl/dev/reference/nv_unserialize.md),
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
