# Primitive Concatenate

Concatenates arrays along an axis.

## Usage

``` r
prim_concatenate(..., axis)
```

## Arguments

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Arrays to concatenate. Must all have the same data type, naxes, and
  shape except along `axis`.

- axis:

  (`integer(1)`)  
  Axis along which to concatenate. Negative values count from the end,
  i.e. `-1` refers to the last axis.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same data type as the inputs. The output shape matches the
inputs in all axes except `axis`, which is the sum of the input sizes
along that axis. It is ambiguous if all inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_concatenate()`](https://r-xla.github.io/stablehlo/reference/hlo_concatenate.html).

## See also

[`nv_concatenate()`](https://r-xla.github.io/anvl/reference/nv_concatenate.md)

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(4, 5, 6))
prim_concatenate(x, y, axis = 1L)
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#> [ CPUf32{6} ] 
```
