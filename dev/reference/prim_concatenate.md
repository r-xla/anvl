# Primitive Concatenate

Concatenates arrays along a dimension.

## Usage

``` r
prim_concatenate(..., dimension)
```

## Arguments

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrays to concatenate. Must all have the same data type, ndims, and
  shape except along `dimension`.

- dimension:

  (`integer(1)`)  
  Dimension along which to concatenate (1-indexed).

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as the inputs. The output shape matches the
inputs in all dimensions except `dimension`, which is the sum of the
input sizes along that dimension. It is ambiguous if all inputs are
ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_concatenate()`](https://r-xla.github.io/stablehlo/reference/hlo_concatenate.html).

## See also

[`nv_concatenate()`](https://r-xla.github.io/anvl/dev/reference/nv_concatenate.md)

## Examples

``` r
x <- nv_array(c(1, 2, 3))
y <- nv_array(c(4, 5, 6))
prim_concatenate(x, y, dimension = 1L)
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#> [ CPUf32{6} ] 
```
