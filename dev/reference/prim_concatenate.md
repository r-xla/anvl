# Primitive Concatenate

Concatenates arrays along an axis.

## Usage

``` r
prim_concatenate(..., axis)
```

## Arguments

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrays to concatenate. Can be of any data type. Must all have the same
  number of axes, and the same shape except along `axis`. All inputs
  must have the same data type. An R value among them assumes the data
  type of the others when it is in its [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md), and
  its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when none of them has one.

- axis:

  (`integer(1)`)  
  Axis along which to concatenate. Negative values count from the end,
  i.e. `-1` refers to the last axis.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the data type the inputs agreed on. The output shape matches the
inputs in all axes except `axis`, which is the sum of the input sizes
along that axis.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_concatenate()`](https://r-xla.github.io/stablehlo/reference/hlo_concatenate.html),
specified under
[concatenate](https://openxla.org/stablehlo/spec#concatenate).

## See also

[`nv_concatenate()`](https://r-xla.github.io/anvl/dev/reference/nv_concatenate.md)

## Examples

``` r
# the inputs already agree on a data type; axis 1 grows to 6
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
