# Primitive Pad

Pads an array with a given padding value.

## Usage

``` r
prim_pad(
  x,
  padding_value,
  edge_padding_low,
  edge_padding_high,
  interior_padding
)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- padding_value:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Scalar value to use for padding. Must have the same dtype as `x`.

- edge_padding_low:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Amount of padding to add at the start of each axis.

- edge_padding_high:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Amount of padding to add at the end of each axis.

- interior_padding:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Amount of padding to add between elements in each axis.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as `x`. For the output shape see the underlying
stablehlo documentation
([`hlo_pad()`](https://r-xla.github.io/stablehlo/reference/hlo_pad.html)).
It is ambiguous if the input is ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_pad()`](https://r-xla.github.io/stablehlo/reference/hlo_pad.html).

## Examples

``` r
x <- nv_array(c(1, 2, 3))
prim_pad(x, nv_scalar(0),
  edge_padding_low = 2L, edge_padding_high = 1L, interior_padding = 0L
)
#> AnvlArray
#>  0
#>  0
#>  1
#>  2
#>  3
#>  0
#> [ CPUf32{6} ] 
```
