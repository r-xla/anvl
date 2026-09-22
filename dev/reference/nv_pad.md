# Pad

Pads an array with a given value at the edges and optionally between
elements.

## Usage

``` r
nv_pad(
  x,
  padding_value,
  edge_padding_low,
  edge_padding_high,
  interior_padding = NULL
)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The array to pad. Can be any data type; `padding_value` is brought to
  it.

- padding_value:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Scalar value to use for padding. It is brought to `x`'s data type: an
  R value is built at it when its category can reach it (`0L` serves an
  integer and a float `x` alike, `0` only a float one), and a value that
  already has a data type is converted unless that would narrow it – an
  `f64` padding value for an `f32` `x` is an error rather than a silent
  narrowing.

- edge_padding_low:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Amount of padding to add at the start of each axis.

- edge_padding_high:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Amount of padding to add at the end of each axis.

- interior_padding:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Amount of padding to add between elements in each axis. If `NULL`
  (default), no interior padding is applied.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has `x`'s data type. Each axis grows by
`edge_padding_low + edge_padding_high`, plus `interior_padding` between
every pair of elements; negative edge padding trims.

## See also

[`prim_pad()`](https://r-xla.github.io/anvl/dev/reference/prim_pad.md)
for the underlying primitive.

## Examples

``` r
# two zeros in front, one behind
x <- nv_array(c(1, 2, 3))
nv_pad(x, nv_scalar(0), edge_padding_low = 2L, edge_padding_high = 1L)
#> AnvlArray
#>  0
#>  0
#>  1
#>  2
#>  3
#>  0
#> [ CPUf32{6} ] 
```
