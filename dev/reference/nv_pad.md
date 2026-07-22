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
  Input array.

- padding_value:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Scalar value to use for padding. Must have the same dtype as `x`.

- edge_padding_low:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Amount of padding to add at the start of each dimension.

- edge_padding_high:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Amount of padding to add at the end of each dimension.

- interior_padding:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Amount of padding to add between elements in each dimension. If `NULL`
  (default), no interior padding is applied.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as `x`.

## See also

[`prim_pad()`](https://r-xla.github.io/anvl/dev/reference/prim_pad.md)
for the underlying primitive.

## Examples

``` r
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
