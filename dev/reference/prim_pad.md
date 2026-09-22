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
  The array to pad. Can be any data type. `x` and `padding_value` must
  have the same data type. An R value among them assumes the data type
  of the others when it is in its [data type
  category](https://r-xla.github.io/anvl/dev/reference/dtypes.md), and
  its [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when none of them has one.

- padding_value:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Scalar value to use for padding. Shares `x`'s data type.

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

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the data type the operands agreed on. Each axis grows by
`edge_padding_low + edge_padding_high`, plus `interior_padding` between
every pair of elements; negative edge padding trims (see
[`hlo_pad()`](https://r-xla.github.io/stablehlo/reference/hlo_pad.html)).

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_pad()`](https://r-xla.github.io/stablehlo/reference/hlo_pad.html),
specified under [pad](https://openxla.org/stablehlo/spec#pad).

## See also

[`nv_pad()`](https://r-xla.github.io/anvl/dev/reference/nv_pad.md)

## Examples

``` r
x <- nv_array(1:3)
# one element before, two after
prim_pad(x, 0L, edge_padding_low = 1L, edge_padding_high = 2L, interior_padding = 0L)
#> AnvlArray
#>  0
#>  1
#>  2
#>  3
#>  0
#>  0
#> [ CPUi32{6} ] 

# interior padding goes between the elements
prim_pad(x, 0L, 0L, 0L, 1L)
#> AnvlArray
#>  1
#>  0
#>  2
#>  0
#>  3
#> [ CPUi32{5} ] 

# negative edge padding trims instead
prim_pad(x, 0L, -1L, 0L, 0L)
#> AnvlArray
#>  2
#>  3
#> [ CPUi32{2} ] 

# one padding amount per axis
prim_pad(nv_matrix(1:4, nrow = 2), 0L, c(1L, 0L), c(0L, 1L), c(0L, 0L))
#> AnvlArray
#>  0 0 0
#>  1 3 0
#>  2 4 0
#> [ CPUi32{3,3} ] 

# the R padding value is built at x's data type
prim_pad(nv_array(c(1.5, 2.5), dtype = "f64"), 0, 1L, 1L, 0L)
#> AnvlArray
#>  0.0000
#>  1.5000
#>  2.5000
#>  0.0000
#> [ CPUf64{4} ] 
```
