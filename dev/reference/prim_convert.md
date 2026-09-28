# Primitive Convert Data Type

Converts the elements of an array to a different data type. An R value
in the target's category (an R integer at `i8`, say) is materialized at
the target directly and checked for out-of-range or missing values. Any
other R value (e.g. an R double at an integer data type) is built at
`f64`, `i32` or `bool` according to its storage type and converted like
an array, without that check.

## Usage

``` r
prim_convert(x, dtype)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  One input. Can be any data type. An R value materializes at its
  [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md).

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Target data type. Can be any data type; the conversion is a value
  conversion, so it may lose precision or wrap around.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the given `dtype` and the input's shape.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_convert()`](https://r-xla.github.io/stablehlo/reference/hlo_convert.html),
specified under [convert](https://openxla.org/stablehlo/spec#convert).

## See also

[`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md)

## Examples

``` r
# the values are preserved, the data type changes
x <- nv_array(c(1L, 2L, 3L))
prim_convert(x, dtype = "f32")
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
