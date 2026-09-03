# Primitive Convert

Converts the elements of an array to a different data type.

## Usage

``` r
prim_convert(x, dtype)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- dtype:

  (`character(1)` \|
  [`DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Target data type.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the given `dtype` and the same shape as `x`.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_convert()`](https://r-xla.github.io/stablehlo/reference/hlo_convert.html).

## See also

[`nv_convert()`](https://r-xla.github.io/anvl/dev/reference/nv_convert.md)

## Examples

``` r
x <- nv_array(c(1L, 2L, 3L))
prim_convert(x, dtype = "f32")
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
