# Primitive Parallel Maximum

Element-wise maximum of two arrays, like
[`base::pmax()`](https://rdrr.io/r/base/Extremes.html).

## Usage

``` r
prim_pmax(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two inputs of the same data type and shape. Can be any data type. R
  values take the other operand's data type when it is in their [data
  type category](https://r-xla.github.io/anvl/dev/reference/dtypes.md),
  and their [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when neither operand has one. An R value outside the other operand's
  category is an error, as are two R values of different storage types.

## Value

([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
Has the inputs' shape and data type.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`hlo_maximum()`](https://r-xla.github.io/stablehlo/reference/hlo_maximum.html),
specified under [maximum](https://openxla.org/stablehlo/spec#maximum).

## See also

[`nv_pmax()`](https://r-xla.github.io/anvl/dev/reference/nv_pmax.md)

## Examples

``` r
# two R values: both take an R double's default data type
prim_pmax(1, 5)
#> AnvlArray
#>  5
#> [ CPUf32{} ] 

# the R value is built at the array's data type instead
prim_pmax(1, nv_scalar(5, "f64"))
#> AnvlArray
#>  5
#> [ CPUf64{} ] 
```
