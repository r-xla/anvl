# Primitive Maximum

Element-wise maximum of two arrays.

## Usage

``` r
prim_max(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Two inputs of the same data type and shape. Can be any data type. R
  values assume the other operand's data type when it is in their [data
  type category](https://r-xla.github.io/anvl/dev/reference/dtypes.md),
  and their [default data
  type](https://r-xla.github.io/anvl/dev/reference/default_dtypes.md)
  when neither operand has one.

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

[`nv_max()`](https://r-xla.github.io/anvl/dev/reference/nv_max.md)

## Examples

``` r
# two R values: both take an R double's default data type
prim_max(1, 5)
#> AnvlArray
#>  5
#> [ CPUf32{} ] 

# the R value is built at the array's data type instead
prim_max(1, nv_scalar(5, "f64"))
#> AnvlArray
#>  5
#> [ CPUf64{} ] 
```
